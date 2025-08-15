#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <fstream>
#include <memory>
#include <vector>
#include <chrono>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "opencv2/cudawarping.hpp"
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

using namespace nvinfer1;

class SuperPointLightGlueNode : public rclcpp::Node
{
public:
    SuperPointLightGlueNode() : Node("splg_node")
    {
        // Parameters
        declare_parameter("engine_path", "");
        declare_parameter("input_height", 400);
        declare_parameter("input_width", 640);
        declare_parameter("max_keypoints", 512);
        declare_parameter("profile_inference", false);
        declare_parameter("use_gpu_preprocessing", true);
        declare_parameter("frame_skip_mode", "every_nth"); // "every_nth", "rate_limit", "none"
        declare_parameter("frame_skip_n", 2);
        declare_parameter("max_process_rate_hz", 20.0);

        engine_path_ = get_parameter("engine_path").as_string();
        input_height_ = get_parameter("input_height").as_int();
        input_width_ = get_parameter("input_width").as_int();
        max_keypoints_ = get_parameter("max_keypoints").as_int();
        profile_inference_ = get_parameter("profile_inference").as_bool();
        use_gpu_preprocessing_ = get_parameter("use_gpu_preprocessing").as_bool();
        
        // Frame skipping parameters
        frame_skip_mode_ = get_parameter("frame_skip_mode").as_string();
        frame_skip_n_ = get_parameter("frame_skip_n").as_int();
        max_process_rate_hz_ = get_parameter("max_process_rate_hz").as_double();

        // Initialize TensorRT
        if (!initTensorRT())
        {
            RCLCPP_ERROR(get_logger(), "Failed to initialize TensorRT");
            return;
        }

        // Discover tensor names
        discoverTensorNames();

        // Allocate buffers
        allocateBuffers();

        // Initialize GPU preprocessing if enabled
        if (use_gpu_preprocessing_)
        {
            initGPUPreprocessing();
        }

        // Synchronized subscribers
        left_image_sub_.subscribe(this, "/image1");
        right_image_sub_.subscribe(this, "/image2");

        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(10), left_image_sub_, right_image_sub_);
        sync_->registerCallback(&SuperPointLightGlueNode::stereoCallback, this);

        matches_pub_ = this->create_publisher<sensor_msgs::msg::Image>("feature_matches_viz", 1);

        RCLCPP_INFO(this->get_logger(), "SuperPoint-LightGlue node initialized [%dx%d]", input_width_, input_height_);
    }

    ~SuperPointLightGlueNode()
    {
        cleanup();
    }

private:
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Image, sensor_msgs::msg::Image>;

    // ROS2 components
    message_filters::Subscriber<sensor_msgs::msg::Image> left_image_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> right_image_sub_;

    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr matches_pub_;

    // TensorRT components
    std::unique_ptr<IRuntime> runtime_;
    std::unique_ptr<ICudaEngine> engine_;
    std::unique_ptr<IExecutionContext> context_;

    // GPU memory buffers
    void *d_input_;
    void *d_output_keypoints_;
    void *d_output_matches_;
    void *d_output_scores_;

    // Host memory buffers
    std::vector<float> h_keypoints_;
    std::vector<int> h_matches_;
    std::vector<float> h_scores_;

    // GPU preprocessing buffers
    cv::cuda::GpuMat gpu_left_, gpu_right_;
    cv::cuda::GpuMat gpu_left_resized_, gpu_right_resized_;
    cv::cuda::GpuMat gpu_left_gray_, gpu_right_gray_;
    cv::cuda::GpuMat gpu_left_norm_, gpu_right_norm_;
    cv::cuda::GpuMat gpu_batch_; // Combined batch input

    // Parameters
    std::string engine_path_;
    int input_height_, input_width_;
    int max_keypoints_;
    bool profile_inference_;
    bool use_gpu_preprocessing_;
    int actual_num_matches_;
    
    // Frame skipping parameters
    std::string frame_skip_mode_;
    int frame_skip_n_;
    double max_process_rate_hz_;
    
    // Frame skipping state
    std::atomic<bool> processing_;
    std::chrono::steady_clock::time_point last_process_time_;
    int frame_counter_;

    // Tensor names (discovered from engine)
    std::string input_tensor_name_;
    std::string output_keypoints_name_;
    std::string output_matches_name_;
    std::string output_scores_name_;

    // Batch processing
    static constexpr int BATCH_SIZE = 2;
    static constexpr int CHANNELS = 1;

    cudaStream_t stream_;
    cv::cuda::Stream cv_stream_;

    class Logger : public ILogger
    {
        void log(Severity severity, const char *msg) noexcept override
        {
            if (severity <= Severity::kWARNING)
            {
                std::cout << msg << std::endl;
            }
        }
    } gLogger;

    bool initTensorRT()
    {
        runtime_ = std::unique_ptr<IRuntime>(createInferRuntime(gLogger));
        if (!runtime_)
            return false;

        std::ifstream file(engine_path_, std::ios::binary);
        if (!file.good())
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to read engine: %s", engine_path_.c_str());
            return false;
        }

        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> engine_data(size);
        file.read(engine_data.data(), size);

        engine_ = std::unique_ptr<ICudaEngine>(
            runtime_->deserializeCudaEngine(engine_data.data(), size));
        if (!engine_)
            return false;

        context_ = std::unique_ptr<IExecutionContext>(engine_->createExecutionContext());
        if (!context_)
            return false;

        cudaStreamCreate(&stream_);
        return true;
    }

    void discoverTensorNames()
    {
        int32_t nbIOTensors = engine_->getNbIOTensors();
        RCLCPP_INFO(this->get_logger(), "Engine has %d I/O tensors:", nbIOTensors);
        
        for (int32_t i = 0; i < nbIOTensors; ++i)
        {
            const char* tensorName = engine_->getIOTensorName(i);
            TensorIOMode ioMode = engine_->getTensorIOMode(tensorName);
            auto dims = engine_->getTensorShape(tensorName);
            
            std::string shape_str = "[";
            for (int j = 0; j < dims.nbDims; ++j)
            {
                if (j > 0) shape_str += ", ";
                shape_str += std::to_string(dims.d[j]);
            }
            shape_str += "]";
            
            RCLCPP_INFO(this->get_logger(), "  %s: %s, shape: %s", 
                tensorName, 
                (ioMode == TensorIOMode::kINPUT) ? "INPUT" : "OUTPUT",
                shape_str.c_str());
            
            // Assign tensor names based on exact names from your model
            if (ioMode == TensorIOMode::kINPUT)
            {
                input_tensor_name_ = tensorName;
            }
            else // OUTPUT
            {
                std::string name = tensorName;
                if (name == "keypoints")
                {
                    output_keypoints_name_ = tensorName;
                }
                else if (name == "matches")
                {
                    output_matches_name_ = tensorName;
                }
                else if (name == "mscores")
                {
                    output_scores_name_ = tensorName;
                }
            }
        }
        
        RCLCPP_INFO(this->get_logger(), "Assigned tensor names:");
        RCLCPP_INFO(this->get_logger(), "  Input: %s", input_tensor_name_.c_str());
        RCLCPP_INFO(this->get_logger(), "  Keypoints: %s", output_keypoints_name_.c_str());
        RCLCPP_INFO(this->get_logger(), "  Matches: %s", output_matches_name_.c_str());
        RCLCPP_INFO(this->get_logger(), "  Scores: %s", output_scores_name_.c_str());
    }

    void allocateBuffers()
    {
        size_t input_size = BATCH_SIZE * CHANNELS * input_height_ * input_width_ * sizeof(float);
        size_t keypoints_size = BATCH_SIZE * max_keypoints_ * 2 * sizeof(float);
        
        // Dynamic allocation based on actual output shapes
        size_t matches_size = max_keypoints_ * 3 * sizeof(int); // Conservative allocation
        size_t scores_size = max_keypoints_ * sizeof(float);   // Conservative allocation

        cudaMalloc(&d_input_, input_size);
        cudaMalloc(&d_output_keypoints_, keypoints_size);
        cudaMalloc(&d_output_matches_, matches_size);
        cudaMalloc(&d_output_scores_, scores_size);

        h_keypoints_.resize(BATCH_SIZE * max_keypoints_ * 2);
        h_matches_.resize(max_keypoints_ * 3);
        h_scores_.resize(max_keypoints_);
        
        // Initialize frame skipping
        processing_ = false;
        last_process_time_ = std::chrono::steady_clock::now();
        frame_counter_ = 0;
        
        RCLCPP_INFO(this->get_logger(), "Frame skipping: %s (N=%d, rate=%.1fHz)", 
                    frame_skip_mode_.c_str(), frame_skip_n_, max_process_rate_hz_);
    }

    void initGPUPreprocessing()
    {
        cv_stream_ = cv::cuda::Stream();

        // Pre-allocate GPU matrices
        gpu_left_.create(input_height_, input_width_, CV_8UC1);
        gpu_right_.create(input_height_, input_width_, CV_8UC1);
        gpu_left_resized_.create(input_height_, input_width_, CV_8UC1);
        gpu_right_resized_.create(input_height_, input_width_, CV_8UC1);
        gpu_left_gray_.create(input_height_, input_width_, CV_8UC1);
        gpu_right_gray_.create(input_height_, input_width_, CV_8UC1);
        gpu_left_norm_.create(input_height_, input_width_, CV_32FC1);
        gpu_right_norm_.create(input_height_, input_width_, CV_32FC1);
        gpu_batch_.create(input_height_ * BATCH_SIZE, input_width_, CV_32FC1);
    }

    void preprocessGPU(const cv::Mat &left_img, const cv::Mat &right_img)
    {
        try 
        {
            // Upload images to GPU
            gpu_left_.upload(left_img, cv_stream_);
            gpu_right_.upload(right_img, cv_stream_);

            // Resize if needed
            if (left_img.rows != input_height_ || left_img.cols != input_width_)
            {
                cv::cuda::resize(gpu_left_, gpu_left_resized_, cv::Size(input_width_, input_height_), 0, 0, cv::INTER_LINEAR, cv_stream_);
                cv::cuda::resize(gpu_right_, gpu_right_resized_, cv::Size(input_width_, input_height_), 0, 0, cv::INTER_LINEAR, cv_stream_);
            }
            else
            {
                gpu_left_resized_ = gpu_left_;
                gpu_right_resized_ = gpu_right_;
            }

            // Convert to grayscale
            if (left_img.channels() == 3)
            {
                cv::cuda::cvtColor(gpu_left_resized_, gpu_left_gray_, cv::COLOR_BGR2GRAY, 0, cv_stream_);
                cv::cuda::cvtColor(gpu_right_resized_, gpu_right_gray_, cv::COLOR_BGR2GRAY, 0, cv_stream_);
            }
            else
            {
                gpu_left_gray_ = gpu_left_resized_;
                gpu_right_gray_ = gpu_right_resized_;
            }

            // Normalize to [0,1]
            gpu_left_gray_.convertTo(gpu_left_norm_, CV_32FC1, 1.0 / 255.0, 0, cv_stream_);
            gpu_right_gray_.convertTo(gpu_right_norm_, CV_32FC1, 1.0 / 255.0, 0, cv_stream_);

            // Create batch: stack vertically for easier memory layout
            cv::cuda::GpuMat left_roi = gpu_batch_(cv::Rect(0, 0, input_width_, input_height_));
            cv::cuda::GpuMat right_roi = gpu_batch_(cv::Rect(0, input_height_, input_width_, input_height_));

            gpu_left_norm_.copyTo(left_roi, cv_stream_);
            gpu_right_norm_.copyTo(right_roi, cv_stream_);

            copyBatchToTensorRT();
        }
        catch (const cv::Exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "GPU preprocessing error: %s", e.what());
            throw;
        }
    }

    void copyBatchToTensorRT()
    {
        try 
        {
            cv::Mat batch_host;
            gpu_batch_.download(batch_host, cv_stream_);
            cv_stream_.waitForCompletion();

            // Directly copy to GPU buffer using CUDA API for better performance
            const size_t single_image_size = input_height_ * input_width_ * sizeof(float);
            
            for (int b = 0; b < BATCH_SIZE; ++b)
            {
                cv::Mat img_roi = batch_host(cv::Rect(0, b * input_height_, input_width_, input_height_));
                float* dst_ptr = static_cast<float*>(d_input_) + b * input_height_ * input_width_;
                
                cudaMemcpyAsync(dst_ptr, img_roi.data, single_image_size, 
                               cudaMemcpyHostToDevice, stream_);
            }
        }
        catch (const cv::Exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "Batch copy error: %s", e.what());
            throw;
        }
    }

    void preprocessCPU(const cv::Mat &left_img, const cv::Mat &right_img, float *output)
    {
        cv::Mat imgs[2] = {left_img, right_img};

        for (int i = 0; i < 2; ++i)
        {
            cv::Mat processed;

            // Convert to grayscale if needed
            if (imgs[i].channels() == 3)
            {
                cv::cvtColor(imgs[i], processed, cv::COLOR_BGR2GRAY);
            }
            else
            {
                processed = imgs[i];
            }

            // Resize if needed
            if (processed.rows != input_height_ || processed.cols != input_width_)
            {
                cv::resize(processed, processed, cv::Size(input_width_, input_height_));
            }

            // Normalize
            processed.convertTo(processed, CV_32F, 1.0 / 255.0);

            // Copy to output buffer (NCHW format)
            int offset = i * CHANNELS * input_height_ * input_width_;
            memcpy(output + offset, processed.data, CHANNELS * input_height_ * input_width_ * sizeof(float));
        }
    }

    void stereoCallback(
        const sensor_msgs::msg::Image::ConstSharedPtr &left_img,
        const sensor_msgs::msg::Image::ConstSharedPtr &right_img)
    {
        // Apply frame skipping based on mode
        if (!shouldProcessFrame())
        {
            return;
        }

        processing_.store(true);
        auto start = std::chrono::high_resolution_clock::now();

        try
        {
            RCLCPP_DEBUG(this->get_logger(), "Processing stereo images: %s, %s",
                left_img->header.frame_id.c_str(), right_img->header.frame_id.c_str());
            cv_bridge::CvImagePtr left_cv = cv_bridge::toCvCopy(left_img, "mono8");
            cv_bridge::CvImagePtr right_cv = cv_bridge::toCvCopy(right_img, "mono8");

            if (use_gpu_preprocessing_)
            {
                preprocessGPU(left_cv->image, right_cv->image);
            }
            else
            {
                std::vector<float> h_input(BATCH_SIZE * CHANNELS * input_height_ * input_width_);
                preprocessCPU(left_cv->image, right_cv->image, h_input.data());

                cudaMemcpyAsync(d_input_, h_input.data(),
                                h_input.size() * sizeof(float),
                                cudaMemcpyHostToDevice, stream_);
            }
            
            RCLCPP_DEBUG(this->get_logger(), "Starting inference...");
            
            // TensorRT inference with correct tensor names
            if (!input_tensor_name_.empty())
                context_->setTensorAddress(input_tensor_name_.c_str(), d_input_);
            if (!output_keypoints_name_.empty())
                context_->setTensorAddress(output_keypoints_name_.c_str(), d_output_keypoints_);
            if (!output_matches_name_.empty())
                context_->setTensorAddress(output_matches_name_.c_str(), d_output_matches_);
            if (!output_scores_name_.empty())
                context_->setTensorAddress(output_scores_name_.c_str(), d_output_scores_);
            
            if (!context_->enqueueV3(stream_))
            {
                RCLCPP_ERROR(this->get_logger(), "Failed to enqueue inference");
                processing_.store(false);
                return;
            }

            // Get actual output dimensions after inference
            auto matches_dims = context_->getTensorShape("matches");
            actual_num_matches_ = matches_dims.d[0]; // First dimension is number of matches

            // Copy results back with actual sizes
            cudaMemcpyAsync(h_keypoints_.data(), d_output_keypoints_,
                            BATCH_SIZE * max_keypoints_ * 2 * sizeof(float), 
                            cudaMemcpyDeviceToHost, stream_);
            
            if (actual_num_matches_ > 0)
            {
                cudaMemcpyAsync(h_matches_.data(), d_output_matches_,
                                actual_num_matches_ * 3 * sizeof(int), 
                                cudaMemcpyDeviceToHost, stream_);
                cudaMemcpyAsync(h_scores_.data(), d_output_scores_,
                                actual_num_matches_ * sizeof(float), 
                                cudaMemcpyDeviceToHost, stream_);
            }

            cudaStreamSynchronize(stream_);

            auto end = std::chrono::high_resolution_clock::now();
            if (profile_inference_)
            {
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                RCLCPP_INFO(this->get_logger(), "Total pipeline: %.2f ms", duration.count() / 1000.0);
            }

            processAndPublishResults(left_cv->image, right_cv->image, left_img->header);
        }
        catch (const cv_bridge::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "CV bridge exception: %s", e.what());
        }
        catch (const cv::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "OpenCV exception: %s", e.what());
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "Standard exception: %s", e.what());
        }

        processing_.store(false);
    }

    bool shouldProcessFrame()
    {
        // Always skip if still processing
        if (processing_.load())
        {
            RCLCPP_DEBUG(this->get_logger(), "Skipping frame - still processing");
            return false;
        }

        if (frame_skip_mode_ == "none")
        {
            return true;
        }
        else if (frame_skip_mode_ == "every_nth")
        {
            frame_counter_++;
            if (frame_counter_ % frame_skip_n_ == 0)
            {
                return true;
            }
            RCLCPP_DEBUG(this->get_logger(), "Skipping frame %d (every %d)", frame_counter_, frame_skip_n_);
            return false;
        }
        else if (frame_skip_mode_ == "rate_limit")
        {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_process_time_);
            double min_interval_ms = 1000.0 / max_process_rate_hz_;
            
            if (elapsed.count() >= min_interval_ms)
            {
                last_process_time_ = now;
                return true;
            }
            RCLCPP_DEBUG(this->get_logger(), "Rate limiting: %ldms < %.1fms", elapsed.count(), min_interval_ms);
            return false;
        }
        
        return true;
    }

    void processAndPublishResults(const cv::Mat &left_img, const cv::Mat &right_img,
                                  const std_msgs::msg::Header &header)
    {
        // Only process visualization if there are subscribers
        if (matches_pub_->get_subscription_count() == 0)
        {
            return;
        }

        cv::Mat viz_img;
        cv::hconcat(left_img, right_img, viz_img);

        if (viz_img.channels() == 1)
        {
            cv::cvtColor(viz_img, viz_img, cv::COLOR_GRAY2BGR);
        }

        // Parse keypoints for both images
        std::vector<cv::Point2f> keypoints_left, keypoints_right;
        parseKeypoints(keypoints_left, keypoints_right);

        // Draw keypoints
        drawKeypoints(viz_img, keypoints_left, keypoints_right);

        // Parse and draw matches
        std::vector<cv::DMatch> matches;
        parseMatches(matches, keypoints_left, keypoints_right);
        drawMatches(viz_img, keypoints_left, keypoints_right, matches);

        // Add info overlay
        drawInfoOverlay(viz_img, keypoints_left.size(), keypoints_right.size(), matches.size());

        auto viz_msg = cv_bridge::CvImage(header, "bgr8", viz_img).toImageMsg();
        matches_pub_->publish(*viz_msg);
    }

    void parseKeypoints(std::vector<cv::Point2f> &left_kpts, std::vector<cv::Point2f> &right_kpts)
    {
        left_kpts.clear();
        right_kpts.clear();

        // Parse keypoints from h_keypoints_ buffer
        // Format: [batch_size, max_keypoints, 2] where 2 = (x, y)
        const int kpts_per_image = max_keypoints_ * 2;
        RCLCPP_INFO(this->get_logger(), "Parsing keypoints %d", static_cast<int>(h_keypoints_.size()));
        // Left image keypoints (first batch)
        for (int i = 0; i < max_keypoints_; ++i)
        {
            float x = h_keypoints_[i * 2];
            float y = h_keypoints_[i * 2 + 1];
            
            // Filter out invalid keypoints (usually marked as negative or very large values)
            if (x >= 0 && y >= 0 && x < input_width_ && y < input_height_)
            {
                left_kpts.emplace_back(x, y);
            }
        }

        // Right image keypoints (second batch)
        int right_offset = kpts_per_image;
        for (int i = 0; i < max_keypoints_; ++i)
        {
            float x = h_keypoints_[right_offset + i * 2];
            float y = h_keypoints_[right_offset + i * 2 + 1];
            
            if (x >= 0 && y >= 0 && x < input_width_ && y < input_height_)
            {
                // Offset x coordinate for right image in concatenated visualization
                right_kpts.emplace_back(x + input_width_, y);
            }
        }
    }

    void parseMatches(std::vector<cv::DMatch> &matches, 
                      const std::vector<cv::Point2f> &left_kpts,
                      const std::vector<cv::Point2f> &right_kpts)
    {
        matches.clear();
        
        // Parse matches based on SP-LG model's output format:
        // matches: [N, 3] - (left_idx, right_idx, confidence_or_distance)
        // mscores: [N] - match confidence scores
        
        int num_matches = std::min(actual_num_matches_, static_cast<int>(h_scores_.size()));
        RCLCPP_INFO(this->get_logger(), "Found %d matches, actual num matches: %d", num_matches, actual_num_matches_);
        for (int i = 0; i < num_matches && matches.size() < 200; ++i)
        {
            int left_idx = h_matches_[i * 3];
            int right_idx = h_matches_[i * 3 + 1];
            float confidence = h_scores_[i]; // Use mscores output
            
            // Validate indices and confidence
            if (left_idx >= 0 && left_idx < static_cast<int>(left_kpts.size()) &&
                right_idx >= 0 && right_idx < static_cast<int>(right_kpts.size()) &&
                confidence > 0.1f) // Confidence threshold
            {
                cv::DMatch match;
                match.queryIdx = left_idx;
                match.trainIdx = right_idx;
                match.distance = 1.0f - confidence; // Convert confidence to distance
                matches.push_back(match);
            }
        }

        // Sort by confidence (lower distance = higher confidence)
        std::sort(matches.begin(), matches.end(), 
                  [](const cv::DMatch &a, const cv::DMatch &b) {
                      return a.distance < b.distance;
                  });
        
        RCLCPP_DEBUG(this->get_logger(), "Parsed %zu matches from %d candidates", 
                     matches.size(), num_matches);
    }

    void drawKeypoints(cv::Mat &viz_img, 
                       const std::vector<cv::Point2f> &left_kpts,
                       const std::vector<cv::Point2f> &right_kpts)
    {
        // Draw left keypoints in green
        for (const auto &pt : left_kpts)
        {
            cv::circle(viz_img, pt, 3, cv::Scalar(0, 255, 0), -1);
            cv::circle(viz_img, pt, 4, cv::Scalar(0, 0, 0), 1);
        }

        // Draw right keypoints in blue  
        for (const auto &pt : right_kpts)
        {
            cv::circle(viz_img, pt, 3, cv::Scalar(255, 0, 0), -1);
            cv::circle(viz_img, pt, 4, cv::Scalar(0, 0, 0), 1);
        }
    }

    void drawMatches(cv::Mat &viz_img,
                     const std::vector<cv::Point2f> &left_kpts,
                     const std::vector<cv::Point2f> &right_kpts,
                     const std::vector<cv::DMatch> &matches)
    {
        // Draw match lines with color coding based on confidence
        for (const auto &match : matches)
        {
            if (match.queryIdx < static_cast<int>(left_kpts.size()) && 
                match.trainIdx < static_cast<int>(right_kpts.size()))
            {
                cv::Point2f pt1 = left_kpts[match.queryIdx];
                cv::Point2f pt2 = right_kpts[match.trainIdx];
                
                // Color based on confidence (distance)
                cv::Scalar color;
                if (match.distance < 0.3f)
                    color = cv::Scalar(0, 255, 255); // Yellow - high confidence
                else if (match.distance < 0.6f)
                    color = cv::Scalar(0, 165, 255); // Orange - medium confidence  
                else
                    color = cv::Scalar(0, 0, 255);   // Red - low confidence
                
                cv::line(viz_img, pt1, pt2, color, 1);
                
                // Draw larger circles for matched points
                cv::circle(viz_img, pt1, 5, color, 2);
                cv::circle(viz_img, pt2, 5, color, 2);
            }
        }
    }

    void drawInfoOverlay(cv::Mat &viz_img, int left_count, int right_count, int match_count)
    {
        // Add semi-transparent background for text
        cv::Mat overlay = viz_img.clone();
        cv::rectangle(overlay, cv::Point(10, 10), cv::Point(300, 80), cv::Scalar(0, 0, 0), -1);
        cv::addWeighted(viz_img, 0.7, overlay, 0.3, 0, viz_img);
        
        // Add text info
        std::string info1 = "Left: " + std::to_string(left_count) + " pts";
        std::string info2 = "Right: " + std::to_string(right_count) + " pts";  
        std::string info3 = "Matches: " + std::to_string(match_count);
        
        cv::putText(viz_img, info1, cv::Point(15, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        cv::putText(viz_img, info2, cv::Point(15, 50), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
        cv::putText(viz_img, info3, cv::Point(15, 70), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
        
        // Add dividing line between images
        cv::line(viz_img, cv::Point(input_width_, 0), cv::Point(input_width_, viz_img.rows), cv::Scalar(255, 255, 255), 2);
    }

    void cleanup()
    {
        if (d_input_)
            cudaFree(d_input_);
        if (d_output_keypoints_)
            cudaFree(d_output_keypoints_);
        if (d_output_matches_)
            cudaFree(d_output_matches_);
        if (d_output_scores_)
            cudaFree(d_output_scores_);
        if (stream_)
            cudaStreamDestroy(stream_);
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SuperPointLightGlueNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
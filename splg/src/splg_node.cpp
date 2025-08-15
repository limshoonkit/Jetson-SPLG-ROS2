#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <fstream>
#include <memory>
#include <vector>
#include <chrono>
#include <unordered_map>
#include <algorithm>
#include <cstdint>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "opencv2/cudawarping.hpp"
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include "type_conv_helper.cuh"

using namespace nvinfer1;

// https://github.com/NVIDIA/TensorRT/issues/4120
class DynamicOutputAllocator : public nvinfer1::IOutputAllocator
{
public:
    DynamicOutputAllocator() = default;

    ~DynamicOutputAllocator()
    {
        if (mBuffer)
        {
            cudaFree(mBuffer);
        }
    }

    // FIX: Removed unused parameter name to silence warning
    void notifyShape(const char * /*tensorName*/, const nvinfer1::Dims &dims) noexcept override
    {
        mShape = dims;
    }

    void *reallocateOutputAsync(const char *tensorName, void * /*currentMemory*/, uint64_t size, uint64_t /*alignment*/, cudaStream_t /*stream*/) noexcept override
    {
        if (size > mSize)
        {
            if (mBuffer)
            {
                cudaFree(mBuffer);
            }
            if (cudaMalloc(&mBuffer, size) != cudaSuccess)
            {
                std::cerr << "ERROR: Failed to allocate GPU memory for output tensor " << tensorName << std::endl;
                mSize = 0;
                mBuffer = nullptr;
                return nullptr;
            }
            mSize = size;
        }
        return mBuffer;
    }

    void *getBuffer() const
    {
        return mBuffer;
    }

    nvinfer1::Dims getShape() const
    {
        return mShape;
    }

private:
    void *mBuffer{nullptr};
    uint64_t mSize{0};
    nvinfer1::Dims mShape{};
};

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
        frame_skip_mode_ = get_parameter("frame_skip_mode").as_string();
        frame_skip_n_ = get_parameter("frame_skip_n").as_int();
        max_process_rate_hz_ = get_parameter("max_process_rate_hz").as_double();

        if (!initTensorRT())
        {
            RCLCPP_ERROR(get_logger(), "Failed to initialize TensorRT");
            return;
        }

        discoverTensorNames();
        allocateBuffers();

        setupBindingsAndAllocators();

        if (use_gpu_preprocessing_)
        {
            initGPUPreprocessing();
        }

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

    message_filters::Subscriber<sensor_msgs::msg::Image> left_image_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> right_image_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr matches_pub_;

    std::unique_ptr<IRuntime> runtime_;
    std::unique_ptr<ICudaEngine> engine_;
    std::unique_ptr<IExecutionContext> context_;

    // GPU memory buffers
    void *d_input_;
    void *d_output_keypoints_fp16_; // Direct FP16 output from TensorRT
    void *d_output_keypoints_fp32_; // Buffer for converted FP32 keypoints

    // Map to manage the lifetime of dynamic allocator objects
    std::unordered_map<std::string, std::unique_ptr<DynamicOutputAllocator>> mAllocatorMap;

    // Host memory buffers
    std::vector<float> h_keypoints_;
    std::vector<int> h_matches_;
    std::vector<float> h_scores_;

    cv::cuda::GpuMat gpu_left_, gpu_right_;
    cv::cuda::GpuMat gpu_left_resized_, gpu_right_resized_;
    cv::cuda::GpuMat gpu_left_gray_, gpu_right_gray_;
    cv::cuda::GpuMat gpu_left_norm_, gpu_right_norm_;
    cv::cuda::GpuMat gpu_batch_;

    std::string engine_path_;
    int input_height_, input_width_;
    int max_keypoints_;
    bool profile_inference_;
    bool use_gpu_preprocessing_;
    int actual_num_matches_;

    std::string frame_skip_mode_;
    int frame_skip_n_;
    double max_process_rate_hz_;

    std::atomic<bool> processing_;
    std::chrono::steady_clock::time_point last_process_time_;
    int frame_counter_;

    std::string input_tensor_name_;
    std::string output_keypoints_name_;
    std::string output_matches_name_;
    std::string output_scores_name_;

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
            const char *tensorName = engine_->getIOTensorName(i);
            TensorIOMode ioMode = engine_->getTensorIOMode(tensorName);
            auto dims = engine_->getTensorShape(tensorName);

            std::string shape_str = "[";
            for (int j = 0; j < dims.nbDims; ++j)
            {
                if (j > 0)
                    shape_str += ", ";
                shape_str += std::to_string(dims.d[j]);
            }
            shape_str += "]";

            RCLCPP_INFO(this->get_logger(), "  %s: %s, shape: %s",
                        tensorName,
                        (ioMode == TensorIOMode::kINPUT) ? "INPUT" : "OUTPUT",
                        shape_str.c_str());

            if (ioMode == TensorIOMode::kINPUT)
            {
                input_tensor_name_ = tensorName;
            }
            else
            {
                std::string name = tensorName;
                if (name == "keypoints")
                    output_keypoints_name_ = tensorName;
                else if (name == "matches")
                    output_matches_name_ = tensorName;
                else if (name == "mscores")
                    output_scores_name_ = tensorName;
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
        cudaMalloc(&d_input_, input_size);

        // Allocate for FP16 output from TensorRT
        size_t keypoints_size_fp16 = BATCH_SIZE * max_keypoints_ * 2 * sizeof(uint16_t);
        cudaMalloc(&d_output_keypoints_fp16_, keypoints_size_fp16);

        // Allocate for FP32 converted keypoints
        size_t keypoints_size_fp32 = BATCH_SIZE * max_keypoints_ * 2 * sizeof(float);
        cudaMalloc(&d_output_keypoints_fp32_, keypoints_size_fp32);

        // Host buffers
        h_keypoints_.resize(BATCH_SIZE * max_keypoints_ * 2);
        h_matches_.resize(max_keypoints_ * 3);
        h_scores_.resize(max_keypoints_);

        processing_ = false;
        last_process_time_ = std::chrono::steady_clock::now();
        frame_counter_ = 0;

        RCLCPP_INFO(this->get_logger(), "Frame skipping: %s (N=%d, rate=%.1fHz)",
                    frame_skip_mode_.c_str(), frame_skip_n_, max_process_rate_hz_);
    }

    void setupBindingsAndAllocators()
    {
        context_->setTensorAddress(input_tensor_name_.c_str(), d_input_);
        // Point the engine output to the FP16 buffer
        context_->setTensorAddress(output_keypoints_name_.c_str(), d_output_keypoints_fp16_);

        for (const auto &name : {output_matches_name_, output_scores_name_})
        {
            if (!name.empty())
            {
                RCLCPP_INFO(this->get_logger(), "Setting up dynamic allocator for: %s", name.c_str());
                auto allocator = std::make_unique<DynamicOutputAllocator>();
                context_->setOutputAllocator(name.c_str(), allocator.get());
                mAllocatorMap.emplace(name, std::move(allocator));
            }
        }
    }

    void initGPUPreprocessing()
    {
        cv_stream_ = cv::cuda::Stream();
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
        gpu_left_.upload(left_img, cv_stream_);
        gpu_right_.upload(right_img, cv_stream_);

        cv::cuda::resize(gpu_left_, gpu_left_resized_, cv::Size(input_width_, input_height_), 0, 0, cv::INTER_LINEAR, cv_stream_);
        cv::cuda::resize(gpu_right_, gpu_right_resized_, cv::Size(input_width_, input_height_), 0, 0, cv::INTER_LINEAR, cv_stream_);

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

        gpu_left_gray_.convertTo(gpu_left_norm_, CV_32FC1, 1.0 / 255.0, 0, cv_stream_);
        gpu_right_gray_.convertTo(gpu_right_norm_, CV_32FC1, 1.0 / 255.0, 0, cv_stream_);

        gpu_left_norm_.copyTo(gpu_batch_(cv::Rect(0, 0, input_width_, input_height_)), cv_stream_);
        gpu_right_norm_.copyTo(gpu_batch_(cv::Rect(0, input_height_, input_width_, input_height_)), cv_stream_);

        cv::Mat batch_host;
        gpu_batch_.download(batch_host, cv_stream_);
        cv_stream_.waitForCompletion();

        cudaMemcpyAsync(d_input_, batch_host.data, batch_host.total() * batch_host.elemSize(), cudaMemcpyHostToDevice, stream_);
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
        const sensor_msgs::msg::Image::ConstSharedPtr &left_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr &right_msg)
    {
        if (!shouldProcessFrame())
            return;

        processing_.store(true);
        auto start = std::chrono::high_resolution_clock::now();

        try
        {
            cv_bridge::CvImagePtr left_cv = cv_bridge::toCvCopy(left_msg, "bgr8");
            cv_bridge::CvImagePtr right_cv = cv_bridge::toCvCopy(right_msg, "bgr8");

            if (use_gpu_preprocessing_)
            {
                preprocessGPU(left_cv->image, right_cv->image);
            }
            else
            {
                std::vector<float> h_input(BATCH_SIZE * CHANNELS * input_height_ * input_width_);
                preprocessCPU(left_cv->image, right_cv->image, h_input.data());
                cudaMemcpyAsync(d_input_, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice, stream_);
            }

            if (!context_->enqueueV3(stream_))
            {
                RCLCPP_ERROR(this->get_logger(), "Failed to enqueue inference");
                processing_.store(false);
                return;
            }

            const int total_keypoint_elements = BATCH_SIZE * max_keypoints_ * 2;
            launchConvertFP16ToFP32(
                reinterpret_cast<const __half *>(d_output_keypoints_fp16_),
                reinterpret_cast<float *>(d_output_keypoints_fp32_),
                total_keypoint_elements, stream_);

            // Add error checking
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                RCLCPP_ERROR(this->get_logger(), "CUDA kernel error: %s", cudaGetErrorString(err));
                processing_.store(false);
                return;
            }

            auto &matches_allocator = mAllocatorMap.at(output_matches_name_);
            auto matches_dims = matches_allocator->getShape();
            actual_num_matches_ = matches_dims.nbDims > 0 ? matches_dims.d[0] : 0;
            void *d_matches_ptr = matches_allocator->getBuffer();

            auto &scores_allocator = mAllocatorMap.at(output_scores_name_);
            void *d_scores_ptr = scores_allocator->getBuffer();

            cudaMemcpyAsync(h_keypoints_.data(), d_output_keypoints_fp32_,
                            BATCH_SIZE * max_keypoints_ * 2 * sizeof(float),
                            cudaMemcpyDeviceToHost, stream_);

            if (actual_num_matches_ > 0)
            {
                if (actual_num_matches_ > static_cast<int>(h_matches_.size() / 3))
                {
                    RCLCPP_WARN(this->get_logger(), "Model produced more matches (%d) than host buffer capacity (%zu). Truncating.", actual_num_matches_, h_matches_.size() / 3);
                    actual_num_matches_ = h_matches_.size() / 3;
                }
                cudaMemcpyAsync(h_matches_.data(), d_matches_ptr, actual_num_matches_ * 3 * sizeof(int), cudaMemcpyDeviceToHost, stream_);
                cudaMemcpyAsync(h_scores_.data(), d_scores_ptr, actual_num_matches_ * sizeof(float), cudaMemcpyDeviceToHost, stream_);
            }

            cudaStreamSynchronize(stream_);
            auto end = std::chrono::high_resolution_clock::now();
            if (profile_inference_)
            {
                RCLCPP_INFO(this->get_logger(), "Total pipeline: %.2f ms", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0);
            }

            processAndPublishResults(left_cv->image, right_cv->image, left_msg->header);
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "Exception in stereoCallback: %s", e.what());
        }

        processing_.store(false);
    }

    bool shouldProcessFrame()
    {
        if (processing_.load())
            return false;
        if (frame_skip_mode_ == "none")
            return true;

        if (frame_skip_mode_ == "every_nth")
        {
            frame_counter_++;
            return (frame_counter_ % frame_skip_n_ == 0);
        }
        else if (frame_skip_mode_ == "rate_limit")
        {
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_process_time_).count() >= (1000.0 / max_process_rate_hz_))
            {
                last_process_time_ = now;
                return true;
            }
            return false;
        }
        return true;
    }

    void processAndPublishResults(const cv::Mat &left_img, const cv::Mat &right_img,
                                  const std_msgs::msg::Header &header)
    {
        if (matches_pub_->get_subscription_count() == 0)
            return;

        // Calculate scaling factors to draw on original image size
        const float scale_x = static_cast<float>(left_img.cols) / input_width_;
        const float scale_y = static_cast<float>(left_img.rows) / input_height_;

        cv::Mat viz_img;
        cv::hconcat(left_img, right_img, viz_img);
        if (viz_img.channels() == 1)
            cv::cvtColor(viz_img, viz_img, cv::COLOR_GRAY2BGR);

        std::vector<cv::Point2f> kpts_left_raw, kpts_right_raw;
        parseKeypoints(kpts_left_raw, kpts_right_raw);

        std::vector<cv::DMatch> matches;
        parseMatches(matches, kpts_left_raw, kpts_right_raw);

        // Draw all valid keypoints
        drawKeypoints(viz_img, kpts_left_raw, kpts_right_raw, scale_x, scale_y, left_img.cols);

        // Draw matches on top
        drawMatches(viz_img, kpts_left_raw, kpts_right_raw, matches, scale_x, scale_y, left_img.cols);

        int valid_left_kpts = std::count_if(kpts_left_raw.begin(), kpts_left_raw.end(), [](const cv::Point2f &pt)
                                            { return pt.x >= 0; });
        int valid_right_kpts = std::count_if(kpts_right_raw.begin(), kpts_right_raw.end(), [](const cv::Point2f &pt)
                                             { return pt.x >= 0; });

        drawInfoOverlay(viz_img, valid_left_kpts, valid_right_kpts, matches.size());

        auto viz_msg = cv_bridge::CvImage(header, "bgr8", viz_img).toImageMsg();
        matches_pub_->publish(*viz_msg);
    }

    void parseKeypoints(std::vector<cv::Point2f> &left_kpts, std::vector<cv::Point2f> &right_kpts)
    {
        // Create full-sized vectors. Invalid keypoints will have negative coords from the model.
        left_kpts.resize(max_keypoints_);
        right_kpts.resize(max_keypoints_);

        const int kpts_per_image = max_keypoints_ * 2;

        for (int i = 0; i < max_keypoints_; ++i)
        {
            left_kpts[i] = cv::Point2f(h_keypoints_[i * 2], h_keypoints_[i * 2 + 1]);
        }
        for (int i = 0; i < max_keypoints_; ++i)
        {
            right_kpts[i] = cv::Point2f(h_keypoints_[kpts_per_image + i * 2], h_keypoints_[kpts_per_image + i * 2 + 1]);
        }
    }

    void parseMatches(std::vector<cv::DMatch> &matches,
                      const std::vector<cv::Point2f> &left_kpts,
                      const std::vector<cv::Point2f> &right_kpts)
    {
        matches.clear();
        int num_candidates = std::min(actual_num_matches_, static_cast<int>(h_scores_.size()));

        for (int i = 0; i < num_candidates; ++i)
        {
            int left_idx = h_matches_[i * 3];
            int right_idx = h_matches_[i * 3 + 1];
            float confidence = h_scores_[i];

            // Check if indices are valid and if the keypoints they refer to are also valid (not filtered out)
            if (left_idx >= 0 && left_idx < static_cast<int>(left_kpts.size()) &&
                right_idx >= 0 && right_idx < static_cast<int>(right_kpts.size()) &&
                left_kpts[left_idx].x >= 0 && right_kpts[right_idx].x >= 0 &&
                confidence > 0.1f) // Confidence threshold
            {
                matches.emplace_back(left_idx, right_idx, 1.0f - confidence);
            }
        }
        std::sort(matches.begin(), matches.end(), [](const cv::DMatch &a, const cv::DMatch &b)
                  { return a.distance < b.distance; });
        RCLCPP_DEBUG(this->get_logger(), "Parsed %zu matches from %d candidates", matches.size(), num_candidates);
    }

    void drawKeypoints(cv::Mat &viz_img,
                       const std::vector<cv::Point2f> &left_kpts,
                       const std::vector<cv::Point2f> &right_kpts,
                       float scale_x, float scale_y, int left_img_width)
    {
        for (const auto &pt_raw : left_kpts)
        {
            if (pt_raw.x < 0)
                continue; // Skip invalid keypoints
            cv::Point2f pt(pt_raw.x * scale_x, pt_raw.y * scale_y);
            cv::circle(viz_img, pt, 2, cv::Scalar(0, 255, 0), -1);
        }

        for (const auto &pt_raw : right_kpts)
        {
            if (pt_raw.x < 0)
                continue;
            cv::Point2f pt(pt_raw.x * scale_x + left_img_width, pt_raw.y * scale_y);
            cv::circle(viz_img, pt, 2, cv::Scalar(255, 0, 0), -1);
        }
    }

    void drawMatches(cv::Mat &viz_img,
                     const std::vector<cv::Point2f> &left_kpts,
                     const std::vector<cv::Point2f> &right_kpts,
                     const std::vector<cv::DMatch> &matches,
                     float scale_x, float scale_y, int left_img_width)
    {
        for (const auto &match : matches)
        {
            cv::Point2f pt1_raw = left_kpts[match.queryIdx];
            cv::Point2f pt2_raw = right_kpts[match.trainIdx];

            cv::Point2f pt1(pt1_raw.x * scale_x, pt1_raw.y * scale_y);
            cv::Point2f pt2(pt2_raw.x * scale_x + left_img_width, pt2_raw.y * scale_y);

            cv::Scalar color;
            if (match.distance < 0.3f)
                color = cv::Scalar(0, 255, 255); // Yellow - high confidence
            else if (match.distance < 0.6f)
                color = cv::Scalar(0, 165, 255); // Orange - medium confidence
            else
                color = cv::Scalar(0, 0, 255); // Red - low confidence

            cv::line(viz_img, pt1, pt2, color, 1);
            cv::circle(viz_img, pt1, 3, color, 1);
            cv::circle(viz_img, pt2, 3, color, 1);
        }
    }

    void drawInfoOverlay(cv::Mat &viz_img, int left_count, int right_count, int match_count)
    {
        cv::rectangle(viz_img, cv::Point(5, 5), cv::Point(200, 75), cv::Scalar(0, 0, 0), -1);
        cv::addWeighted(viz_img(cv::Rect(5, 5, 195, 70)), 0.5, viz_img(cv::Rect(5, 5, 195, 70)), 0.5, 0, viz_img(cv::Rect(5, 5, 195, 70)));

        cv::putText(viz_img, "Left kpts: " + std::to_string(left_count), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        cv::putText(viz_img, "Right kpts: " + std::to_string(right_count), cv::Point(10, 45), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
        cv::putText(viz_img, "Matches: " + std::to_string(match_count), cv::Point(10, 65), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);

        cv::line(viz_img, cv::Point(viz_img.cols / 2, 0), cv::Point(viz_img.cols / 2, viz_img.rows), cv::Scalar(255, 255, 255), 1);
    }

    void cleanup()
    {
        if (d_input_)
            cudaFree(d_input_);
        if (d_output_keypoints_fp16_)
            cudaFree(d_output_keypoints_fp16_);
        if (d_output_keypoints_fp32_)
            cudaFree(d_output_keypoints_fp32_);
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
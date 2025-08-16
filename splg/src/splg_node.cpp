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
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include "type_conv_helper.cuh"

using namespace nvinfer1;

#define CUDA_CHECK(call)                                                                                                     \
    do                                                                                                                       \
    {                                                                                                                        \
        cudaError_t error = call;                                                                                            \
        if (error != cudaSuccess)                                                                                            \
        {                                                                                                                    \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1);                                                                                                         \
        }                                                                                                                    \
    } while (0)

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
        declare_parameter("use_unified_memory", false); // For Jetson Orin optimization

        engine_path_ = get_parameter("engine_path").as_string();
        input_height_ = get_parameter("input_height").as_int();
        input_width_ = get_parameter("input_width").as_int();
        max_keypoints_ = get_parameter("max_keypoints").as_int();
        profile_inference_ = get_parameter("profile_inference").as_bool();
        use_gpu_preprocessing_ = get_parameter("use_gpu_preprocessing").as_bool();
        frame_skip_mode_ = get_parameter("frame_skip_mode").as_string();
        frame_skip_n_ = get_parameter("frame_skip_n").as_int();
        max_process_rate_hz_ = get_parameter("max_process_rate_hz").as_double();
        use_unified_memory_ = get_parameter("use_unified_memory").as_bool();

        if (!initTensorRT())
        {
            RCLCPP_ERROR(get_logger(), "Failed to initialize TensorRT");
            rclcpp::shutdown();
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
        RCLCPP_INFO(this->get_logger(), "SuperPoint-LightGlue node initialized [%dx%d] GPU:%s Unified:%s",
                    input_width_, input_height_,
                    use_gpu_preprocessing_ ? "ON" : "OFF",
                    use_unified_memory_ ? "ON" : "OFF");
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
    void *d_input_;            // Input tensor
    void *d_output_keypoints_; // Output keypoints tensor
    std::unordered_map<std::string, std::unique_ptr<DynamicOutputAllocator>> mAllocatorMap;

    // Host memory buffers for results
    std::vector<uint64_t> h_keypoints_;
    std::vector<uint64_t> h_matches_;
    std::vector<float> h_scores_;

    // GPU Preprocessing Mats with pre-allocation
    cv::cuda::GpuMat gpu_left_uploaded_, gpu_right_uploaded_;
    cv::cuda::GpuMat gpu_left_resized_, gpu_right_resized_;
    cv::cuda::GpuMat gpu_left_gray_, gpu_right_gray_;
    cv::cuda::GpuMat gpu_left_norm_fp32_, gpu_right_norm_fp32_;

    std::string engine_path_;
    int input_height_, input_width_;
    int max_keypoints_;
    bool profile_inference_;
    bool use_gpu_preprocessing_;
    bool use_unified_memory_;
    int actual_num_matches_;
    int actual_num_keypoints_;

    std::string frame_skip_mode_;
    int frame_skip_n_;
    double max_process_rate_hz_;

    std::atomic<bool> processing_{false};
    std::chrono::steady_clock::time_point last_process_time_;
    int frame_counter_{0};

    std::string input_tensor_name_;     // FLOAT32
    std::string output_keypoints_name_; // INT64
    std::string output_matches_name_;   // INT64
    std::string output_scores_name_;    // FLOAT32

    static constexpr int BATCH_SIZE = 2;
    static constexpr int CHANNELS = 1;

    cudaStream_t stream_;
    // cv::cuda::Stream cv_stream_;

    // Performance monitoring
    std::chrono::high_resolution_clock::time_point preprocess_start_, inference_start_;

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

        engine_ = std::unique_ptr<ICudaEngine>(runtime_->deserializeCudaEngine(engine_data.data(), size));
        if (!engine_)
            return false;

        context_ = std::unique_ptr<IExecutionContext>(engine_->createExecutionContext());
        if (!context_)
            return false;

        CUDA_CHECK(cudaStreamCreate(&stream_));
        return true;
    }

    void discoverTensorNames()
    {
        int32_t nbIOTensors = engine_->getNbIOTensors();
        for (int32_t i = 0; i < nbIOTensors; ++i)
        {
            const char *tensorName = engine_->getIOTensorName(i);
            if (engine_->getTensorIOMode(tensorName) == TensorIOMode::kINPUT)
            {
                input_tensor_name_ = tensorName;
            }
            else
            {
                std::string name = tensorName;
                if (name == "keypoints")
                    output_keypoints_name_ = name;
                else if (name == "matches")
                    output_matches_name_ = name;
                else if (name == "mscores")
                    output_scores_name_ = name;
            }
        }
        RCLCPP_INFO(this->get_logger(), "Input: %s, Keypoints: %s, Matches: %s, Scores: %s",
                    input_tensor_name_.c_str(), output_keypoints_name_.c_str(),
                    output_matches_name_.c_str(), output_scores_name_.c_str());
    }

    void allocateBuffers()
    {
        // Allocate buffer for input and output tensors
        size_t input_size = BATCH_SIZE * CHANNELS * input_height_ * input_width_ * sizeof(float); // NCHW (2, 1, H, W)
        size_t output_keypoints_size = BATCH_SIZE * max_keypoints_ * 2 * sizeof(uint64_t);           // (2, max_keypoints, 2)

        if (use_unified_memory_)
        {
            CUDA_CHECK(cudaMallocManaged(&d_input_, input_size));
            CUDA_CHECK(cudaMallocManaged(&d_output_keypoints_, output_keypoints_size));
        }
        else
        {
            CUDA_CHECK(cudaMalloc(&d_input_, input_size));
            CUDA_CHECK(cudaMalloc(&d_output_keypoints_, output_keypoints_size));
        }

        // Host buffers for results - pre-allocate maximum sizes
        h_keypoints_.resize(BATCH_SIZE * max_keypoints_ * 2);
        h_matches_.resize(max_keypoints_ * 3); // (-1, 3)
        h_scores_.resize(max_keypoints_);      // (-1)

        last_process_time_ = std::chrono::steady_clock::now();
        RCLCPP_INFO(this->get_logger(), "Frame skipping: %s (N=%d, rate=%.1fHz)",
                    frame_skip_mode_.c_str(), frame_skip_n_, max_process_rate_hz_);
    }

    void setupBindingsAndAllocators()
    {
        if (profile_inference_)
        {
            auto input_shape = engine_->getTensorShape(input_tensor_name_.c_str());
            if (input_shape.nbDims == 4)
                RCLCPP_WARN(this->get_logger(), "Input shape: %ld, %ld, %ld, %ld", input_shape.d[0],
                            input_shape.d[1], input_shape.d[2], input_shape.d[3]);

            auto output_keypoints_shape = engine_->getTensorShape(output_keypoints_name_.c_str());
            if (output_keypoints_shape.nbDims == 3)
                RCLCPP_WARN(this->get_logger(), "Output keypoints shape: %ld, %ld, %ld", output_keypoints_shape.d[0],
                            output_keypoints_shape.d[1], output_keypoints_shape.d[2]);

            auto output_matches_shape = engine_->getTensorShape(output_matches_name_.c_str());
            if (output_matches_shape.nbDims == 2)
                RCLCPP_WARN(this->get_logger(), "Output matches shape: %ld, %ld", output_matches_shape.d[0],
                            output_matches_shape.d[1]);

            auto output_scores_shape = engine_->getTensorShape(output_scores_name_.c_str());
            if (output_scores_shape.nbDims == 1)
                RCLCPP_WARN(this->get_logger(), "Output scores shape: %ld", output_scores_shape.d[0]);
        }

        // Set fixed input tensor address and shape
        context_->setInputTensorAddress(input_tensor_name_.c_str(), d_input_);
        context_->setInputShape(input_tensor_name_.c_str(),
                                nvinfer1::Dims4{BATCH_SIZE, CHANNELS, input_height_, input_width_});

        // Set output tensor address and shape
        context_->setOutputTensorAddress(output_keypoints_name_.c_str(), d_output_keypoints_);

        // Use dynamic allocators for all outputs
        for (const auto &name : {output_keypoints_name_, output_matches_name_, output_scores_name_})
        {
            if (!name.empty())
            {
                auto allocator = std::make_unique<DynamicOutputAllocator>();
                context_->setOutputAllocator(name.c_str(), allocator.get());
                mAllocatorMap.emplace(name, std::move(allocator));
            }
        }
    }

    void initGPUPreprocessing()
    {
    }

    void preprocessGPU(const cv::Mat &left_img, const cv::Mat &right_img, bool is_grayscale)
    {
        if (profile_inference_)
        {
            preprocess_start_ = std::chrono::high_resolution_clock::now();
        }

        // Ensure images are the expected size and format
        CV_Assert(left_img.rows == input_height_ && left_img.cols == input_width_);
        CV_Assert(right_img.rows == input_height_ && right_img.cols == input_width_);
        CV_Assert(left_img.type() == CV_8UC1 && right_img.type() == CV_8UC1);

        cv::cuda::GpuMat d_left, d_right;
        d_left.upload(left_img);
        d_right.upload(right_img);
        launchToNCHW(d_left, d_right, d_input_, input_height_, input_width_);

        CUDA_CHECK(cudaGetLastError());

        if (!use_unified_memory_)
        {
            // Synchronize to ensure kernel completion before inference
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        if (profile_inference_)
        {
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            auto preprocess_end = std::chrono::high_resolution_clock::now();
            auto preprocess_time = std::chrono::duration_cast<std::chrono::microseconds>(preprocess_end - preprocess_start_).count() / 1000.0;
            RCLCPP_INFO(this->get_logger(), "GPU Preprocessing: %.2f ms", preprocess_time);
        }
    }

    void preprocessCPU(const cv::Mat &left_img, const cv::Mat &right_img, float *output, bool is_grayscale)
    {
        if (profile_inference_)
        {
            preprocess_start_ = std::chrono::high_resolution_clock::now();
        }

        // TODO: Implement CPU preprocessing

        if (profile_inference_)
        {
            auto preprocess_end = std::chrono::high_resolution_clock::now();
            auto preprocess_time = std::chrono::duration_cast<std::chrono::microseconds>(preprocess_end - preprocess_start_).count() / 1000.0;
            RCLCPP_INFO(this->get_logger(), "CPU Preprocessing: %.2f ms", preprocess_time);
        }
    }

    void stereoCallback(
        const sensor_msgs::msg::Image::ConstSharedPtr &left_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr &right_msg)
    {
        if (!shouldProcessFrame())
            return;
        processing_.store(true);
        auto pipeline_start = std::chrono::high_resolution_clock::now();

        try
        {
            bool is_grayscale = (left_msg->encoding == "mono8");
            const char *desired_encoding = is_grayscale ? "mono8" : "bgr8";

            cv_bridge::CvImagePtr left_cv = cv_bridge::toCvCopy(left_msg, desired_encoding);
            cv_bridge::CvImagePtr right_cv = cv_bridge::toCvCopy(right_msg, desired_encoding);

            if (left_cv->image.empty() || right_cv->image.empty())
            {
                RCLCPP_WARN(get_logger(), "Received empty image(s), skipping frame.");
                processing_.store(false);
                return;
            }

            // Preprocessing
            if (use_gpu_preprocessing_)
            {
                preprocessGPU(left_cv->image, right_cv->image, is_grayscale);
            }
            else
            {
                std::vector<float> h_input(BATCH_SIZE * CHANNELS * input_height_ * input_width_);
                preprocessCPU(left_cv->image, right_cv->image, h_input.data(), is_grayscale);
                CUDA_CHECK(cudaMemcpyAsync(d_input_, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice, stream_));
            }

            // Inference
            if (profile_inference_)
            {
                inference_start_ = std::chrono::high_resolution_clock::now();
            }

            if (!context_->enqueueV3(stream_))
            {
                RCLCPP_ERROR(this->get_logger(), "Failed to enqueue inference");
                processing_.store(false);
                return;
            }

            // TODO: Handle output tensors

            if (profile_inference_)
            {
                CUDA_CHECK(cudaStreamSynchronize(stream_));
                auto inference_end = std::chrono::high_resolution_clock::now();
                auto inference_time = std::chrono::duration_cast<std::chrono::microseconds>(inference_end - inference_start_).count() / 1000.0;
                RCLCPP_INFO(this->get_logger(), "Inference: %.2f ms", inference_time);
            }

            auto pipeline_end = std::chrono::high_resolution_clock::now();
            if (profile_inference_)
            {
                auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(pipeline_end - pipeline_start).count() / 1000.0;
                RCLCPP_INFO(this->get_logger(), "Total pipeline: %.2f ms, Keypoints: %d, Matches: %d",
                            total_time, actual_num_keypoints_, actual_num_matches_);
            }

            processAndPublishResults(left_cv->image, right_cv->image, left_msg->header);
        }
        catch (const cv_bridge::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
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
        if (frame_skip_mode_ == "rate_limit")
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

    void parseKeypoints(std::vector<cv::Point2f> &left_kpts, std::vector<cv::Point2f> &right_kpts)
    {
        left_kpts.resize(max_keypoints_);
        right_kpts.resize(max_keypoints_);
        const int kpts_per_image = max_keypoints_ * 2;
        for (int i = 0; i < max_keypoints_; ++i)
        {
            left_kpts[i] = cv::Point2f(h_keypoints_[i * 2], h_keypoints_[i * 2 + 1]);
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

            if (left_idx >= 0 && left_idx < static_cast<int>(left_kpts.size()) &&
                right_idx >= 0 && right_idx < static_cast<int>(right_kpts.size()) &&
                left_kpts[left_idx].x >= 0 && right_kpts[right_idx].x >= 0 &&
                confidence > 0.1f)
            {
                matches.emplace_back(left_idx, right_idx, 1.0f - confidence);
            }
        }
    }

    cv::Scalar getMatchColor(float confidence)
    {
        float hue = std::min(1.0f - confidence, 1.0f) * 60.0f; // Red (bad) to Green (good)
        cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 255, 255));
        cv::Mat bgr;
        cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
        return cv::Scalar(bgr.at<cv::Vec3b>(0, 0));
    }

    void processAndPublishResults(const cv::Mat &left_img, const cv::Mat &right_img,
                                  const std_msgs::msg::Header &header)
    {
        if (matches_pub_->get_subscription_count() == 0)
            return;

        cv::Mat viz_img;
        cv::hconcat(left_img, right_img, viz_img);
        if (viz_img.channels() == 1)
            cv::cvtColor(viz_img, viz_img, cv::COLOR_GRAY2BGR);

        const float scale_x = static_cast<float>(left_img.cols) / input_width_;
        const float scale_y = static_cast<float>(left_img.rows) / input_height_;

        std::vector<cv::Point2f> kpts_left_raw, kpts_right_raw;
        parseKeypoints(kpts_left_raw, kpts_right_raw);

        std::vector<cv::DMatch> all_matches;
        parseMatches(all_matches, kpts_left_raw, kpts_right_raw);

        std::vector<bool> left_matched(kpts_left_raw.size(), false);
        std::vector<bool> right_matched(kpts_right_raw.size(), false);
        std::vector<float> match_confidences(kpts_left_raw.size(), 0.f);

        for (const auto &match : all_matches)
        {
            left_matched[match.queryIdx] = true;
            right_matched[match.trainIdx] = true;
            match_confidences[match.queryIdx] = 1.0f - match.distance;
        }

        for (const auto &match : all_matches)
        {
            cv::Point2f pt1_raw = kpts_left_raw[match.queryIdx];
            cv::Point2f pt2_raw = kpts_right_raw[match.trainIdx];
            cv::Point pt1(cvRound(pt1_raw.x * scale_x), cvRound(pt1_raw.y * scale_y));
            cv::Point pt2(cvRound(pt2_raw.x * scale_x) + left_img.cols, cvRound(pt2_raw.y * scale_y));
            cv::Scalar color = getMatchColor(1.0f - match.distance);
            cv::line(viz_img, pt1, pt2, color, 1, cv::LINE_AA);
        }

        for (size_t i = 0; i < kpts_left_raw.size(); ++i)
        {
            if (kpts_left_raw[i].x < 0)
                continue;
            cv::Point pt(cvRound(kpts_left_raw[i].x * scale_x), cvRound(kpts_left_raw[i].y * scale_y));
            cv::Scalar color = left_matched[i] ? getMatchColor(match_confidences[i]) : cv::Scalar(0, 0, 255); // Red if unmatched
            cv::circle(viz_img, pt, 2, color, -1, cv::LINE_AA);
        }

        for (size_t i = 0; i < kpts_right_raw.size(); ++i)
        {
            if (kpts_right_raw[i].x < 0)
                continue;
            cv::Point pt(cvRound(kpts_right_raw[i].x * scale_x) + left_img.cols, cvRound(kpts_right_raw[i].y * scale_y));
            // To color right points, we need to find the match it's part of
            cv::Scalar color = cv::Scalar(0, 0, 255); // Red if unmatched
            if (right_matched[i])
            {
                // This is slightly inefficient but fine for visualization
                for (const auto &m : all_matches)
                {
                    if (m.trainIdx == static_cast<int>(i))
                    {
                        color = getMatchColor(1.0f - m.distance);
                        break;
                    }
                }
            }
            cv::circle(viz_img, pt, 2, color, -1, cv::LINE_AA);
        }

        cv::putText(viz_img, "Matches: " + std::to_string(all_matches.size()), cv::Point(10, 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 20, 20), 1);

        auto viz_msg = cv_bridge::CvImage(header, "bgr8", viz_img).toImageMsg();
        matches_pub_->publish(*viz_msg);
    }

    void cleanup()
    {
        if (d_input_)
            cudaFree(d_input_);
        if (d_output_keypoints_)
            cudaFree(d_output_keypoints_);
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
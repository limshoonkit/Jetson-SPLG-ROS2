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
    void *d_input_;
    std::unordered_map<std::string, std::unique_ptr<DynamicOutputAllocator>> mAllocatorMap;

    // Host memory buffers for results
    std::vector<int64_t> h_keypoints_;
    std::vector<int64_t> h_matches_;
    std::vector<float> h_scores_;

    // GPU Preprocessing Mats
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
            auto dtype = engine_->getTensorDataType(tensorName);
            std::string dtype_str = "OTHER";
            if (dtype == DataType::kFLOAT)
                dtype_str = "FLOAT32";
            else if (dtype == DataType::kHALF)
                dtype_str = "HALF";
            else if (dtype == DataType::kINT64)
                dtype_str = "INT64";
            else if (dtype == DataType::kINT32)
                dtype_str = "INT32";
            else if (dtype == DataType::kINT8)
                dtype_str = "INT8";
            else if (dtype == DataType::kBOOL)
                dtype_str = "BOOL";

            if (engine_->getTensorIOMode(tensorName) == TensorIOMode::kINPUT)
            {
                input_tensor_name_ = tensorName;
                RCLCPP_INFO(this->get_logger(), "Found Input tensor '%s' with dtype: %s", tensorName, dtype_str.c_str());
            }
            else
            {
                std::string name(tensorName);
                if (name.find("keypoints") != std::string::npos)
                    output_keypoints_name_ = name;
                else if (name.find("matches") != std::string::npos)
                    output_matches_name_ = name;
                else if (name.find("scores") != std::string::npos)
                    output_scores_name_ = name;
                RCLCPP_INFO(this->get_logger(), "Found Output tensor '%s' with dtype: %s", tensorName, dtype_str.c_str());
            }
        }
        RCLCPP_INFO(this->get_logger(), "Using I/O tensors: Input='%s', Keypoints='%s', Matches='%s', Scores='%s'",
                    input_tensor_name_.c_str(), output_keypoints_name_.c_str(),
                    output_matches_name_.c_str(), output_scores_name_.c_str());
    }

    void allocateBuffers()
    {
        // Allocate buffer for input tensor. Output buffers are handled by DynamicOutputAllocator.
        size_t input_size = BATCH_SIZE * CHANNELS * input_height_ * input_width_ * sizeof(float); // NCHW (2, 1, H, W)

        if (use_unified_memory_)
        {
            CUDA_CHECK(cudaMallocManaged(&d_input_, input_size));
        }
        else
        {
            CUDA_CHECK(cudaMalloc(&d_input_, input_size));
        }

        // Host buffers for results.
        h_keypoints_.reserve(BATCH_SIZE * max_keypoints_ * 2);
        h_matches_.reserve(max_keypoints_ * 3);
        h_scores_.reserve(max_keypoints_);

        last_process_time_ = std::chrono::steady_clock::now();
        RCLCPP_INFO(this->get_logger(), "Frame skipping: %s (N=%d, rate=%.1fHz)",
                    frame_skip_mode_.c_str(), frame_skip_n_, max_process_rate_hz_);
    }

    void setupBindingsAndAllocators()
    {
        if (profile_inference_)
        {
            auto print_shape = [this](const char *name)
            {
                if (std::string(name).empty())
                    return;
                auto shape = engine_->getTensorShape(name);
                std::string shape_str;
                for (int i = 0; i < shape.nbDims; ++i)
                    shape_str += std::to_string(shape.d[i]) + ", ";
                RCLCPP_WARN(this->get_logger(), "Tensor '%s' shape: [ %s]", name, shape_str.c_str());
            };
            print_shape(input_tensor_name_.c_str());
            print_shape(output_keypoints_name_.c_str());
            print_shape(output_matches_name_.c_str());
            print_shape(output_scores_name_.c_str());
        }

        // Set fixed input tensor address and shape
        context_->setInputTensorAddress(input_tensor_name_.c_str(), d_input_);
        context_->setInputShape(input_tensor_name_.c_str(),
                                nvinfer1::Dims4{BATCH_SIZE, CHANNELS, input_height_, input_width_});

        // Use dynamic allocators for all outputs.
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

    void preprocessGPU(const cv::Mat &left_img, const cv::Mat &right_img, bool is_grayscale)
    {
        if (profile_inference_)
        {
            preprocess_start_ = std::chrono::high_resolution_clock::now();
        }

        CV_Assert(left_img.rows == input_height_ && left_img.cols == input_width_);
        CV_Assert(right_img.rows == input_height_ && right_img.cols == input_width_);
        CV_Assert(left_img.type() == CV_8UC1 && right_img.type() == CV_8UC1);

        cv::cuda::GpuMat d_left(left_img);
        cv::cuda::GpuMat d_right(right_img);
        launchToNCHW(d_left, d_right, d_input_, input_height_, input_width_);

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
        // TODO: Implement CPU preprocessing if needed
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
            cv_bridge::CvImagePtr left_cv = cv_bridge::toCvCopy(left_msg, "mono8");
            cv_bridge::CvImagePtr right_cv = cv_bridge::toCvCopy(right_msg, "mono8");

            if (left_cv->image.empty() || right_cv->image.empty())
            {
                RCLCPP_WARN(get_logger(), "Received empty image(s), skipping frame.");
                processing_.store(false);
                return;
            }

            // Preprocessing
            if (use_gpu_preprocessing_)
            {
                preprocessGPU(left_cv->image, right_cv->image, true);
            }
            else
            {
                std::vector<float> h_input(BATCH_SIZE * CHANNELS * input_height_ * input_width_);
                preprocessCPU(left_cv->image, right_cv->image, h_input.data(), true);
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

            // Keypoints
            auto &keypoints_allocator = mAllocatorMap.at(output_keypoints_name_);
            auto keypoints_shape = keypoints_allocator->getShape();
            actual_num_keypoints_ = (keypoints_shape.nbDims == 3) ? keypoints_shape.d[1] : 0; // Shape [2, N, 2]
            size_t keypoints_count = BATCH_SIZE * actual_num_keypoints_ * 2;
            h_keypoints_.resize(keypoints_count);
            CUDA_CHECK(cudaMemcpyAsync(h_keypoints_.data(), keypoints_allocator->getBuffer(), keypoints_count * sizeof(int64_t), cudaMemcpyDeviceToHost, stream_));

            // Matches
            auto &matches_allocator = mAllocatorMap.at(output_matches_name_);
            auto matches_shape = matches_allocator->getShape();
            actual_num_matches_ = (matches_shape.nbDims == 2) ? matches_shape.d[0] : 0; // Shape [M, 3]
            size_t matches_count = actual_num_matches_ * 3;
            h_matches_.resize(matches_count);
            CUDA_CHECK(cudaMemcpyAsync(h_matches_.data(), matches_allocator->getBuffer(), matches_count * sizeof(int64_t), cudaMemcpyDeviceToHost, stream_));

            // Scores
            auto &scores_allocator = mAllocatorMap.at(output_scores_name_);
            auto scores_shape = scores_allocator->getShape();
            size_t scores_count = (scores_shape.nbDims == 1) ? scores_shape.d[0] : 0; // Shape [M]
            h_scores_.resize(scores_count);
            CUDA_CHECK(cudaMemcpyAsync(h_scores_.data(), scores_allocator->getBuffer(), scores_count * sizeof(float), cudaMemcpyDeviceToHost, stream_));

            // Synchronize after all async copies are queued to ensure data is on host
            CUDA_CHECK(cudaStreamSynchronize(stream_));

            if (profile_inference_)
            {
                auto inference_end = std::chrono::high_resolution_clock::now();
                auto inference_time = std::chrono::duration_cast<std::chrono::microseconds>(inference_end - inference_start_).count() / 1000.0;
                RCLCPP_INFO(this->get_logger(), "Inference: %.2f ms", inference_time);

                auto pipeline_end = std::chrono::high_resolution_clock::now();
                auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(pipeline_end - pipeline_start).count() / 1000.0;
                RCLCPP_INFO(this->get_logger(), "Total pipeline: %.2f ms, Keypoints/img: %d, Matches: %d",
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
        left_kpts.resize(actual_num_keypoints_);
        right_kpts.resize(actual_num_keypoints_);
        const int kpts_per_image_flat_size = actual_num_keypoints_ * 2;

        for (int i = 0; i < actual_num_keypoints_; ++i)
        {
            // Left image (image 0) keypoints are at the start of the buffer
            left_kpts[i] = cv::Point2f(
                static_cast<float>(h_keypoints_[i * 2]),
                static_cast<float>(h_keypoints_[i * 2 + 1]));

            // Right image (image 1) keypoints are offset by the size of the first image's data
            right_kpts[i] = cv::Point2f(
                static_cast<float>(h_keypoints_[kpts_per_image_flat_size + i * 2]),
                static_cast<float>(h_keypoints_[kpts_per_image_flat_size + i * 2 + 1]));
        }
    }

    void parseMatches(std::vector<cv::DMatch> &final_matches)
    {
        final_matches.clear();
        final_matches.reserve(actual_num_matches_);

        for (int i = 0; i < actual_num_matches_; ++i)
        {
            // Match layout: [batch_idx, query_idx, train_idx]
            int64_t query_idx = h_matches_[i * 3 + 1];
            int64_t train_idx = h_matches_[i * 3 + 2];
            float score = h_scores_[i];

            // Ensure indices are within bounds of the keypoint vectors
            if (query_idx >= 0 && query_idx < actual_num_keypoints_ &&
                train_idx >= 0 && train_idx < actual_num_keypoints_)
            {
                // cv::DMatch(queryIdx, trainIdx, distance/score)
                final_matches.emplace_back(
                    static_cast<int>(query_idx),
                    static_cast<int>(train_idx),
                    score);
            }
        }
    }

    void processAndPublishResults(const cv::Mat &left_img, const cv::Mat &right_img,
                                  const std_msgs::msg::Header &header)
    {
        if (matches_pub_->get_subscription_count() == 0)
            return;

        std::vector<cv::Point2f> kpts_left_raw, kpts_right_raw;
        parseKeypoints(kpts_left_raw, kpts_right_raw);

        std::vector<cv::DMatch> all_matches;
        parseMatches(all_matches);

        cv::Mat viz_img;
        cv::hconcat(left_img, right_img, viz_img);
        if (viz_img.channels() == 1)
            cv::cvtColor(viz_img, viz_img, cv::COLOR_GRAY2BGR);

        const cv::Scalar kMatchColor(0, 255, 0);

        for (const auto &match : all_matches)
        {
            cv::Point2f pt_left = kpts_left_raw[match.queryIdx];
            cv::Point2f pt_right = kpts_right_raw[match.trainIdx];
            cv::Point2f pt_right_offset(pt_right.x + left_img.cols, pt_right.y);

            // Use the constant color for everything
            cv::line(viz_img, pt_left, pt_right_offset, kMatchColor, 1);
            cv::circle(viz_img, pt_left, 3, kMatchColor, cv::FILLED);
            cv::circle(viz_img, pt_right_offset, 3, kMatchColor, cv::FILLED);
        }

        std::string text = "Matches: " + std::to_string(all_matches.size());
        cv::putText(viz_img, text, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

        auto viz_msg = cv_bridge::CvImage(header, "bgr8", viz_img).toImageMsg();
        matches_pub_->publish(*viz_msg);
    }

    void cleanup()
    {
        if (d_input_)
            cudaFree(d_input_);
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
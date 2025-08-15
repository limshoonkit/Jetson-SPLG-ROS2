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

inline void checkCuda(const char *where)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error after " << where << ": " << cudaGetErrorString(err) << std::endl;
    }
}

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
    void *d_temp_input_fp32_;
    void *d_output_keypoints_fp16_;
    void *d_output_keypoints_fp32_;

    std::unordered_map<std::string, std::unique_ptr<DynamicOutputAllocator>> mAllocatorMap;

    // Host memory buffers
    std::vector<float> h_keypoints_;
    std::vector<int> h_matches_;
    std::vector<float> h_scores_;

    // GPU Preprocessing Mats
    cv::cuda::GpuMat gpu_left_resized_, gpu_right_resized_;
    cv::cuda::GpuMat gpu_left_gray_, gpu_right_gray_;
    cv::cuda::GpuMat gpu_left_norm_, gpu_right_norm_;

    std::string engine_path_;
    int input_height_, input_width_;
    int max_keypoints_;
    bool profile_inference_;
    bool use_gpu_preprocessing_;
    int actual_num_matches_;

    std::string frame_skip_mode_;
    int frame_skip_n_;
    double max_process_rate_hz_;

    std::atomic<bool> processing_{false};
    std::chrono::steady_clock::time_point last_process_time_;
    int frame_counter_{0};

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

    // Helper for CPU-based float to FP16 conversion
    static uint16_t float_to_half_bits(float f)
    {
        union
        {
            float f_val;
            uint32_t u_val;
        } converter;
        converter.f_val = f;
        uint32_t u = converter.u_val;

        uint32_t sign = (u >> 31) & 0x0001;
        uint32_t exp = (u >> 23) & 0x00ff;
        uint32_t mant = u & 0x007fffff;

        uint16_t h_sign = sign << 15;
        uint16_t h_exp, h_mant;

        if (exp == 0)
        {
            h_exp = 0;
            h_mant = 0;
        }
        else if (exp == 255)
        {
            h_exp = 0x1f;
            h_mant = (mant == 0) ? 0 : 0x0200;
        }
        else
        {
            int16_t new_exp = exp - 127;
            if (new_exp < -14)
            {
                h_exp = 0;
                h_mant = 0;
            }
            else if (new_exp > 15)
            {
                h_exp = 0x1f;
                h_mant = 0;
            }
            else
            {
                h_exp = new_exp + 15;
                h_mant = mant >> 13;
            }
        }
        return h_sign | (h_exp << 10) | h_mant;
    }

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

        cudaStreamCreate(&stream_);
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
        // Allocate final input buffer as FP16
        size_t input_size_fp16 = BATCH_SIZE * CHANNELS * input_height_ * input_width_ * sizeof(uint16_t);
        cudaMalloc(&d_input_, input_size_fp16);

        // Allocate a temporary buffer for preprocessing with FP32 for the GPU path
        if (use_gpu_preprocessing_)
        {
            size_t input_size_fp32 = BATCH_SIZE * CHANNELS * input_height_ * input_width_ * sizeof(float);
            cudaMalloc(&d_temp_input_fp32_, input_size_fp32);
        }

        size_t keypoints_size_fp16 = BATCH_SIZE * max_keypoints_ * 2 * sizeof(uint16_t);
        cudaMalloc(&d_output_keypoints_fp16_, keypoints_size_fp16);

        size_t keypoints_size_fp32 = BATCH_SIZE * max_keypoints_ * 2 * sizeof(float);
        cudaMalloc(&d_output_keypoints_fp32_, keypoints_size_fp32);

        h_keypoints_.resize(BATCH_SIZE * max_keypoints_ * 2);
        h_matches_.resize(max_keypoints_ * 3);
        h_scores_.resize(max_keypoints_);

        last_process_time_ = std::chrono::steady_clock::now();
        RCLCPP_INFO(this->get_logger(), "Frame skipping: %s (N=%d, rate=%.1fHz)",
                    frame_skip_mode_.c_str(), frame_skip_n_, max_process_rate_hz_);
    }

    void setupBindingsAndAllocators()
    {
        context_->setTensorAddress(input_tensor_name_.c_str(), d_input_);
        context_->setTensorAddress(output_keypoints_name_.c_str(), d_output_keypoints_fp16_);

        for (const auto &name : {output_matches_name_, output_scores_name_})
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
        cv_stream_ = cv::cuda::StreamAccessor::wrapStream(stream_);
    }

    void preprocessGPU(const cv::Mat &left_img, const cv::Mat &right_img, bool is_grayscale)
    {
        cv::cuda::GpuMat gpu_imgs[2];
        cv::cuda::GpuMat gpu_resized_imgs[2];
        cv::cuda::GpuMat gpu_gray_imgs[2];
        cv::cuda::GpuMat gpu_norm_imgs[2];

        // Upload to GPU
        gpu_imgs[0].upload(left_img, cv_stream_);
        gpu_imgs[1].upload(right_img, cv_stream_);
        cv_stream_.waitForCompletion();

        for (int i = 0; i < 2; ++i)
        {
            if (gpu_imgs[i].empty())
            {
                std::cerr << "[WARN] GPU image " << i << " is EMPTY after upload!" << std::endl;
            }
            else
            {
                std::cerr << "[INFO] GPU image " << i << ": "
                          << gpu_imgs[i].cols << "x" << gpu_imgs[i].rows
                          << " ch=" << gpu_imgs[i].channels()
                          << " step=" << gpu_imgs[i].step << std::endl;
            }
        }

        // Resize
        for (int i = 0; i < 2; ++i)
        {
            cv::cuda::resize(gpu_imgs[i], gpu_resized_imgs[i],
                             cv::Size(input_width_, input_height_),
                             0, 0, cv::INTER_LINEAR, cv_stream_);
        }
        cv_stream_.waitForCompletion();

        // Convert to grayscale if needed
        if (is_grayscale)
        {
            for (int i = 0; i < 2; ++i)
            {
                if (gpu_resized_imgs[i].channels() == 1)
                {
                    gpu_resized_imgs[i].copyTo(gpu_gray_imgs[i], cv_stream_); // deep copy
                }
                else
                {
                    cv::cuda::cvtColor(gpu_resized_imgs[i], gpu_gray_imgs[i],
                                       cv::COLOR_BGR2GRAY, 1, cv_stream_);
                }
            }
        }
        else
        {
            for (int i = 0; i < 2; ++i)
            {
                gpu_resized_imgs[i].copyTo(gpu_gray_imgs[i], cv_stream_);
            }
        }
        cv_stream_.waitForCompletion();

        // Normalize to [0,1] float
        for (int i = 0; i < 2; ++i)
        {
            gpu_gray_imgs[i].convertTo(gpu_norm_imgs[i], CV_32F, 1.0 / 255.0, 0.0, cv_stream_);
        }
        cv_stream_.waitForCompletion();

        // Debug — dump normalized images
        for (int i = 0; i < 2; ++i)
        {
            cv::Mat dbg;
            gpu_norm_imgs[i].download(dbg, cv_stream_);
            cv_stream_.waitForCompletion();
            cv::imwrite("/home/nvidia/ros2_ws/tmp/debug_norm_" + std::to_string(i) + ".png", dbg * 255);
            std::cerr << "[DEBUG] Saved /home/nvidia/ros2_ws/tmp/debug_norm_" << i << ".png" << std::endl;
        }
    }

    void preprocessCPU(const cv::Mat &left_img, const cv::Mat &right_img, uint16_t *output, bool is_grayscale)
    {
        cv::Mat imgs[2] = {left_img, right_img};
        const int image_area = input_height_ * input_width_;

        for (int i = 0; i < 2; ++i)
        {
            cv::Mat resized_img;
            if (imgs[i].rows != input_height_ || imgs[i].cols != input_width_)
            {
                cv::resize(imgs[i], resized_img, cv::Size(input_width_, input_height_));
            }
            else
            {
                resized_img = imgs[i];
            }

            cv::Mat gray_img;
            if (!is_grayscale)
            {
                cv::cvtColor(resized_img, gray_img, cv::COLOR_BGR2GRAY);
            }
            else
            {
                gray_img = resized_img;
            }

            cv::Mat float_img;
            gray_img.convertTo(float_img, CV_32FC1, 1.0 / 255.0);

            if (!float_img.isContinuous())
            {
                float_img = float_img.clone();
            }

            float *float_ptr = reinterpret_cast<float *>(float_img.data);
            uint16_t *buffer_offset = output + i * image_area;
            for (int p = 0; p < image_area; ++p)
            {
                buffer_offset[p] = float_to_half_bits(float_ptr[p]);
            }
        }
    }

    void debugInputTensorFP32(const float *d_fp32, int sample_size, cudaStream_t stream)
    {
        std::vector<float> host_fp32(sample_size);
        cudaMemcpyAsync(host_fp32.data(), d_fp32,
                        sample_size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        RCLCPP_INFO(this->get_logger(), "[DEBUG FP32] First 10 values:");
        for (int i = 0; i < 10; ++i)
            RCLCPP_INFO(this->get_logger(), "  Pixel[%d]: %.6f", i, host_fp32[i]);

        auto [min_it, max_it] = std::minmax_element(host_fp32.begin(), host_fp32.end());
        RCLCPP_INFO(this->get_logger(), "[DEBUG FP32] Range: [%.6f, %.6f]", *min_it, *max_it);
    }

    void debugInputTensorFP16(const __half *d_fp16, int sample_size, cudaStream_t stream)
    {
        std::vector<float> host_fp32(sample_size);

        // Temporary GPU buffer for FP32
        float *d_temp_fp32 = nullptr;
        cudaMalloc(&d_temp_fp32, sample_size * sizeof(float));

        // Convert FP16 → FP32 for inspection
        launchConvertFP16ToFP32(d_fp16, d_temp_fp32, sample_size, stream);

        cudaMemcpyAsync(host_fp32.data(), d_temp_fp32,
                        sample_size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        cudaFree(d_temp_fp32);

        RCLCPP_INFO(this->get_logger(), "[DEBUG FP16] First 10 values:");
        for (int i = 0; i < 10; ++i)
            RCLCPP_INFO(this->get_logger(), "  Pixel[%d]: %.6f", i, host_fp32[i]);

        auto [min_it, max_it] = std::minmax_element(host_fp32.begin(), host_fp32.end());
        RCLCPP_INFO(this->get_logger(), "[DEBUG FP16] Range: [%.6f, %.6f]", *min_it, *max_it);
    }

    void debugRawInput(const cv::Mat &left_img, const cv::Mat &right_img)
    {
        RCLCPP_INFO(this->get_logger(), "Raw image info:");
        RCLCPP_INFO(this->get_logger(), " Left: %dx%d, channels=%d, type=%d",
                    left_img.cols, left_img.rows, left_img.channels(), left_img.type());
        RCLCPP_INFO(this->get_logger(), " Right: %dx%d, channels=%d, type=%d",
                    right_img.cols, right_img.rows, right_img.channels(), right_img.type());
        // Check if images have data

        if (left_img.empty() || right_img.empty())
        {
            RCLCPP_ERROR(this->get_logger(), "Input images are empty!");
            return;
        }

        // Sample pixel values from raw images
        RCLCPP_INFO(this->get_logger(), "Raw pixel samples:");
        for (int i = 0; i < 5; ++i)
        {
            if (left_img.channels() == 3)
            {
                cv::Vec3b pixel = left_img.at<cv::Vec3b>(100, 100 + i);
                RCLCPP_INFO(this->get_logger(), " Left[%d]: (%d,%d,%d)", i, pixel[0], pixel[1], pixel[2]);
            }
            else
            {
                uint8_t pixel = left_img.at<uint8_t>(100, 100 + i);
                RCLCPP_INFO(this->get_logger(), " Left[%d]: %d", i, pixel);
            }
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
            bool is_grayscale = (left_msg->encoding == "mono8");
            const char *desired_encoding = is_grayscale ? "mono8" : "bgr8";

            cv_bridge::CvImagePtr left_cv = cv_bridge::toCvCopy(left_msg, desired_encoding);
            cv_bridge::CvImagePtr right_cv = cv_bridge::toCvCopy(right_msg, desired_encoding);
            debugRawInput(left_cv->image, right_cv->image);

            if (left_cv->image.empty() || right_cv->image.empty())
            {
                RCLCPP_WARN(get_logger(), "Received empty image(s), skipping frame.");
                processing_.store(false);
                return;
            }

            if (use_gpu_preprocessing_)
            {
                preprocessGPU(left_cv->image, right_cv->image, is_grayscale);
                // const int elements = BATCH_SIZE * CHANNELS * input_height_ * input_width_;
                // debugInputTensorFP16(
                //     reinterpret_cast<const __half *>(d_input_),
                //     elements,
                //     stream_);
                // debugInputTensorFP32(
                //     reinterpret_cast<const float *>(d_temp_input_fp32_),
                //     elements,
                //     stream_);
            }
            else
            {
                std::vector<uint16_t> h_input(BATCH_SIZE * CHANNELS * input_height_ * input_width_);
                preprocessCPU(left_cv->image, right_cv->image, h_input.data(), is_grayscale);
                cudaMemcpyAsync(d_input_, h_input.data(), h_input.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, stream_);
            }

            nvinfer1::Dims input_dims;
            input_dims.nbDims = 4;
            input_dims.d[0] = BATCH_SIZE;
            input_dims.d[1] = CHANNELS;
            input_dims.d[2] = input_height_;
            input_dims.d[3] = input_width_;

            if (!context_->setInputShape(input_tensor_name_.c_str(), input_dims))
            {
                RCLCPP_ERROR(this->get_logger(), "Failed to set input shape");
                processing_.store(false);
                return;
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
                total_keypoint_elements,
                stream_);

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
                size_t max_host_matches = h_matches_.size() / 3;
                if (actual_num_matches_ > static_cast<int>(max_host_matches))
                {
                    RCLCPP_WARN(this->get_logger(), "Model produced more matches (%d) than host buffer capacity (%zu). Truncating.", actual_num_matches_, max_host_matches);
                    actual_num_matches_ = max_host_matches;
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

        int valid_left_kpts = std::count_if(kpts_left_raw.begin(), kpts_left_raw.end(), [](const cv::Point2f &pt)
                                            { return pt.x >= 0; });
        int valid_right_kpts = std::count_if(kpts_right_raw.begin(), kpts_right_raw.end(), [](const cv::Point2f &pt)
                                             { return pt.x >= 0; });
        drawInfoOverlay(viz_img, valid_left_kpts, valid_right_kpts, all_matches.size());

        auto viz_msg = cv_bridge::CvImage(header, "bgr8", viz_img).toImageMsg();
        matches_pub_->publish(*viz_msg);
    }

    void drawInfoOverlay(cv::Mat &viz_img, int left_count, int right_count, int match_count)
    {
        cv::rectangle(viz_img, cv::Point(5, 5), cv::Point(200, 75), cv::Scalar(0, 0, 0), -1);
        cv::addWeighted(viz_img(cv::Rect(5, 5, 195, 70)), 0.5, viz_img(cv::Rect(5, 5, 195, 70)), 0.5, 0, viz_img(cv::Rect(5, 5, 195, 70)));
        cv::putText(viz_img, "Left kpts: " + std::to_string(left_count), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        cv::putText(viz_img, "Right kpts: " + std::to_string(right_count), cv::Point(10, 45), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
        cv::putText(viz_img, "Matches: " + std::to_string(match_count), cv::Point(10, 65), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
    }

    void cleanup()
    {
        if (d_input_)
            cudaFree(d_input_);
        if (d_temp_input_fp32_)
            cudaFree(d_temp_input_fp32_);
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
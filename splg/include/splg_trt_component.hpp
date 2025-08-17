#ifndef SPLG_TRT_COMPONENT_HPP_
#define SPLG_TRT_COMPONENT_HPP_

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
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

#include <nvtx3/nvToolsExt.h> // For profiling with NVTX
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include "type_conv_helper.cuh"

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
namespace uosm
{
    class SuperPointLightGlueTrt : public rclcpp::Node
    {
    public:
        explicit SuperPointLightGlueTrt(const rclcpp::NodeOptions &options);
        virtual ~SuperPointLightGlueTrt();

    private:
        class Logger : public nvinfer1::ILogger
        {
            void log(nvinfer1::ILogger::Severity severity, const char *msg) noexcept override
            {
                if (severity <= nvinfer1::ILogger::Severity::kWARNING)
                {
                    std::cout << msg << std::endl;
                }
            }
        } gLogger;

        // Constants
        const std::string IMAGE0 = "image0";
        const std::string IMAGE1 = "image1";
        static constexpr int BATCH_SIZE = 2;
        static constexpr int CHANNELS = 1;

        // Parameters
        std::string engine_path_;
        int input_width_, input_height_;
        int max_keypoints_;
        bool use_gpu_preprocessing_;
        bool profile_inference_;
        bool use_unified_memory_;

        std::string frame_skip_mode_;
        int frame_skip_n_;
        double max_process_rate_hz_;

        std::string input_tensor_name_;     // FLOAT32
        std::string output_keypoints_name_; // INT64
        std::string output_matches_name_;   // INT64
        std::string output_scores_name_;    // FLOAT32

        // Flags
        int actual_num_matches_;
        int actual_num_keypoints_;
        std::atomic<bool> processing_{false};
        int frame_counter_{0};
        std::chrono::steady_clock::time_point last_process_time_;
        std::chrono::high_resolution_clock::time_point preprocess_start_, inference_start_;

        // TensorRT
        std::unique_ptr<nvinfer1::IRuntime> runtime_;
        std::unique_ptr<nvinfer1::ICudaEngine> engine_;
        std::unique_ptr<nvinfer1::IExecutionContext> context_;
        cudaStream_t stream_;

        // GPU memory
        void *d_input_;
        std::unordered_map<std::string, std::unique_ptr<DynamicOutputAllocator>> mAllocatorMap;

        cv::cuda::GpuMat gpu_image0_uploaded_, gpu_image1_uploaded_;
        cv::cuda::GpuMat gpu_image0_resized_, gpu_image1_resized_;
        cv::cuda::GpuMat gpu_image0_gray_, gpu_image1_gray_;
        cv::cuda::GpuMat gpu_image0_norm_fp32_, gpu_image1_norm_fp32_;

        // Host buffer
        std::vector<int64_t> h_keypoints_;
        std::vector<int64_t> h_matches_;
        std::vector<float> h_scores_;

        // ROS2
        using SyncPolicy = message_filters::sync_policies::ApproximateTime<
            sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
        message_filters::Subscriber<sensor_msgs::msg::Image> image0_sub_;
        message_filters::Subscriber<sensor_msgs::msg::Image> image1_sub_;
        std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr matches_pub_;
        rclcpp::TimerBase::SharedPtr mInitTimer;

        // Callbacks
        void stereoCallback(const sensor_msgs::msg::Image::ConstSharedPtr &image0_msg, const sensor_msgs::msg::Image::ConstSharedPtr &image1_msg);

        // Functions
        bool initTensorRT();
        bool shouldProcessFrame();
        void init();
        void discoverTensorNames();
        void allocateBuffers();
        void setupBindingsAndAllocators();
        void preprocessImagesGPU(const cv::Mat &image0, const cv::Mat &image1);
        void preprocessImagesCPU(const cv::Mat &image0, const cv::Mat &image1);
        void parseKeypoints(std::vector<cv::Point2f> &image0_kpts, std::vector<cv::Point2f> &image1_kpts);
        void parseMatches(std::vector<cv::DMatch> &final_matches);
        void publishViz(const cv::Mat &image0, const cv::Mat &image1,
                        const std_msgs::msg::Header &header);
    };
} // namespace uosm
#endif // SPLG_TRT_COMPONENT_HPP_
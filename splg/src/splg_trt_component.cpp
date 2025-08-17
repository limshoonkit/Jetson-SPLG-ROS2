#include "../include/splg_trt_component.hpp"

namespace uosm
{
    SuperPointLightGlueTrt::SuperPointLightGlueTrt(const rclcpp::NodeOptions &options) : Node("splg_trt", options)
    {
        RCLCPP_INFO(get_logger(), "**********************************");
        RCLCPP_INFO(get_logger(), " SuperPointLightGlueTrt Component ");
        RCLCPP_INFO(get_logger(), "**********************************");
        RCLCPP_INFO(get_logger(), " * namespace: %s", get_namespace());
        RCLCPP_INFO(get_logger(), " * node name: %s", get_name());
        RCLCPP_INFO(get_logger(), "**********************************");

        std::chrono::milliseconds init_msec(static_cast<int>(10.0));
        mInitTimer = create_wall_timer(std::chrono::duration_cast<std::chrono::milliseconds>(init_msec), std::bind(&SuperPointLightGlueTrt::init, this));
    }

    SuperPointLightGlueTrt::~SuperPointLightGlueTrt()
    {
        // Clean up
        RCLCPP_INFO(get_logger(), "Freeing CUDA memory");
        if (d_input_)
            cudaFree(d_input_);
        if (stream_)
            cudaStreamDestroy(stream_);
    }

    void SuperPointLightGlueTrt::init()
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
        declare_parameter("use_unified_memory", false);

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
        image0_sub_.subscribe(this, IMAGE0);
        image1_sub_.subscribe(this, IMAGE1);

        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(10), image0_sub_, image1_sub_);
        sync_->registerCallback(&SuperPointLightGlueTrt::stereoCallback, this);

        matches_pub_ = this->create_publisher<sensor_msgs::msg::Image>("debug_splg_kpts_matches", 1);
        RCLCPP_INFO(this->get_logger(), "SuperPoint-LightGlue Trt initialized [%dx%d]",
                    input_width_, input_height_);
    }

    bool SuperPointLightGlueTrt::initTensorRT()
    {
        runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
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

        engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(engine_data.data(), size));
        if (!engine_)
            return false;

        context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
        if (!context_)
            return false;

        CUDA_CHECK(cudaStreamCreate(&stream_));
        return true;
    }

    void SuperPointLightGlueTrt::discoverTensorNames()
    {
        int32_t nbIOTensors = engine_->getNbIOTensors();
        for (int32_t i = 0; i < nbIOTensors; ++i)
        {
            const char *tensorName = engine_->getIOTensorName(i);
            auto dtype = engine_->getTensorDataType(tensorName);
            std::string dtype_str = "OTHER";
            if (dtype == nvinfer1::DataType::kFLOAT)
                dtype_str = "FLOAT32";
            else if (dtype == nvinfer1::DataType::kHALF)
                dtype_str = "HALF";
            else if (dtype == nvinfer1::DataType::kINT64)
                dtype_str = "INT64";
            else if (dtype == nvinfer1::DataType::kINT32)
                dtype_str = "INT32";
            else if (dtype == nvinfer1::DataType::kINT8)
                dtype_str = "INT8";
            else if (dtype == nvinfer1::DataType::kBOOL)
                dtype_str = "BOOL";

            if (engine_->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT)
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

    void SuperPointLightGlueTrt::allocateBuffers()
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

    void SuperPointLightGlueTrt::setupBindingsAndAllocators()
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

    void SuperPointLightGlueTrt::preprocessImagesGPU(const cv::Mat &image0, const cv::Mat &image1)
    {
        if (profile_inference_)
        {
            preprocess_start_ = std::chrono::high_resolution_clock::now();
        }

        CV_Assert(image0.rows == input_height_ && image0.cols == input_width_);
        CV_Assert(image1.rows == input_height_ && image1.cols == input_width_);
        CV_Assert(image0.type() == CV_8UC1 && image1.type() == CV_8UC1);

        cv::cuda::GpuMat d_image0(image0);
        cv::cuda::GpuMat d_image1(image1);
        launchToNCHW(d_image0, d_image1, d_input_, input_height_, input_width_);

        if (profile_inference_)
        {
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            auto preprocess_end = std::chrono::high_resolution_clock::now();
            auto preprocess_time = std::chrono::duration_cast<std::chrono::microseconds>(preprocess_end - preprocess_start_).count() / 1000.0;
            RCLCPP_INFO(this->get_logger(), "GPU Preprocessing: %.2f ms", preprocess_time);
        }
    }

    void SuperPointLightGlueTrt::preprocessImagesCPU(const cv::Mat &image0, const cv::Mat &image1)
    {
        if (profile_inference_)
        {
            preprocess_start_ = std::chrono::high_resolution_clock::now();
        }

        CV_Assert(image0.rows == input_height_ && image0.cols == input_width_);
        CV_Assert(image1.rows == input_height_ && image1.cols == input_width_);
        CV_Assert(image0.type() == CV_8UC1 && image1.type() == CV_8UC1);
        CV_Assert(image0.isContinuous() && image1.isContinuous());

        const int channel_size = input_height_ * input_width_;

        const uint8_t *p_image0 = image0.ptr<uint8_t>(0);
        const uint8_t *p_image1 = image1.ptr<uint8_t>(0);

        std::vector<float> h_input_(BATCH_SIZE * CHANNELS * input_height_ * input_width_);
        float *p_out0 = h_input_.data();
        float *p_out1 = h_input_.data() + channel_size;

        // Normalization and Stacking
        for (int i = 0; i < channel_size; ++i)
        {
            p_out0[i] = static_cast<float>(p_image0[i]) / 255.0f;
            p_out1[i] = static_cast<float>(p_image1[i]) / 255.0f;
        }

        // Copy the processed data from host to device
        CUDA_CHECK(cudaMemcpyAsync(d_input_, h_input_.data(), h_input_.size() * sizeof(float), cudaMemcpyHostToDevice, stream_));

        if (profile_inference_)
        {
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            auto preprocess_end = std::chrono::high_resolution_clock::now();
            auto preprocess_time = std::chrono::duration_cast<std::chrono::microseconds>(preprocess_end - preprocess_start_).count() / 1000.0;
            RCLCPP_INFO(this->get_logger(), "CPU Preprocessing + HtoD Transfer: %.2f ms", preprocess_time);
        }
    }

    bool SuperPointLightGlueTrt::shouldProcessFrame()
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

    void SuperPointLightGlueTrt::parseKeypoints(std::vector<cv::Point2f> &image0_kpts, std::vector<cv::Point2f> &image1_kpts)
    {
        image0_kpts.resize(actual_num_keypoints_);
        image1_kpts.resize(actual_num_keypoints_);
        const int kpts_per_image_flat_size = actual_num_keypoints_ * 2;

        for (int i = 0; i < actual_num_keypoints_; ++i)
        {
            // Image 0 keypoints are at the start of the buffer
            image0_kpts[i] = cv::Point2f(
                static_cast<float>(h_keypoints_[i * 2]),
                static_cast<float>(h_keypoints_[i * 2 + 1]));

            // Image 1 keypoints are offset by the size of the first image's data
            image1_kpts[i] = cv::Point2f(
                static_cast<float>(h_keypoints_[kpts_per_image_flat_size + i * 2]),
                static_cast<float>(h_keypoints_[kpts_per_image_flat_size + i * 2 + 1]));
        }
    }

    void SuperPointLightGlueTrt::parseMatches(std::vector<cv::DMatch> &final_matches)
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

    void SuperPointLightGlueTrt::stereoCallback(
        const sensor_msgs::msg::Image::ConstSharedPtr &image0_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr &image1_msg)
    {
        if (!shouldProcessFrame())
            return;
        processing_.store(true);
        auto pipeline_start = std::chrono::high_resolution_clock::now();
        nvtxRangePushA("StereoCallback");

        try
        {
            nvtxRangePushA("Preprocessing");
            cv_bridge::CvImagePtr image0_cv = cv_bridge::toCvCopy(image0_msg);
            cv_bridge::CvImagePtr image1_cv = cv_bridge::toCvCopy(image1_msg);

            if (image0_cv->image.empty() || image1_cv->image.empty())
            {
                RCLCPP_WARN(get_logger(), "Received empty image(s), skipping frame.");
                processing_.store(false);
                return;
            }

            // Preprocessing
            if (use_gpu_preprocessing_)
            {
                preprocessImagesGPU(image0_cv->image, image1_cv->image);
            }
            else
            {
                preprocessImagesCPU(image0_cv->image, image1_cv->image);
            }
            nvtxRangePop();
            // Inference
            if (profile_inference_)
            {
                inference_start_ = std::chrono::high_resolution_clock::now();
            }
            nvtxRangePushA("Inference");
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
            nvtxRangePop();
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
            nvtxRangePushA("debug viz");
            publishViz(image0_cv->image, image1_cv->image, image0_cv->header);
            nvtxRangePop();
        }
        catch (const cv_bridge::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "Exception in stereoCallback: %s", e.what());
        }
        nvtxRangePop();
        processing_.store(false);
    }

    void SuperPointLightGlueTrt::publishViz(const cv::Mat &image0, const cv::Mat &image1,
                                            const std_msgs::msg::Header &header)
    {
        if (matches_pub_->get_subscription_count() == 0)
            return;

        std::vector<cv::Point2f> kpts_image0_raw, kpts_image1_raw;
        parseKeypoints(kpts_image0_raw, kpts_image1_raw);

        std::vector<cv::DMatch> all_matches;
        parseMatches(all_matches);

        cv::Mat viz_img;
        cv::hconcat(image0, image1, viz_img);
        if (viz_img.channels() == 1)
            cv::cvtColor(viz_img, viz_img, cv::COLOR_GRAY2BGR);

        const cv::Scalar kMatchColor(0, 255, 0);

        for (const auto &match : all_matches)
        {
            cv::Point2f pt_image0 = kpts_image0_raw[match.queryIdx];
            cv::Point2f pt_image1 = kpts_image1_raw[match.trainIdx];
            cv::Point2f pt_image1_offset(pt_image1.x + image0.cols, pt_image1.y);

            cv::line(viz_img, pt_image0, pt_image1_offset, kMatchColor, 1);
            cv::circle(viz_img, pt_image0, 3, kMatchColor, cv::FILLED);
            cv::circle(viz_img, pt_image1_offset, 3, kMatchColor, cv::FILLED);
        }

        std::string text = "Matches: " + std::to_string(all_matches.size());
        cv::putText(viz_img, text, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

        auto viz_msg = cv_bridge::CvImage(header, "bgr8", viz_img).toImageMsg();
        matches_pub_->publish(*viz_msg);
    }
} // namespace uosm

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(uosm::SuperPointLightGlueTrt)
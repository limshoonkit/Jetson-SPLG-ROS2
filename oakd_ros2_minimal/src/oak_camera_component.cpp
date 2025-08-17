#include "../include/oak_camera_component.hpp"

#include <rclcpp/time.hpp>
#include <rclcpp/utilities.hpp>

using namespace std::chrono_literals;
using namespace std::placeholders;

namespace uosm
{
    namespace depthai
    {
        OakCamera::OakCamera(const rclcpp::NodeOptions &options) : Node("oak_node", options)
        {
            RCLCPP_INFO(get_logger(), "********************************");
            RCLCPP_INFO(get_logger(), "      OAK Camera Component ");
            RCLCPP_INFO(get_logger(), "********************************");
            RCLCPP_INFO(get_logger(), " * namespace: %s", get_namespace());
            RCLCPP_INFO(get_logger(), " * node name: %s", get_name());
            RCLCPP_INFO(get_logger(), "********************************");

            std::chrono::milliseconds init_msec(static_cast<int>(10.0));
            mInitTimer = create_wall_timer(std::chrono::duration_cast<std::chrono::milliseconds>(init_msec), std::bind(&OakCamera::init, this));
        }

        void OakCamera::init()
        {
            // RCLCPP_INFO(get_logger(), "Init params");

            mInitTimer->cancel();
            declare_parameter("monoResolution", "480p");
            declare_parameter("lrFPS", 30);
            declare_parameter("syncThreshold", 10);

            monoResolution = get_parameter("monoResolution").as_string();
            lrFPS = get_parameter("lrFPS").as_int();
            syncThreshold = get_parameter("syncThreshold").as_int();

            declare_parameter("convInterleaved", false);
            declare_parameter("convGetBaseDeviceTimestamp", false);
            declare_parameter("convUpdateROSBaseTimeOnToRosMsg", true);
            declare_parameter("convReverseSocketOrder", true);
            convInterleaved = get_parameter("convInterleaved").as_bool();
            convGetBaseDeviceTimestamp = get_parameter("convGetBaseDeviceTimestamp").as_bool();
            convUpdateROSBaseTimeOnToRosMsg = get_parameter("convUpdateROSBaseTimeOnToRosMsg").as_bool();
            convReverseSocketOrder = get_parameter("convReverseSocketOrder").as_bool();

            initPipeline();

            initPubSub();

            // start thread
            mPipelineThread = std::thread(&OakCamera::thread_OakPipeline, this);
        }

        OakCamera::~OakCamera()
        {
            RCLCPP_INFO(get_logger(), "Destroying node");

            RCLCPP_INFO(get_logger(), "Waiting for pipeline thread...");
            try
            {
                if (mPipelineThread.joinable())
                {
                    mPipelineThread.join();
                }
            }
            catch (std::system_error &e)
            {
                RCLCPP_INFO(get_logger(), "Pipeline thread joining exception");
            }
            RCLCPP_INFO(get_logger(), "... Pipeline thread stopped");
        }

        void OakCamera::initPipeline()
        {
            // RCLCPP_INFO(get_logger(), "Initializing dai pipeline");
            pipeline = std::make_shared<dai::Pipeline>();
            // MonoCamera
            auto sync = pipeline->create<dai::node::Sync>();
            auto xoutGroup = pipeline->create<dai::node::XLinkOut>();
            auto monoLeft = pipeline->create<dai::node::MonoCamera>();
            auto monoRight = pipeline->create<dai::node::MonoCamera>();

            xoutGroup->setStreamName("xout");
            xoutGroup->input.setBlocking(false);
            sync->setSyncThreshold(std::chrono::milliseconds(syncThreshold));
            sync->setSyncAttempts(-1); // Infinite attempts

            dai::node::MonoCamera::Properties::SensorResolution monoResolutionProperties;
            if (monoResolution == "400p")
            {
                monoResolutionProperties = dai::node::MonoCamera::Properties::SensorResolution::THE_400_P;
                monoWidth = 640;
                monoHeight = 400;
            }
            else if (monoResolution == "480p")
            {
                monoResolutionProperties = dai::node::MonoCamera::Properties::SensorResolution::THE_480_P;
                monoWidth = 640;
                monoHeight = 480;
            }
            else
            {
                RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Invalid parameter. -> monoResolution: %s", monoResolution.c_str());
                throw std::runtime_error("Invalid mono camera resolution.");
            }
            monoLeft->setResolution(monoResolutionProperties);
            monoLeft->setBoardSocket(dai::CameraBoardSocket::CAM_B);
            monoLeft->setFps(lrFPS);
            monoRight->setResolution(monoResolutionProperties);
            monoRight->setBoardSocket(dai::CameraBoardSocket::CAM_C);
            monoRight->setFps(lrFPS);

            RCLCPP_INFO(get_logger(), "Camera resolution: %dx%d", monoWidth, monoHeight);
            RCLCPP_INFO(get_logger(), "Camera FPS: %d", lrFPS);

            auto manipLeft = pipeline->create<dai::node::ImageManip>();
            auto manipRight = pipeline->create<dai::node::ImageManip>();
            dai::RotatedRect rrLeft = {{monoLeft->getResolutionWidth() / 2.0f, monoLeft->getResolutionHeight() / 2.0f},
                                       {monoLeft->getResolutionWidth() * 1.0f, monoLeft->getResolutionHeight() * 1.0f},
                                       180};
            dai::RotatedRect rrRight = {{monoRight->getResolutionWidth() / 2.0f, monoRight->getResolutionHeight() / 2.0f},
                                        {monoRight->getResolutionWidth() * 1.0f, monoRight->getResolutionHeight() * 1.0f},
                                        180};
            manipLeft->initialConfig.setCropRotatedRect(rrLeft, false);
            monoLeft->out.link(manipLeft->inputImage);
            manipRight->initialConfig.setCropRotatedRect(rrRight, false);
            monoRight->out.link(manipRight->inputImage);

            auto stereo = pipeline->create<dai::node::StereoDepth>();
            stereo->initialConfig.setConfidenceThreshold(200);
            stereo->setRectifyEdgeFillColor(0); // black, to better see the cutout
            stereo->initialConfig.setLeftRightCheckThreshold(5);
            stereo->setLeftRightCheck(true);
            stereo->setExtendedDisparity(false);
            stereo->setSubpixel(true);
            manipLeft->out.link(stereo->left);
            manipRight->out.link(stereo->right);

            stereo->rectifiedLeft.link(sync->inputs["left"]);
            stereo->rectifiedRight.link(sync->inputs["right"]);

            sync->out.link(xoutGroup->input);
        }

        void OakCamera::initPubSub()
        {
            RCLCPP_INFO(get_logger(), "Initializing publishers");
            rclcpp::PublisherOptions pubOptions;
            pubOptions.qos_overriding_options = rclcpp::QosOverridingOptions::with_default_policies();

            leftImgPub = create_publisher<sensor_msgs::msg::Image>(LEFT_TOPIC_RECT, rclcpp::QoS(10), pubOptions);
            rightImgPub = create_publisher<sensor_msgs::msg::Image>(RIGHT_TOPIC_RECT, rclcpp::QoS(10), pubOptions);

            leftInfoPub = create_publisher<sensor_msgs::msg::CameraInfo>(LEFT_INFO_TOPIC, rclcpp::QoS(10), pubOptions);
            rightInfoPub = create_publisher<sensor_msgs::msg::CameraInfo>(RIGHT_INFO_TOPIC, rclcpp::QoS(10), pubOptions);
        }

        void OakCamera::thread_OakPipeline()
        {
            RCLCPP_INFO(get_logger(), "Pipeline thread started");
            device = std::make_shared<dai::Device>(*pipeline, dai::UsbSpeed::SUPER);
            auto calibrationHandler = device->readCalibration();

            auto leftConverter = std::make_shared<dai::ros::ImageConverter>("oak_left_camera_optical_frame", convInterleaved, convGetBaseDeviceTimestamp);
            auto rightConverter = std::make_shared<dai::ros::ImageConverter>("oak_right_camera_optical_frame", convInterleaved, convGetBaseDeviceTimestamp);

            if (convUpdateROSBaseTimeOnToRosMsg)
            {
                leftConverter->setUpdateRosBaseTimeOnToRosMsg(convUpdateROSBaseTimeOnToRosMsg);
                rightConverter->setUpdateRosBaseTimeOnToRosMsg(convUpdateROSBaseTimeOnToRosMsg);
            }

            if (convReverseSocketOrder)
            {
                leftConverter->reverseStereoSocketOrder();
                rightConverter->reverseStereoSocketOrder();
            }

            auto leftCameraInfo = leftConverter->calibrationToCameraInfo(calibrationHandler, dai::CameraBoardSocket::CAM_B, monoWidth, monoHeight);
            auto rightCameraInfo = rightConverter->calibrationToCameraInfo(calibrationHandler, dai::CameraBoardSocket::CAM_C, monoWidth, monoHeight);
            groupQueue = device->getOutputQueue("xout", 8, false);

            while (rclcpp::ok())
            {
                RCLCPP_INFO_ONCE(get_logger(), "Publishing started!");
                try
                {
                    auto msgGrp = groupQueue->get<dai::MessageGroup>();

                    auto leftImgData = msgGrp->get<dai::ImgFrame>("left");
                    auto rightImgData = msgGrp->get<dai::ImgFrame>("right");

                    if (!leftImgData || !rightImgData)
                    {
                        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "Received incomplete stereo message group");
                        continue;
                    }

                    auto leftRawMsg = leftConverter->toRosMsgRawPtr(leftImgData, leftCameraInfo);
                    leftCameraInfo.header = leftRawMsg.header;

                    auto rightRawMsg = rightConverter->toRosMsgRawPtr(rightImgData, rightCameraInfo);
                    rightCameraInfo.header = rightRawMsg.header;

                    leftImgPub->publish(leftRawMsg);
                    leftInfoPub->publish(leftCameraInfo);
                    rightImgPub->publish(rightRawMsg);
                    rightInfoPub->publish(rightCameraInfo);
                }
                catch (const std::exception &e)
                {
                    RCLCPP_ERROR(get_logger(), "Standard exception in pipeline thread: %s", e.what());
                }
                catch (...)
                {
                    rcutils_reset_error();
                    RCLCPP_ERROR(get_logger(), "Unknown exception in pipeline thread");
                }
            }
            RCLCPP_INFO(get_logger(), "Pipeline thread finished");
        }

    } // namespace depthai
} // namespace uosm

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(uosm::depthai::OakCamera)
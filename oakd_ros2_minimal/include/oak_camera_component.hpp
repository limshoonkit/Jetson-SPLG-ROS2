#ifndef OAK_CAMERA_COMPONENT_HPP_
#define OAK_CAMERA_COMPONENT_HPP_

#include "depthai/depthai.hpp"
#include "depthai/pipeline/Pipeline.hpp"
#include "depthai/pipeline/node/MonoCamera.hpp"
#include "depthai/pipeline/node/StereoDepth.hpp"
#include "depthai/pipeline/node/Sync.hpp"
#include "depthai/pipeline/node/ImageManip.hpp"
#include "depthai/pipeline/node/VideoEncoder.hpp"
#include "depthai/pipeline/node/XLinkOut.hpp"
#include "depthai_bridge/ImageConverter.hpp"

#include <image_transport/camera_publisher.hpp>
#include <image_transport/image_transport.hpp>
#include <image_transport/publisher.hpp>
#include <ffmpeg_image_transport_msgs/msg/ffmpeg_packet.hpp>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

#include <memory>
#include <string>
#include <vector>

namespace uosm
{
    namespace depthai
    {
        class OakCamera : public rclcpp::Node
        {
        public:
            explicit OakCamera(const rclcpp::NodeOptions &options);

            virtual ~OakCamera();

        protected:
            void init();
            void initPipeline();
            void initPubSub();

            void thread_OakPipeline();

        private:
            const std::string LEFT_TOPIC_RECT = "left/image_rect";
            const std::string RIGHT_TOPIC_RECT = "right/image_rect";

            const std::string LEFT_INFO_TOPIC = "left/camera_info";
            const std::string RIGHT_INFO_TOPIC = "right/camera_info";

            std::string monoResolution;
            int monoWidth, monoHeight;
            int lrFPS;
            int syncThreshold;
            bool convInterleaved, convGetBaseDeviceTimestamp, convUpdateROSBaseTimeOnToRosMsg, convReverseSocketOrder;

            std::shared_ptr<dai::Pipeline> pipeline;
            std::shared_ptr<dai::Device> device;
            std::shared_ptr<dai::DataOutputQueue> groupQueue;

            rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr leftImgPub, rightImgPub;
            rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr leftInfoPub, rightInfoPub;

            rclcpp::TimerBase::SharedPtr mInitTimer;

            std::thread mPipelineThread;
        };

    } // namespace depthai
} // namespace uosm

#endif // OAK_CAMERA_COMPONENT_HPP_
#include <rclcpp/rclcpp.hpp>
#include "splg_trt_component.hpp"

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<uosm::SuperPointLightGlueTrt>(rclcpp::NodeOptions());
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
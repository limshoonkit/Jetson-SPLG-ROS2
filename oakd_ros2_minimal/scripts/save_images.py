#!/usr/bin/env python3
import rclpy
import cv2
import os
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters

class ImageSaverNode(Node):
    def __init__(self):
        super().__init__('image_saver_node')
        self.get_logger().info("Image pair saver node started.")

        # Declare a parameter for the output directory
        self.declare_parameter('save_path', '~/data/calib_images')
        self.save_path = self.get_parameter('save_path').get_parameter_value().string_value
        self.save_path = os.path.expanduser(self.save_path) # Expands '~' to home dir

        # Create directory if it doesn't exist
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            self.get_logger().info(f"Created directory: {self.save_path}")

        self.bridge = CvBridge()
        self.counter = 0

        left_topic = "/left/image_rect"
        right_topic = "/right/image_rect"

        # Create subscribers
        self.left_sub = message_filters.Subscriber(self, Image, left_topic)
        self.right_sub = message_filters.Subscriber(self, Image, right_topic)

        # Use ApproximateTimeSynchronizer to get paired messages
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.left_sub, self.right_sub], 
            queue_size=10, 
            slop=0.1 # allowed time difference in seconds
        )
        self.ts.registerCallback(self.image_callback)

        self.get_logger().info(f"Listening for synchronized image pairs on '{left_topic}' and '{right_topic}'...")
        self.get_logger().info(f"Saving images to: {self.save_path}")

    def image_callback(self, left_msg, right_msg):
        try:
            # Convert ROS Image messages to OpenCV images
            cv_left = self.bridge.imgmsg_to_cv2(left_msg, "mono8")
            cv_right = self.bridge.imgmsg_to_cv2(right_msg, "mono8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert images: {e}")
            return

        # Create filenames with zero-padding for easy sorting
        # e.g., frame_0000_left.png, frame_0000_right.png
        left_filename = os.path.join(self.save_path, f"frame_{self.counter:04d}_left.png")
        right_filename = os.path.join(self.save_path, f"frame_{self.counter:04d}_right.png")

        # Save the images
        cv2.imwrite(left_filename, cv_left)
        cv2.imwrite(right_filename, cv_right)

        if self.counter % 20 == 0: # Log every 20 frames to avoid spam
            self.get_logger().info(f"Saved pair {self.counter:04d}")

        self.counter += 1

def main(args=None):
    rclpy.init(args=args)
    node = ImageSaverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
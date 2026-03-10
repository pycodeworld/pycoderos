#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2025, www.pycodeworld.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
功能：视频压缩
描述：默认订阅camera/image_raw主题的原始图像，发布为jpeg图片，可用网页显示等。
     在视频图像上进行图像处理建议直接订阅camera/image_raw，而不是本主题camera/image_jpeg。
作者：pycodeworld
'''

from .node_unique import unique_call
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
from threading import Lock
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult


class NodeCameraJpeg(Node):
    def __init__(self, name):
        super().__init__(name)

        # Declare parameters with default values
        self.declare_parameter('fps', 12)
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('quality', 80)
        self.declare_parameter('input', 'camera/image_raw')

        # Get parameters
        self.publish_rate = self.get_parameter('fps').value
        self.output_width = self.get_parameter('width').value
        self.output_height = self.get_parameter('height').value
        self.jpeg_quality = self.get_parameter('quality').value
        self.input_topic = self.get_parameter('input').value

        self.get_logger().info(
            f"视频压缩参数: 帧率={self.publish_rate}, 分辨率={self.output_width}x{self.output_height}, 图片质量={self.jpeg_quality}")

        self.input_topic = 'camera/image_raw'
        # Create publisher for compressed images
        self.compressed_pub = self.create_publisher(
            CompressedImage, 'camera/image_jpeg', qos_profile_sensor_data)

        self.subscription = self.create_subscription(
            Image, self.input_topic, self.image_callback, qos_profile_sensor_data)
        # CV bridge
        self.bridge = CvBridge()

        # Parameter callback
        self.add_on_set_parameters_callback(self.parameters_callback)

    def parameters_callback(self, params):
        for param in params:
            if param.name == 'fps':
                self.publish_rate = param.value
                self.timer.reset()
                self.timer = self.create_timer(
                    1.0 / self.publish_rate, self.timer_callback)
            elif param.name == 'width':
                self.output_width = param.value
            elif param.name == 'height':
                self.output_height = param.value
            elif param.name == 'quality':
                self.jpeg_quality = param.value
            elif param.name == 'input':
                self.input_topic = param.value
                self.destroy_subscription(self.subscription)
                self.subscription = self.create_subscription(
                    Image, self.input_topic, self.image_callback, 1)

        self.get_logger().info(
            f"视频压缩参数: 帧率={self.publish_rate}, 分辨率={self.output_width}x{self.output_height}, 图片质量={self.jpeg_quality}")
        return SetParametersResult(successful=True)

    def image_callback(self, msg):
        self.latest_image = msg

        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(
            self.latest_image, desired_encoding='passthrough')

        # Resize if needed
        if (self.output_width != cv_image.shape[1]) or (self.output_height != cv_image.shape[0]):
            cv_image = cv2.resize(
                cv_image,
                (self.output_width, self.output_height),
                interpolation=cv2.INTER_LINEAR
            )

        # Compress to JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        _, jpeg_data = cv2.imencode('.jpg', cv_image, encode_param)

        # Create and publish CompressedImage message
        compressed_msg = CompressedImage()
        compressed_msg.header = self.latest_image.header
        compressed_msg.format = "jpeg"
        compressed_msg.data = jpeg_data.tobytes()
        self.compressed_pub.publish(compressed_msg)


def main():
    unique_call(NodeCameraJpeg)


if __name__ == '__main__':
    main()

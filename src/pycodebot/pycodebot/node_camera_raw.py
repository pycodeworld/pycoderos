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
功能：视频采集
描述：USB摄像头视频图像采集
作者：pycodeworld
'''

from .node_unique import unique_call
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class NodeCameraRaw(Node):
    def __init__(self, name):
        super().__init__(name)

        # 声明参数及默认值
        self.declare_parameters(
            namespace='',
            parameters=[('device', "0"),
                        ('width', 640),
                        ('height', 480),
                        ('fps', 30)
                        ]
        )

        # 获取参数值
        self.device = self.get_parameter('device').value
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.fps = self.get_parameter('fps').value

        # 打印参数配置
        self.get_logger().info(
            f"摄像头配置: 设备={self.device}, 分辨率={self.width}x{self.height}, 帧率={self.fps}")

        # 创建发布者
        self.publisher_ = self.create_publisher(
            Image, 'camera/image_raw', qos_profile_sensor_data)
        self.bridge = CvBridge()

        # 用于FPS计算
        self.frame_count = 0
        self.last_time = self.get_clock().now()
        self.cap = None

        # 初始化摄像头
        self.init_camera()

        # 计算发布间隔(秒)
        timer_period = 1.0 / self.fps
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # 添加参数变更回调
        self.add_on_set_parameters_callback(self.parameter_callback)

    def find_camera_node(self):
        cap = cv2.VideoCapture(0)
        if cap and cap.isOpened():
            self.get_logger().info(f"摄像头设备已打开。")
            return cap

        for i in range(8):
            dev = f"/dev/video{i}"
            cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
            if cap and cap.isOpened():
                self.get_logger().info(f"摄像头设备已打开： {dev}")
                return cap
        return None

    def init_camera(self):
        """初始化摄像头设备"""
        if not self.device:
            return
        try:
            self.cap = self.find_camera_node()

            if not self.cap or not self.cap.isOpened():
                self.get_logger().error(f"无法打开摄像头设备: {self.device}")
                return

            # 设置摄像头参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            # 验证实际设置值
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))

            if actual_width != self.width or actual_height != self.height:
                self.get_logger().warn(
                    f"请求分辨率 {self.width}x{self.height} 不可用，实际使用 {actual_width}x{actual_height}")
                self.width, self.height = actual_width, actual_height

            if actual_fps != self.fps:
                self.get_logger().warn(
                    f"请求FPS {self.fps} 不可用，实际使用 {actual_fps}")
                self.fps = actual_fps

            self.get_logger().info(
                f"摄像头配置: 设备={self.device}, 分辨率={self.width}x{self.height}, 帧率={self.fps}")

        except Exception as e:
            self.get_logger().error(f"摄像头初始化异常: {str(e)}")
            raise

    def timer_callback(self):
        """定时器回调函数，捕获并发布帧"""
        if not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("从摄像头读取帧失败", throttle_duration_sec=5)
            return

        try:
            # 转换图像消息
            msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera_frame"
            self.publisher_.publish(msg)

        except Exception as e:
            self.get_logger().error(
                f"图像处理异常: {str(e)}", throttle_duration_sec=1)

    def parameter_callback(self, params):
        """参数变更回调函数"""
        updated = False
        for param in params:
            if param.name == 'fps' and param.type_ == Parameter.Type.INTEGER:
                if param.value > 0 and param.value != self.fps:
                    self.fps = param.value
                    self.timer.timer_period_ns = int(1e9 / self.fps)
                    self.get_logger().info(f"更新FPS为: {self.fps}")
                    updated = True

            elif param.name == 'width' and param.type_ == Parameter.Type.INTEGER:
                if param.value > 0 and param.value != self.width:
                    self.width = param.value
                    updated = True

            elif param.name == 'height' and param.type_ == Parameter.Type.INTEGER:
                if param.value > 0 and param.value != self.height:
                    self.height = param.value
                    updated = True

            elif param.name == 'device' and param.type_ == Parameter.Type.STRING:
                if param.value != self.device:
                    self.device = param.value
                    updated = True

        if updated:
            # 重新初始化摄像头
            self.cap and self.cap.release()
            self.init_camera()

        return rclpy.node.SetParametersResult(successful=True)

    def __del__(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.get_logger().info("摄像头资源已释放")


def main():
    unique_call(NodeCameraRaw)


if __name__ == '__main__':
    main()

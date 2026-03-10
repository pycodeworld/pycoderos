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
功能：图像识别基类
描述：可在元图像和压缩图像格式上做识别，并且支持发布识别后的图像和文字信息
作者：pycodeworld
'''

from abc import abstractmethod
import os
import cv2
import numpy as np
import rclpy
import json
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image as RosImage, CompressedImage
from std_msgs.msg import String
from cv_bridge import CvBridge

from PIL import ImageFont, ImageDraw, Image as PILImage


class RecognizerMessage:

    # value:  识别结果 >=0 表示识别成功 ， <0 表示失败错误吗
    # message：识别结果文字描述
    def __init__(self, value=0, message=None):
        self.value = value
        self.message = message


class AbsBaseRecognizer(Node):
    def __init__(self, name: str):
        super().__init__(name)

        # 用于统一发布的占位（子类在 image_callback 内更新）
        self.pub_image = None          # OpenCV BGR 图像（np.ndarray）
        self.pub_message = ""          # 文本消息（str）
        self.image_publisher = None

        # ---------------- 参数 ----------------
        self.declare_parameter('input_topic', 'camera/image_raw')
        self.declare_parameter('input_encoding', 'bgr8')  # 仅支持 'bgr8' 或 'jpeg'
        self.declare_parameter('output_fps', 6)
        self.declare_parameter('output_quality', 60)
        self.declare_parameter('output_image_close', False)

        self.input_topic = self.get_parameter('input_topic').value
        self.input_encoding = self.get_parameter('input_encoding').value
        self.pub_image_close = self.get_parameter('output_image_close').value
        self.output_quality = self.get_parameter('output_quality').value
        self.fps = self.get_parameter('output_fps').value
        if self.fps <= 0:
            self.fps = 5

        # ---------------- 订阅（按编码选择消息类型） ----------------
        self.subscription = self.create_subscription(
            RosImage if self.input_encoding == 'bgr8' else CompressedImage,
            self.input_topic,
            self._image_callback,  qos_profile_sensor_data)

        # ---------------- 发布：处理后的图像（CompressedImage） ----------------
        if not self.pub_image_close:
            self.image_publisher = self.create_publisher(
                CompressedImage,
                f'/{name}/image_jpeg',
                qos_profile_sensor_data)

        # ---------------- 发布：识别文字结果 ----------------
        self.posture_text_publisher = self.create_publisher(
            String, f'/{name}/message', 10)

        self.timer = self.create_timer(1.0/self.fps, self.timer_callback)
        self.bridge = CvBridge()
        self.font = self.load_chinese_font()
        self.input_msg = None

        # Parameter callback
        self.add_on_set_parameters_callback(self.parameters_callback)
        self.get_logger().info(
            f"{name}参数: 订阅主题={self.input_topic}, 是否发布图像={self.pub_image_close}, 图片格式={self.input_encoding}, 发布帧率={self.fps}, 发布图片质量={self.output_quality}")

    def parameters_callback(self, params):
        sub_changed = False
        for param in params:
            if param.name == 'output_image_close':
                self.pub_image_close = param.value
                if param.value and not self.image_publisher:
                    self.image_publisher = self.create_publisher(
                        CompressedImage, f'/{self.name}/image_jpeg', qos_profile_sensor_data)
                if not param.value and self.image_publisher:
                    self.destroy_publisher(self.image_publisher)
                    self.image_publisher = None
            elif param.name == "input_encoding":
                self.input_encoding = param.value
                sub_changed = True
            elif param.name == 'input_quality':
                self.jpeg_quality = param.value
            elif param.name == 'input_topic':
                self.input_topic = param.value
                sub_changed = True
            elif param.name == "output_fps":
                self.fps = param.value
                self.timer.cancel()
                self.destroy_timer(self.timer)
                self.timer = self.create_timer(
                    1.0/self.fps, self.timer_callback)

        if sub_changed:
            self.destroy_subscription(self.subscription)
            self.subscription = self.create_subscription(
                RosImage if self.input_encoding == 'bgr8' else CompressedImage,
                self.input_topic,
                self._image_callback,  qos_profile_sensor_data)

        self.get_logger().info(
            f"{self.get_name()}参数: 订阅主题={self.input_topic}, 是否发布图像={self.pub_image_close}, 图片格式={self.input_encoding}, 发布帧率={self.fps}, 发布图片质量={self.output_quality}")
        return rclpy.node.SetParametersResult(successful=True)

    # ---------------- 字体加载与中文绘制 ----------------
    def load_chinese_font(self):
        """加载中文字体（优先参数/环境变量，其次常见系统/本地字体；支持 .ttc 多索引）"""
        try:
            cfg_path = ''
            cfg_size = 30
            env_path = os.getenv('POSE_FONT_PATH', '')
            local_font = os.path.join(os.path.dirname(
                __file__), 'fonts', 'NotoSansSC-Regular.otf')

            candidates = [
                cfg_path, env_path, local_font,
                # macOS
                '/System/Library/Fonts/PingFang.ttc',
                '/System/Library/Fonts/STHeiti Light.ttc',
                '/System/Library/Fonts/STHeiti Medium.ttc',
                '/Library/Fonts/Songti.ttc',
                # Linux
                '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
                '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
                # Windows
                'C:/Windows/Fonts/simhei.ttf',
                'C:/Windows/Fonts/msyh.ttc',
                'C:/Windows/Fonts/msyh.ttf',
            ]
            candidates = [p for p in candidates if p and os.path.exists(p)]

            def try_load(path: str):
                if path.lower().endswith('.ttc'):
                    # .ttc 可能包含多个字形，逐个尝试索引
                    for idx in range(12):
                        try:
                            return ImageFont.truetype(path, cfg_size, index=idx)
                        except Exception:
                            continue
                    raise RuntimeError(f'.ttc 子字体索引均失败: {path}')
                else:
                    return ImageFont.truetype(path, cfg_size)

            for path in candidates:
                try:
                    font = try_load(path)
                    self.get_logger().info(f'中文字体已加载: {path}')
                    return font
                except Exception as e:
                    self.get_logger().warning(f'字体加载失败 {path}: {e}')

            self.get_logger().warning('未找到可用中文字体，回退到默认字体（可能出现乱码）')
            return ImageFont.load_default()

        except Exception as e:
            self.get_logger().error(f"字体加载错误: {str(e)}")
            return ImageFont.load_default()

    def draw_chinese_text(self, image, text, position, color=(0, 255, 0)):
        """使用 PIL 在 OpenCV 图像上绘制中文文本；image 为 BGR 格式"""
        img_pil = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text(position, text, font=self.font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def compressed_image(self, cv_image):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.output_quality]
        _, jpeg_data = cv2.imencode('.jpg', cv_image, encode_param)
        compressed = CompressedImage()
        compressed.header = self.input_msg.header
        compressed.format = "jpeg"
        compressed.data = jpeg_data.tobytes()
        return compressed

    # ---------------- 内部订阅回调（不要在子类重载） ----------------
    def _image_callback(self, msg):
        self.input_msg = msg

    def timer_callback(self):
        if not self.input_msg:
            return

        # 将 ROS 图像转换为 OpenCV（BGR）
        if self.input_encoding == 'bgr8':
            cv_image = self.bridge.imgmsg_to_cv2(
                self.input_msg, desired_encoding="bgr8")
        else:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(
                self.input_msg, desired_encoding="bgr8"
            )

        self.image_callback(cv_image.copy())
        if self.pub_message:
            self.posture_text_publisher.publish(
                String(data=json.dumps(self.pub_message.__dict__)))

        if self.pub_image_close:
            return

        # 发布处理后的图像（若启用且子类已写入 pub_image）,如果未检查则发布压缩图
        if self.pub_image is not None:
            processed_compressed = self.compressed_image(self.pub_image)
        elif self.input_encoding == 'bgr8':
            processed_compressed = self.compressed_image(cv_image)
        else:
            processed_compressed = self.input_msg
        self.image_publisher.publish(processed_compressed)

    # ---------------- 子类需要实现的方法 ----------------
    @abstractmethod
    def image_callback(self, cv_image):
        """
        子类实现：
          - 读取 BGR OpenCV 图像 cv_image
          - 处理后将 BGR 图像写入 self.pub_image（或置 None 表示不发布图像）
          - 将文本结果写入 self.pub_message（str）
        """
        pass

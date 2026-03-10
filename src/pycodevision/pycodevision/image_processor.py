#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2025
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

"""
功能：图像灰度化 / 二值化处理节点（默认压缩发布，且不降低分辨率）
描述：
  - 订阅/接收 BGR OpenCV 图像，按照参数做灰度化或二值化处理；
  - 灰度模式下以 JPEG 压缩发布；二值化模式下以 **PNG(1-bit)** 发布；
  - 压缩/编码仅做编码不做缩放，分辨率不变；
  - JPEG 压缩质量可通过参数动态调节。
作者：pycodeworld
"""

import io
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import CompressedImage

# 说明：基类提供了发布图像/消息的通道与生命周期管理
from .abs_base_recognition import AbsBaseRecognizer, RecognizerMessage

# Pillow 用于真正写出 1-bit PNG
try:
    from PIL import Image
    _PIL_AVAILABLE = True
except Exception:
    _PIL_AVAILABLE = False


class ImageProcessor(AbsBaseRecognizer):
    def __init__(self, name: str = 'image_processor'):
        super().__init__(name)

        # ---------------- 声明参数（默认灰度 + 默认压缩质量 80） ----------------
        self.declare_parameter('process_mode', 'gray')      # "gray" / "binary"
        self.declare_parameter('binary_threshold', 127)     # 0~255
        self.declare_parameter('use_otsu', False)           # bool OTSU（大津法）是一种自动选阈值的图像二值化方法。它不需要你手动给阈值，而是从灰度直方图里计算出一个阈值，使前景/背景两类分得最“开”。
        self.declare_parameter('jpeg_quality', 80)          # 0~100，默认 80（保持与原代码一致）

        self._load_params()
        self.add_on_set_parameters_callback(self._on_param_change)

        # 基类会在首次回调后发布 self.pub_image / self.pub_message
        self.pub_image = None
        self.pub_message = ""

        self.get_logger().info(
            f"图像处理节点已启动。当前参数：process_mode={self.process_mode}, "
            f"binary_threshold={self.binary_threshold}, use_otsu={self.use_otsu}, "
            f"jpeg_quality={self.jpeg_quality}; PIL={'OK' if _PIL_AVAILABLE else 'MISSING'}"
        )

    # ---------------- 参数读取/更新 ----------------
    def _load_params(self):
        self.process_mode = str(
            self.get_parameter('process_mode').get_parameter_value().string_value or 'gray'
        ).lower()
        if self.process_mode not in ('gray', 'binary'):
            self.get_logger().warn(f'非法 process_mode={self.process_mode}，已回退为 "gray"')
            self.process_mode = 'gray'

        self.binary_threshold = int(
            self.get_parameter('binary_threshold').get_parameter_value().integer_value or 127
        )
        self.binary_threshold = int(np.clip(self.binary_threshold, 0, 255))

        self.use_otsu = bool(
            self.get_parameter('use_otsu').get_parameter_value().bool_value
        )

        self.jpeg_quality = int(
            self.get_parameter('jpeg_quality').get_parameter_value().integer_value or 70
        )
        self.jpeg_quality = int(np.clip(self.jpeg_quality, 0, 100))

    def _on_param_change(self, params):
        super().parameters_callback(params)
        for p in params:
            if p.name == 'process_mode':
                val = str(p.value).lower()
                if val not in ('gray', 'binary'):
                    self.get_logger().error('process_mode 必须为 "gray" 或 "binary"')
                    return SetParametersResult(successful=False)
            elif p.name == 'binary_threshold':
                try:
                    v = int(p.value)
                except Exception:
                    self.get_logger().error('binary_threshold 必须为整数（0~255）')
                    return SetParametersResult(successful=False)
                if not (0 <= v <= 255):
                    self.get_logger().error('binary_threshold 超出范围（0~255）')
                    return SetParametersResult(successful=False)
            elif p.name == 'use_otsu':
                if not isinstance(p.value, bool):
                    self.get_logger().error('use_otsu 必须为布尔值')
                    return SetParametersResult(successful=False)
            elif p.name == 'jpeg_quality':
                try:
                    q = int(p.value)
                except Exception:
                    self.get_logger().error('jpeg_quality 必须为整数（0~100）')
                    return SetParametersResult(successful=False)
                if not (0 <= q <= 100):
                    self.get_logger().error('jpeg_quality 超出范围（0~100）')
                    return SetParametersResult(successful=False)

        self._load_params()
        self.get_logger().info(
            f"[参数已更新] process_mode={self.process_mode}, "
            f"binary_threshold={self.binary_threshold}, use_otsu={self.use_otsu}, "
            f"jpeg_quality={self.jpeg_quality}"
        )
        return SetParametersResult(successful=True)

    # ---------------- 图像处理核心 ----------------
    def process_image(self, bgr_image: np.ndarray) -> np.ndarray:
        """
        根据参数对图像做灰度化/二值化处理。
        返回：3 通道 BGR 图像（灰度或二值也会被转换回 BGR）。
        """
        # 统一先转灰度
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        if self.process_mode == 'gray':
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # 二值化
        if self.use_otsu:
            _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            th = int(np.clip(self.binary_threshold, 0, 255))
            _, bin_img = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)

    # ---------------- 覆写：压缩编码（binary→PNG 1-bit；gray→JPEG） ----------------
    def compressed_image(self, cv_image: np.ndarray) -> CompressedImage:
        """
        由基类在发布前调用。
        - 灰度模式：以 JPEG + 可调质量编码；
        - 二值化模式：以 **PNG(1-bit)** 编码（真正 1 位深，非 8 位灰度）。
        均不缩放分辨率。
        返回：sensor_msgs.msg.CompressedImage
        """
        if not isinstance(cv_image, np.ndarray):
            raise TypeError("compressed_image 期望传入 np.ndarray")

        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()

        # ---------- 二值化模式：发布 1-bit PNG ----------
        if self.process_mode == 'binary':
            if not _PIL_AVAILABLE:
                # 明确提示
                err = ("Pillow 未安装，无法写出 1-bit PNG。请安装：pip install pillow\n"
                       "已终止发布。")
                self.get_logger().error(err)
                raise RuntimeError(err)

            # 从 BGR(0/255) 还原到单通道二值
            if cv_image.ndim == 3 and cv_image.shape[2] == 3:
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = cv_image

            # 确保仅 0/255
            bin_img = np.where(gray > 0, 255, 0).astype(np.uint8)  # (H, W), 0 or 255

            # 使用 Pillow 写出真正 1-bit PNG（mode='1'）
            pil_img = Image.fromarray(bin_img, mode='L').convert('1', dither=Image.NONE)
            buf = io.BytesIO()
            pil_img.save(buf, format='PNG', optimize=True)
            data = buf.getvalue()

            raw_bytes = int(bin_img.size)  # 以 1 像素=1 字节（L 模式）为参考的未打包体积
            comp_bytes = len(data)
            self.get_logger().info(
                f"[compressed_image] (binary PNG-1bit) raw_ref={raw_bytes}B, "
                f"compressed={comp_bytes}B, ratio={comp_bytes/max(raw_bytes,1):.3f}"
            )

            msg.format = "png; bit_depth=1"
            msg.data = data
            return msg

        # ---------- 灰度模式：沿用原 JPEG 流程 ----------
        # 不做 resize，确保分辨率不变
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)]
        ok, buf = cv2.imencode('.jpg', cv_image, encode_param)
        if not ok:
            # 兜底
            self.get_logger().error(f"JPEG(quality={self.jpeg_quality}) 编码失败，回退到 JPEG(80)")
            ok2, buf = cv2.imencode('.jpg', cv_image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok2:
                raise RuntimeError("图像压缩编码失败")

        raw_bytes = int(cv_image.nbytes)
        comp_bytes = int(buf.size)
        ratio = (comp_bytes / raw_bytes) if raw_bytes > 0 else 0.0
        self.get_logger().info(
            f"[compressed_image] (gray JPEG) raw={raw_bytes}B, compressed={comp_bytes}B, "
            f"ratio={ratio:.3f}, quality={self.jpeg_quality}"
        )

        msg.format = "jpeg"
        msg.data = buf.tobytes()
        return msg

    # ---------------- 基类回调：接收与发布 ----------------
    def image_callback(self, cv_image):
        """
        接收 BGR OpenCV 图像 → 灰度/二值化 → 发布（发布前会由基类调用 compressed_image 进行压缩）：
          - 当 self.pub_image_close 为 True：不发布图像，仅发布文字（状态）。
          - 当 self.pub_image_close 为 False：发布处理后的图像（BGR），由基类在内部压缩后输出。
        """
        try:
            if not isinstance(cv_image, np.ndarray):
                raise TypeError("image_callback 接收的不是有效的 np.ndarray 图像")

            # 处理图像
            processed = self.process_image(cv_image)

            # 发布逻辑（保持 ndarray，兼容基类压缩流程）
            if not self.pub_image_close:
                self.pub_image = processed  # 交由基类在发布环节调用 compressed_image 进行编码

            # 文本状态（复用 RecognizerMessage 结构）
            status_text = (
                f"processed_mode={self.process_mode}"
                + (f", use_otsu={self.use_otsu}" if self.process_mode == 'binary' else "")
                + (f", threshold={self.binary_threshold}" if (self.process_mode == 'binary' and not self.use_otsu) else "")
                + (f", publish=PNG(1-bit)" if self.process_mode == 'binary' else f", jpeg_quality={self.jpeg_quality}")
            )
            self.pub_message = RecognizerMessage(0, status_text)
            self.get_logger().info(f"图像已处理：{status_text}")

        except Exception as e:
            self.get_logger().error(f"图像处理错误: {e}")
            self.pub_message = RecognizerMessage(-1, "处理错误")


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ImageProcessor()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node:
            node.get_logger().info("节点被用户终止")
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

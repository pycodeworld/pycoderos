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
功能：颜色识别
描述：可以识别图像中的多种颜色，并把识别结果绘制在图像上
作者：pycodeworld
'''

import rclpy
import cv2
import json
import numpy as np
from collections import Counter
from .abs_base_recognition import AbsBaseRecognizer, RecognizerMessage


class ColorRecognizer(AbsBaseRecognizer):
    def __init__(self, name: str = 'color_recognition'):
        super().__init__(name)

        # ---------------- 颜色库（HSV区间，OpenCV: H∈[0,179], S/V∈[0,255]）----------------
        self.color_order = [
            '粉色', '棕色',
            '红色', '橙色', '黄色', '黄绿', '绿色', '青绿', '青色', '天蓝', '蓝色', '靛蓝',
            '紫色', '品红',
            '白色', '灰色', '黑色'
        ]

        self.color_ranges = {
            # 具体色（优先）
            '粉色': [((170, 40, 180), (179, 170, 255)), ((0, 40, 180), (12, 170, 255))],
            '棕色': [((10, 120, 30), (20, 255, 150))],

            # 基础色相更细切分
            '红色': [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (179, 255, 255))],
            '橙色': [((10, 120, 80), (20, 255, 255))],
            '黄色': [((20, 120, 120), (30, 255, 255))],
            '黄绿': [((30, 80, 70), (40, 255, 255))],
            '绿色': [((40, 80, 70), (70, 255, 255))],
            '青绿': [((70, 80, 70), (85, 255, 255))],
            '青色': [((85, 80, 70), (95, 255, 255))],
            '天蓝': [((95, 80, 70), (105, 255, 255))],
            '蓝色': [((105, 100, 70), (130, 255, 255))],
            '靛蓝': [((130, 80, 70), (140, 255, 255))],
            '紫色': [((140, 80, 70), (160, 255, 255))],
            '品红': [((160, 80, 70), (170, 255, 255))],
            # 若需要也可放开无彩色阈值：
            # '白色': [((0, 0, 200), (179, 40, 255))],
            # '灰色': [((0, 0, 60), (179, 40, 200))],
            # '黑色': [((0, 0, 0), (179, 255, 55))],
        }

        # 绘制颜色（BGR）
        self.draw_bgr = {
            '粉色': (203, 192, 255),
            '棕色': (19, 69, 139),
            '红色': (0, 0, 255),
            '橙色': (0, 165, 255),
            '黄色': (0, 215, 255),
            '黄绿': (35, 220, 120),
            '绿色': (0, 180, 0),
            '青绿': (128, 180, 70),
            '青色': (180, 180, 0),
            '天蓝': (255, 200, 100),
            '蓝色': (255, 0, 0),
            '靛蓝': (180, 0, 80),
            '紫色': (180, 0, 180),
            '品红': (255, 0, 255),
            '白色': (220, 220, 220),
            '灰色': (128, 128, 128),
            '黑色': (30, 30, 30),
        }

        # 中文基础色 → 英文名映射（用于 message.name）
        self.cn2en = {
            '粉色': 'pink',
            '棕色': 'brown',
            '红色': 'red',
            '橙色': 'orange',
            '黄色': 'yellow',
            '黄绿': 'yellowgreen',
            '绿色': 'green',
            '青绿': 'teal',
            '青色': 'cyan',
            '天蓝': 'skyblue',
            '蓝色': 'blue',
            '靛蓝': 'indigo',
            '紫色': 'purple',
            '品红': 'magenta',
            '白色': 'white',
            '灰色': 'gray',
            '黑色': 'black',
        }

        # ---------------- 参数 ----------------
        self.min_area_ratio = 0.003          # 过滤很小的噪声，按画面比例
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.use_refined_label = True        # 是否根据均值HSV加“浅/深/淡”前缀
        self.small_blur = True               # 进入HSV前做轻微去噪（应对感光抖动）

        self.pub_image = None
        self.pub_message = None  # 将存放 RecognizerMessage
        self.get_logger().info("颜色识别（多色+细粒度）节点已启动")

    # 合并多个区间掩膜并做形态学去噪
    def _mask_for_ranges(self, hsv, ranges):
        mask_total = None
        for (lo, hi) in ranges:
            mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            mask_total = mask if mask_total is None else cv2.bitwise_or(mask_total, mask)
        # 形态学：开->闭，连通 & 去小孔
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, self.kernel, iterations=1)
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        return mask_total

    # 计算轮廓区域的均值HSV（用于细粒度命名）
    def _mean_hsv_of_contour(self, hsv, contour):
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
        mean = cv2.mean(hsv, mask=mask)  # H,S,V,alpha
        return mean[:3]  # (H, S, V)

    # 根据均值HSV细化标签：浅/深/淡
    def _refine_label(self, base_name, mean_hsv):
        if base_name in ('白色', '灰色', '黑色'):
            return base_name
        if mean_hsv is None:
            return base_name
        H, S, V = mean_hsv
        prefix = ""
        if V >= 200 and S >= 60:
            prefix = "浅"
        elif V <= 80:
            prefix = "深"
        elif S <= 60 and V >= 120:
            prefix = "淡"
        return f"{prefix}{base_name}" if prefix else base_name

    # 去除“浅/深/淡”前缀，得到基础中文色名，用于英文映射与 cn_name
    def _base_cn_color(self, label: str) -> str:
        for p in ("浅", "深", "淡"):
            if label.startswith(p):
                return label[len(p):]
        return label

    # 主检测：为每种颜色找所有连通区域，避免重复覆盖（occupied 掩膜）
    def _detect_all_colors(self, bgr):
        img = bgr
        if self.small_blur:
            img = cv2.GaussianBlur(img, (5, 5), 0)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w = bgr.shape[:2]
        min_area = self.min_area_ratio * h * w

        detections = []   # [(label, bbox, area)]
        counts = Counter()
        occupied = np.zeros((h, w), dtype=np.uint8)  # 已被某色归属的像素（避免重复匹配）

        for name in self.color_order:
            ranges = self.color_ranges.get(name)
            if not ranges:
                continue
            mask = self._mask_for_ranges(hsv, ranges)

            # 去除已占用区域
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(occupied))

            # 轻微膨胀帮助连通零碎小孔，但不要太大
            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, self.kernel, iterations=1)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = float(cv2.contourArea(c))
                if area < min_area:
                    continue
                x, y, bw, bh = cv2.boundingRect(c)
                x1, y1, x2, y2 = x, y, x + bw, y + bh

                mean_hsv = self._mean_hsv_of_contour(hsv, c)
                label = self._refine_label(name, mean_hsv) if self.use_refined_label else name
                detections.append((label, (x1, y1, x2, y2), area))
                counts[label] += 1

                # 标记占用
                cv2.drawContours(occupied, [c], -1, 255, thickness=-1)

        # 统计总体覆盖比例
        coverage = float(np.count_nonzero(occupied)) / float(h * w + 1e-6)
        return detections, counts, coverage, (h, w)

    # 与原节点一致：识别到才更新文字；未识别到透传原图
    def image_callback(self, cv_image):
        try:
            detections, counts, coverage, (h, w) = self._detect_all_colors(cv_image)
            drew_anything = False

            # 1) 画框与标签
            if detections and not self.pub_image_close:
                for label, (x1, y1, x2, y2), _ in detections:
                    base_cn = self._base_cn_color(label)
                    color = self.draw_bgr.get(base_cn, (0, 255, 0))
                    cv2.rectangle(cv_image, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), color, 2)
                    cv_image = self.draw_chinese_text(
                        cv_image, label, (x1 - 3, max(y1 - 32, 0)), color=color
                    )
                self.pub_image = cv_image

            # 2) 构造 RecognizerMessage 的 message 数组
            message_list = []
            for label, (x1, y1, x2, y2), area in detections:
                base_cn = self._base_cn_color(label)          # 去前缀后的中文基础色
                en_name = self.cn2en.get(base_cn, 'unknown')  # 英文名
                width = int(x2 - x1)
                height = int(y2 - y1)
                area_ratio = float(area / (h * w + 1e-6))
                item = {
                    "name": en_name,
                    "cn_name": base_cn,
                    "left": int(x1),
                    "top": int(y1),
                    "height": int(height),
                    "width": int(width),
                    "image_height": int(height),
                    "window_width": int(width),
                }
                message_list.append(item)

            # 3) 发布文字消息（使用 RecognizerMessage）
            if detections:
                result_msg = RecognizerMessage(value=len(message_list), message=message_list)
                self.pub_message = result_msg
                drew_anything = True

                # 友好的人类可读日志（保持原有统计 + 覆盖率）
                parts = [f"{k}×{v}" for k, v in counts.items()]
                msg = ", ".join(parts)
                msg = f"{msg}；覆盖≈{coverage*100:.1f}%"

                # 结构化联调日志：value:message
                try:
                    self.get_logger().info(
                        f"value:{result_msg.value} message:{json.dumps(result_msg.message, ensure_ascii=False)}"
                    )
                except Exception:
                    # 保险：fallback 到 str 打印
                    self.get_logger().info(f"value:{result_msg.value} message:{result_msg.message}")

            # 未检测到：透传原图，并发布空结果（value=0, message=[]）
            if not drew_anything:
                if not self.pub_image_close:
                    self.pub_image = cv_image
                empty_msg = RecognizerMessage(value=0, message=[])
                self.pub_message = empty_msg
                self.get_logger().info("value:0:未检测到颜色")

        except Exception as e:
            self.get_logger().error(f"图像处理错误: {e}")
            try:
                if not self.pub_image_close:
                    self.pub_image = cv_image
            except Exception:
                pass
            # 出错时，发布失败（负值表示错误码，这里统一用 -1）
            try:
                self.pub_message = RecognizerMessage(value=-1, message=str(e))
                self.get_logger().info(f"values:-1:{str(e)}")
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ColorRecognizer()
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

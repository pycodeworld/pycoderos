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

"""
功能：姿态识别
描述：识别四种姿态：站立、坐着、蹲下、躺下
作者：pycodeworld
"""

import rclpy
from std_msgs.msg import String
import cv2
import mediapipe as mp
import numpy as np
import math
import json  # 新增：用于日志按 JSON 打印 message

from .abs_base_recognition import AbsBaseRecognizer, RecognizerMessage


class PoseRecognizer(AbsBaseRecognizer):
    def __init__(self, name: str = 'posture_recognition'):
        super().__init__(name)

        # 初始化姿态识别模型
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 定义姿态标签 (中文)
        self.pose_labels = {
            "standing": "站立",
            "sitting":  "坐着",
            "squatting": "蹲下",
            "lying":    "躺下",
            "unknown":  "未知姿态"
        }

        self.get_logger().info("姿态识别子类已启动，支持识别：站立、坐着、蹲下、躺下；未识别到姿势时将发布原图像")

        # 供基类的 _image_callback 使用：初始化占位
        self.pub_image = None       # 将被设置为处理后的 BGR 图（或 None）
        self.pub_message = None     # 发布 RecognizerMessage 实例

    # ---------- 工具方法 ----------
    def calculate_angle(self, p1, p2, p3):
        """计算三个点形成的角度（单位：度）"""
        v1 = np.array([p1.x - p2.x, p1.y - p2.y], dtype=np.float32)
        v2 = np.array([p3.x - p2.x, p3.y - p2.y], dtype=np.float32)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0.0
        cos_angle = float(np.dot(v1, v2) / (n1 * n2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    def get_body_ratio(self, landmarks, image_width, image_height):
        """计算身体比例和关键角度"""
        mp_pose = self.mp_pose
        nose = landmarks.landmark[mp_pose.PoseLandmark.NOSE.value]
        l_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_knee = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_ankle = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        shoulder_cy = (l_shoulder.y + r_shoulder.y) / 2
        hip_cy = (l_hip.y + r_hip.y) / 2
        knee_cy = (l_knee.y + r_knee.y) / 2
        ankle_cy = (l_ankle.y + r_ankle.y) / 2

        total_h = abs(nose.y - ankle_cy)
        safe_div = (lambda num: num / total_h) if total_h > 0 else (lambda num: 0.0)

        shoulder_hip_ratio = safe_div(abs(shoulder_cy - hip_cy))
        hip_knee_ratio = safe_div(abs(hip_cy - knee_cy))
        knee_ankle_ratio = safe_div(abs(knee_cy - ankle_cy))

        l_knee_angle = self.calculate_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = self.calculate_angle(r_hip, r_knee, r_ankle)
        avg_knee_angle = (l_knee_angle + r_knee_angle) / 2.0

        trunk_angle = math.degrees(math.atan2(
            abs((l_shoulder.x + r_shoulder.x) / 2 - (l_hip.x + r_hip.x) / 2),
            abs(shoulder_cy - hip_cy) + 1e-6  # 防 0
        ))

        return {
            'shoulder_hip_ratio': shoulder_hip_ratio,
            'hip_knee_ratio': hip_knee_ratio,
            'knee_ankle_ratio': knee_ankle_ratio,
            'knee_angle': avg_knee_angle,
            'trunk_angle': trunk_angle,
            'total_height': total_h
        }

    def recognize_pose(self, landmarks, image_h, image_w):
        """根据关键点判断姿态（返回英文枚举：standing/sitting/squatting/lying/unknown）"""
        try:
            bm = self.get_body_ratio(landmarks, image_w, image_h)
            sh_hip = bm['shoulder_hip_ratio']
            hip_k = bm['hip_knee_ratio']
            k_ank = bm['knee_ankle_ratio']
            knee = bm['knee_angle']
            trunk = bm['trunk_angle']

            trunk_straight = (knee > 150 and sh_hip > 0.25)
            if trunk_straight:
                return "lying" if trunk >= 45 else "standing"

            if trunk > 60 or sh_hip < 0.15:
                return "lying"
            elif (sh_hip > 0.35 and knee > 160 and trunk < 20 and hip_k > 0.25):
                return "standing"
            elif (knee < 90 and hip_k < 0.15 and sh_hip > 0.20 and trunk < 30):
                return "squatting"
            elif (90 <= knee <= 140 and sh_hip > 0.25 and trunk < 25 and hip_k > 0.15):
                return "sitting"
            else:
                if knee < 100:
                    return "squatting" if sh_hip > 0.20 else "sitting"
                elif sh_hip > 0.40:
                    return "standing"
                elif trunk > 45:
                    return "lying"
                else:
                    return "sitting"
        except Exception as e:
            self.get_logger().error(f"姿态识别错误: {e}")
            return "unknown"

    def draw_pose_info(self, image, landmarks, pose_type):
        """绘制姿态文本与调试指标（中文使用基类字体）"""
        try:
            h, w = image.shape[:2]
            bm = self.get_body_ratio(landmarks, w, h)
            info = [
                f"姿态: {self.pose_labels.get(pose_type, '未知')}",
                f"膝盖角度: {bm['knee_angle']:.1f}°",
                f"躯干角度: {bm['trunk_angle']:.1f}°"
            ]
            y = 30
            for i, txt in enumerate(info):
                image = self.draw_chinese_text(
                    image, txt, (10, y + i * 35),
                    color=(0, 255, 0) if i == 0 else (255, 255, 255)
                )
            return image
        except Exception as e:
            self.get_logger().error(f"绘制信息错误: {e}")
            return image

    # ---------- 关键：实现基类要求的回调 ----------
    def image_callback(self, cv_image):
        """
        基类 AbsBaseRecognizer 会把 OpenCV 图像传进来。
        我们在这里更新：
           - self.pub_image: 处理后的 BGR 图像（供基类发布）；若关闭发布，则置为 None
           - self.pub_message: RecognizerMessage(value=识别数量, message=[{每个结果的字段...}, ...])
        结果字段约定：
            {
                "name": en_name,           # 英文姿态：standing/sitting/squatting/lying/unknown
                "cn_name": cn_name,        # 中文姿态
                "left": int(x1),           # 边界框左上角 x
                "top": int(y1),            # 边界框左上角 y
                "height": int(h_box),      # 边界框高度
                "width": int(w_box),       # 边界框宽度
                "image_height": int(img_h),
                "image_width": int(img_w)
            }
        """
        try:
            h, w = cv_image.shape[:2]
            rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            results = self.pose.process(rgb)

            # ------- 结果数组（即 message），单人模型：0 或 1 个 -------
            result_items = []

            if results.pose_landmarks:
                pose_type = self.recognize_pose(
                    results.pose_landmarks,
                    h, w
                )
                en_name = pose_type
                cn_name = self.pose_labels.get(pose_type, "未知姿态")

                # 可视化（仅在未关闭图像发布时）
                if not self.pub_image_close:
                    mp.solutions.drawing_utils.draw_landmarks(
                        cv_image,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        mp.solutions.drawing_styles.get_default_pose_landmarks_style()
                    )

                # 计算边界框（基于可见关键点范围）
                xs = [lm.x * w for lm in results.pose_landmarks.landmark]
                ys = [lm.y * h for lm in results.pose_landmarks.landmark]
                x_min = int(max(min(xs), 0))
                x_max = int(min(max(xs), w - 1))
                y_min = int(max(min(ys), 0))
                y_max = int(min(max(ys), h - 1))
                box_w = max(0, x_max - x_min)
                box_h = max(0, y_max - y_min)

                if not self.pub_image_close:
                    cv2.rectangle(cv_image, (x_min - 10, y_min - 10),
                                  (x_max + 10, y_max + 10), (0, 255, 0), 2)
                    self.pub_image = self.draw_pose_info(cv_image, results.pose_landmarks, pose_type)

                # ---- 按你要求的字段组织一条结果 ----
                item = {
                    "name": en_name,
                    "cn_name": cn_name,
                    "left": int(x_min),
                    "top": int(y_min),
                    "height": int(box_h),
                    "width": int(box_w),
                    "image_height": int(h),
                    "image_width": int(w),
                }
                result_items.append(item)
            else:
                # ---- 新增逻辑：未检测到姿态时发布未经处理的原图像 ----
                if not self.pub_image_close:
                    # 直接发布原始帧（不画框、不画文案）
                    self.pub_image = cv_image.copy()

            # 识别数量（单人：0 或 1）
            value_count = len(result_items)

            # ---- 日志：value:message（message 用 JSON 打印）----
            self.get_logger().info(f"{value_count}:{json.dumps(result_items, ensure_ascii=False)}")

            # ---- 发布 RecognizerMessage ----
            self.pub_message = RecognizerMessage(value=value_count, message=result_items)

        except Exception as e:
            self.get_logger().error(f"图像处理错误: {e}")
            # 出错：value 设为负（错误码），message 给空数组（契合“带类数组”的约定）
            self.pub_message = RecognizerMessage(value=-1, message=[])

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PoseRecognizer()
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

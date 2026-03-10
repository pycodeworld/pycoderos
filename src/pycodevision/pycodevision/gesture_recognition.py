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
功能：手势识别
描述：识别六种手势：0 握拳、1 张开手掌、2 指向、3 剪刀手、4 大拇指、5 摇滚
作者：pycodeworld
"""

import rclpy
import cv2
import mediapipe as mp
import numpy as np
from .abs_base_recognition import AbsBaseRecognizer, RecognizerMessage


class GestureRecognizer(AbsBaseRecognizer):
    def __init__(self, name: str = 'gesture_recognition'):
        super().__init__(name)

        # 初始化 MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        # 手势标签（中文）
        self.gesture_labels = {
            0: "握拳",
            1: "张开手掌",
            2: "指向",
            3: "剪刀手",
            4: "大拇指",
            5: "摇滚",
        }

        # 基类会在首次回调后发布 self.pub_image / self.pub_message
        self.pub_image = None
        self.pub_message = ""

        self.get_logger().info(
            "手势识别子类已启动，可识别：握拳/张开手掌/指向/剪刀手/大拇指/摇滚"
        )

    # ------------ 手势判定逻辑（保留你原有思路） ------------
    def recognize_gesture(self, landmarks):
        """根据手部关键点识别手势（使用归一化坐标的几何关系）"""
        thumb_tip = np.array(
            [landmarks.landmark[4].x,  landmarks.landmark[4].y])
        index_tip = np.array(
            [landmarks.landmark[8].x,  landmarks.landmark[8].y])
        middle_tip = np.array(
            [landmarks.landmark[12].x, landmarks.landmark[12].y])
        ring_tip = np.array(
            [landmarks.landmark[16].x,  landmarks.landmark[16].y])
        pinky_tip = np.array(
            [landmarks.landmark[20].x,  landmarks.landmark[20].y])
        wrist = np.array([landmarks.landmark[0].x,   landmarks.landmark[0].y])

        thumb_dist = float(np.linalg.norm(thumb_tip - wrist))
        index_dist = float(np.linalg.norm(index_tip - wrist))
        middle_dist = float(np.linalg.norm(middle_tip - wrist))
        ring_dist = float(np.linalg.norm(ring_tip - wrist))
        pinky_dist = float(np.linalg.norm(pinky_tip - wrist))

        if all(dist > 0.15 for dist in [index_dist, middle_dist, ring_dist, pinky_dist]):
            return 1 if thumb_dist > 0.2 else 0
        elif index_dist > 0.15 and middle_dist < 0.1 and ring_dist < 0.1 and pinky_dist < 0.1:
            return 2
        elif index_dist > 0.15 and middle_dist > 0.15 and ring_dist < 0.1 and pinky_dist < 0.1:
            return 3
        elif thumb_dist > 0.2 and index_dist < 0.1 and middle_dist < 0.1 and ring_dist < 0.1 and pinky_dist < 0.1:
            return 4
        else:
            return 5

    # ------------ 关键：实现基类要求的回调（支持彻底跳过可视化） ------------
    def image_callback(self, cv_image):
        """
        接收 BGR OpenCV 图像 → 手势检测 → 更新：
          - 当 self.pub_image_close 为 True：彻底跳过可视化，仅发布文字，self.pub_image=None
          - 当 self.pub_image_close 为 False：叠加可视化并发布图像与文字
        """
        try:
            # 仅用于推理的输入（始终需要转 RGB 供 MediaPipe 使用）
            rgb_for_infer = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            results = self.hands.process(rgb_for_infer)
            gesture_text = "未检测到手部"

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                gesture_id = self.recognize_gesture(hand_landmarks)
                gesture_text = self.gesture_labels.get(gesture_id, "未知手势")

                if not self.pub_image_close:
                    # 只有需要发布图像时，才进行所有可视化绘制
                    # 关键点/连线
                    self.mp_drawing.draw_landmarks(
                        cv_image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(
                            color=(121, 22, 76), thickness=2, circle_radius=4),
                        self.mp_drawing.DrawingSpec(
                            color=(250, 44, 250), thickness=2, circle_radius=2),
                    )

                    # 边界框
                    h, w = cv_image.shape[:2]
                    xs = [lm.x * w for lm in hand_landmarks.landmark]
                    ys = [lm.y * h for lm in hand_landmarks.landmark]
                    x_min, x_max = int(max(min(xs), 0)), int(
                        min(max(xs), w - 1))
                    y_min, y_max = int(max(min(ys), 0)), int(
                        min(max(ys), h - 1))
                    cv2.rectangle(cv_image, (x_min - 10, y_min - 10),
                                  (x_max + 10, y_max + 10), (0, 255, 0), 2)

                    # 标签（中文）
                    cv_image = self.draw_chinese_text(
                        cv_image,
                        gesture_text,
                        (x_min - 10, max(y_min - 40, 0)),
                        color=(0, 255, 0),
                    )

                    # 发布图像
                    self.pub_image = cv_image

            # 始终发布文字消息
            self.pub_message = RecognizerMessage(gesture_id, gesture_text)
            self.get_logger().info(f"识别结果: {gesture_text}")
            
        except Exception as e:
            self.get_logger().error(f"图像处理错误: {e}")
            # 出错时：遵循是否发布图像的开关
            self.pub_message = RecognizerMessage(-1, "处理错误")


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = GestureRecognizer()
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

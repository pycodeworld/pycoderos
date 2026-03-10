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
功能：猜拳识别（仅输出三类）
描述：识别三种手势：rock 石头 / scissors 剪刀 / paper 布
作者：pycodeworld
"""

import rclpy
import cv2
import mediapipe as mp
import numpy as np
import math
from .abs_base_recognition import AbsBaseRecognizer, RecognizerMessage


class GestureRecognizerRps(AbsBaseRecognizer):
    def __init__(self, name: str = 'gesture_recognition_rps'):
        super().__init__(name)

        # 初始化 MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        # 直接用英文→中文映射
        self.rps_labels = {
            "rock": "石头",
            "scissors": "剪刀",
            "paper": "布",
        }

        self.pub_image = None
        self.pub_message = None  # 发布 RecognizerMessage 对象

        self.get_logger().info("猜拳识别节点已启动：可识别 石头/剪刀/布")

    # -------------------- 几何工具函数 --------------------
    @staticmethod
    def _angle_deg(a, b, c):
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        nba = ba / (np.linalg.norm(ba) + 1e-6)
        nbc = bc / (np.linalg.norm(bc) + 1e-6)
        cosang = float(np.clip(np.dot(nba, nbc), -1.0, 1.0))
        return math.degrees(math.acos(cosang))

    def _finger_states_by_angles(self, lms):
        EXT_TH = 160.0
        FLEX_TH = 120.0

        def xy(idx):
            return (lms.landmark[idx].x, lms.landmark[idx].y)

        states = {}

        def one_finger(mcp, pip, tip):
            ang = self._angle_deg(xy(mcp), xy(pip), xy(tip))
            if ang >= EXT_TH:
                return True
            elif ang <= FLEX_TH:
                return False
            else:
                return None

        states['index']  = one_finger(5,  6,  8)
        states['middle'] = one_finger(9, 10, 12)
        states['ring']   = one_finger(13,14, 16)
        states['pinky']  = one_finger(17,18, 20)

        ang_thumb_mcp = self._angle_deg(xy(1), xy(2), xy(4))
        if ang_thumb_mcp >= 155:
            states['thumb'] = True
        elif ang_thumb_mcp <= 115:
            states['thumb'] = False
        else:
            ang_thumb_ip = self._angle_deg(xy(2), xy(3), xy(4))
            if ang_thumb_ip >= 155:
                states['thumb'] = True
            elif ang_thumb_ip <= 115:
                states['thumb'] = False
            else:
                states['thumb'] = None

        return states

    # -------------------- 核心：猜拳判定 --------------------
    def recognize_rps(self, hand_landmarks):
        """
        返回 (英文, 中文)；若不确定 → (None, None)
        """
        lms = hand_landmarks
        states = self._finger_states_by_angles(lms)

        wrist = np.array([lms.landmark[0].x, lms.landmark[0].y])
        idx_mcp   = np.array([lms.landmark[5].x,  lms.landmark[5].y])
        mid_mcp   = np.array([lms.landmark[9].x,  lms.landmark[9].y])
        pinky_mcp = np.array([lms.landmark[17].x, lms.landmark[17].y])

        palm_width = float(np.linalg.norm(idx_mcp - pinky_mcp) + 1e-6)
        palm_len   = float(np.linalg.norm(mid_mcp - wrist) + 1e-6)
        scale = max(palm_width, palm_len)

        index_tip  = np.array([lms.landmark[8].x,  lms.landmark[8].y])
        middle_tip = np.array([lms.landmark[12].x, lms.landmark[12].y])
        tip_gap = float(np.linalg.norm(index_tip - middle_tip)) / scale

        tips = [8,12,16,20]
        tip_dists = [float(np.linalg.norm(
            np.array([lms.landmark[i].x, lms.landmark[i].y]) - wrist)) for i in tips]
        avg_tip_away = np.mean(tip_dists) / scale

        is_index_ext  = (states['index']  is True)
        is_middle_ext = (states['middle'] is True)
        is_ring_flex  = (states['ring']   is False)
        is_pinky_flex = (states['pinky']  is False)

        four_ext  = all(states[f] is True  for f in ['index','middle','ring','pinky'])
        four_flex = all(states[f] is False for f in ['index','middle','ring','pinky'])

        SCISSOR_GAP_TH  = 0.22
        PAPER_SPREAD_TH = 0.55

        if is_index_ext and is_middle_ext and is_ring_flex and is_pinky_flex and tip_gap > SCISSOR_GAP_TH:
            return "scissors", self.rps_labels["scissors"]

        if four_ext and avg_tip_away > PAPER_SPREAD_TH:
            return "paper", self.rps_labels["paper"]

        if four_flex and (states['thumb'] in [False, None]):
            return "rock", self.rps_labels["rock"]

        return None, None

    # -------------------- 图像回调 --------------------
    def image_callback(self, cv_image):
        try:
            rgb_for_infer = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_for_infer)

            drew_anything = False
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                value_en, msg_cn = self.recognize_rps(hand_landmarks)

                if value_en is not None:
                    if not self.pub_image_close:
                        self.mp_drawing.draw_landmarks(
                            cv_image,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                            self.mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                        )
                        h, w = cv_image.shape[:2]
                        xs = [lm.x * w for lm in hand_landmarks.landmark]
                        ys = [lm.y * h for lm in hand_landmarks.landmark]
                        x_min, x_max = int(max(min(xs), 0)), int(min(max(xs), w - 1))
                        y_min, y_max = int(max(min(ys), 0)), int(min(max(ys), h - 1))
                        cv2.rectangle(cv_image, (x_min - 10, y_min - 10),
                                      (x_max + 10, y_max + 10), (0, 255, 0), 2)
                        cv_image = self.draw_chinese_text(
                            cv_image,
                            msg_cn,
                            (x_min - 10, max(y_min - 40, 0)),
                            color=(0, 255, 0),
                        )
                        self.pub_image = cv_image

                    # 发布消息
                    self.pub_message = RecognizerMessage(value=value_en, message=msg_cn)

                    # 日志
                    self.get_logger().info(f"{value_en}:{msg_cn}")

                    drew_anything = True

            if not drew_anything:
                if not self.pub_image_close:
                    self.pub_image = cv_image

        except Exception as e:
            self.get_logger().error(f"图像处理错误: {e}")
  
def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = GestureRecognizerRps()
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

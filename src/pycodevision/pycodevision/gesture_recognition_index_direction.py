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
功能：食指方向识别（四方向）
描述：当食指被判为“伸直”时，输出指向方向标签：上/下/左/右。
角度约定：0° 向右，逆时针为正（↑≈90°，←≈180°，↓≈270°）。
作者：pycodeworld
"""

import rclpy
import cv2
import mediapipe as mp
import numpy as np
import math
from .abs_base_recognition import AbsBaseRecognizer, RecognizerMessage


class GestureRecognizerIndexDirection(AbsBaseRecognizer):
    def __init__(self, name: str = 'gesture_recognition_index_direction'):
        super().__init__(name)

        # ---------------- 参数（直接变量赋值） ----------------
        self.mirror = False                     # 是否镜像（前置相机常用）
        self.index_vec_scale_thresh = 0.18      # 食指方向向量长度阈值（相对手掌尺度）
        self.draw_landmarks = True              # 是否绘制骨架
        self.max_num_hands = 1
        self.det_conf = 0.7                     # min_detection_confidence
        self.trk_conf = 0.5                     # min_tracking_confidence

        # ---------------- MediaPipe Hands ----------------
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.det_conf,
            min_tracking_confidence=self.trk_conf
        )

        # 发布缓存
        self.pub_image = None
        # 初始化为 None：未识别到手势时不发布消息
        self.pub_message = None

        self.get_logger().info(
            f"方向识别节点已启动：食指四方向(上/下/左/右) | mirror={self.mirror} | "
            f"th={self.index_vec_scale_thresh:.2f} | draw={self.draw_landmarks}"
        )

    # -------------------- 几何与工具函数 --------------------
    @staticmethod
    def _angle_deg(a, b, c):
        """返回 ∠ABC 的角度（度）"""
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        nba = ba / (np.linalg.norm(ba) + 1e-6)
        nbc = bc / (np.linalg.norm(bc) + 1e-6)
        cosang = float(np.clip(np.dot(nba, nbc), -1.0, 1.0))
        return math.degrees(math.acos(cosang))

    def _angle_to_compass4(self, deg: float) -> str:
        dirs = ['右', '上', '左', '下']
        idx = int(((deg % 360) + 45.0) // 90.0) % 4
        return dirs[idx]

    def _palm_scale(self, lms) -> float:
        wrist = np.array([lms.landmark[0].x, lms.landmark[0].y])
        idx_mcp   = np.array([lms.landmark[5].x,  lms.landmark[5].y])
        mid_mcp   = np.array([lms.landmark[9].x,  lms.landmark[9].y])
        pinky_mcp = np.array([lms.landmark[17].x, lms.landmark[17].y])
        palm_width = float(np.linalg.norm(idx_mcp - pinky_mcp) + 1e-6)
        palm_len   = float(np.linalg.norm(mid_mcp - wrist) + 1e-6)
        return max(palm_width, palm_len)

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

    def index_direction(self, lms, states):
        if states.get('index') is not True:
            return None

        tip_i, base_i = 8, 6
        tip  = np.array([lms.landmark[tip_i].x,  lms.landmark[tip_i].y])
        base = np.array([lms.landmark[base_i].x, lms.landmark[base_i].y])

        scale = self._palm_scale(lms)
        if float(np.linalg.norm(tip - base)) < self.index_vec_scale_thresh * scale:
            return None

        dx = float(tip[0] - base[0])
        dy = float(base[1] - tip[1])

        if self.mirror:
            dx = -dx

        deg = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
        label_zh = self._angle_to_compass4(deg)
        return 'index', label_zh, deg, (base, tip)

    def image_callback(self, cv_image):
        try:
            rgb_for_infer = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_for_infer)

            drew_anything = False
            recognized = False  # 本帧是否识别到手势

            if results.multi_hand_landmarks:
                lms = results.multi_hand_landmarks[0]
                states = self._finger_states_by_angles(lms)
                dir_res = self.index_direction(lms, states)

                if dir_res is not None:
                    _, label_zh, deg, (base_norm, tip_norm) = dir_res
                    en_map = {'上': 'UP', '下': 'DOWN', '左': 'LEFT', '右': 'RIGHT'}
                    gesture_id = en_map.get(label_zh, 'UNKNOWN')
                    gesture_text = label_zh

                    if not self.pub_image_close:
                        if self.draw_landmarks:
                            self.mp_drawing.draw_landmarks(
                                cv_image,
                                lms,
                                self.mp_hands.HAND_CONNECTIONS,
                                self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                self.mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                            )
                        h, w = cv_image.shape[:2]
                        bx, by = int(base_norm[0] * w), int(base_norm[1] * h)
                        tx, ty = int(tip_norm[0] * w),  int(tip_norm[1] * h)
                        cv2.arrowedLine(cv_image, (bx, by), (tx, ty), (255, 255, 0), 3, tipLength=0.25)

                        text = f"食指方向：{label_zh}  ({deg:.0f}°)"
                        cv_image = self.draw_chinese_text(
                            cv_image, text, (bx + 8, max(by - 30, 0)), color=(255, 255, 0)
                        )
                        self.pub_image = cv_image

                    # 仅在识别到手势时发布消息
                    self.pub_message = RecognizerMessage(gesture_id, gesture_text)
                    self.get_logger().info(f"{gesture_id}:{gesture_text}")
                    drew_anything = True
                    recognized = True

            # 未识别到手势：不发布消息（清空消息）
            if not recognized:
                self.pub_message = None

            # 影像发布逻辑保持不变
            if not drew_anything:
                if not self.pub_image_close:
                    self.pub_image = cv_image

        except Exception as e:
            self.get_logger().error(f"图像处理错误: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = GestureRecognizerIndexDirection()
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

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
功能：中文数字手势识别
描述：
- 1~5：按伸直手指数量计数（True=伸直，False=弯曲）。
- 特殊手势（优先判定）：
  * 6：仅拇指+小指伸直（shaka / hang-loose）
  * 7：拇指与食指/中指“捏合”（两距离都较小），环/小指弯曲
  * 8：拇指+食指伸直（手枪），其他弯曲
  * 9：食指“钩形”（食指在 PIP 或 DIP 明显弯曲），其余弯曲（拇指可略弯）
  * 10：拳头（全部弯曲）
- 识别顺序：10/6/7/8/9 → 不匹配再按数量计 1~5。
作者：pycodeworld
"""

import math
import cv2
import mediapipe as mp
import numpy as np

import rclpy
from .abs_base_recognition import AbsBaseRecognizer, RecognizerMessage


class GestureRecognizerDigits(AbsBaseRecognizer):
    def __init__(self, name: str = 'gesture_recognition_digits'):
        super().__init__(name)

        # 初始化 MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        # 特殊手势开关
        self.enable_special_6 = True
        self.enable_special_7 = True
        self.enable_special_8 = True
        self.enable_special_9 = True
        self.enable_special_10 = True

        # —— 单指置信度阈值（用于 1~5 逻辑，也用于 8 的判定）——
        self.EXT_CONF_TH = 0.60
        self.FLEX_CONF_TH = 0.60

        self.pub_image = None
        self.pub_message = None  # 将在识别到时设置为 RecognizerMessage

        self.get_logger().info("节点已启动：中文数字手势识别（含 6/7/8/9/10，8=仅拇指+食指竖起）")

    # -------------------- 几何/工具函数 --------------------
    @staticmethod
    def _clamp01(x: float) -> float:
        return max(0.0, min(1.0, float(x)))

    @staticmethod
    def _angle_deg(a, b, c):
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        nba = ba / (np.linalg.norm(ba) + 1e-6)
        nbc = bc / (np.linalg.norm(bc) + 1e-6)
        cosang = float(np.clip(np.dot(nba, nbc), -1.0, 1.0))
        return math.degrees(math.acos(cosang))

    @staticmethod
    def _xy(lms, idx):
        return (lms.landmark[idx].x, lms.landmark[idx].y)

    @staticmethod
    def _bbox_norm_size(lms):
        xs = [p.x for p in lms.landmark]
        ys = [p.y for p in lms.landmark]
        # 归一化尺度，避免与分辨率相关
        return max(max(xs) - min(xs), max(ys) - min(ys)) + 1e-6

    @staticmethod
    def _dist_norm(a, b, scale):
        return math.hypot(a[0] - b[0], a[1] - b[1]) / max(scale, 1e-6)

    @staticmethod
    def _is_true(x):
        return x is True

    @staticmethod
    def _center_of_points(pts):
        pts = np.asarray(pts, dtype=np.float32)
        return (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))

    def _finger_states_by_angles(self, lms):
        """
        返回各手指是否“伸直(True)/弯曲(False)/不确定(None)”
        （供部分特殊手势分支使用）
        """
        EXT_TH = 160.0
        FLEX_TH = 120.0

        def xy(i): return self._xy(lms, i)

        states = {}

        def one_finger(mcp, pip, tip):
            ang = self._angle_deg(xy(mcp), xy(pip), xy(tip))
            if ang >= EXT_TH:
                return True
            elif ang <= FLEX_TH:
                return False
            else:
                return None

        # 四指：食/中/无名/小
        states['index']  = one_finger(5,  6,  8)
        states['middle'] = one_finger(9, 10, 12)
        states['ring']   = one_finger(13, 14, 16)
        states['pinky']  = one_finger(17, 18, 20)

        # 拇指：MCP 与 IP 结合判断（放宽）
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

    # -------- “角度+距离联合置信度”指状态与置信度（用于 1~5 与 8 判定） ----------
    def _finger_state_and_conf(self, lms):
        """
        计算每根手指的扩展/弯曲状态以及对应置信度（ext_conf / flex_conf）
        返回: (states, ext_conf, flex_conf)
        """
        def xy(i): return self._xy(lms, i)

        fingers = {
            'index':  (5, 6, 7, 8),
            'middle': (9, 10, 11, 12),
            'ring':   (13, 14, 15, 16),
            'pinky':  (17, 18, 19, 20),
        }
        palm_pts = [xy(0), xy(1), xy(5), xy(9), xy(13), xy(17)]
        palm_center = self._center_of_points(palm_pts)
        scale = self._bbox_norm_size(lms)

        states = {}
        ext_conf = {}
        flex_conf = {}

        for name, (mcp, pip, dip, tip) in fingers.items():
            pip_angle = self._angle_deg(xy(mcp), xy(pip), xy(dip))
            dip_angle = self._angle_deg(xy(pip), xy(dip), xy(tip))
            tip_far = self._dist_norm(xy(tip), palm_center, scale)
            pip_far = self._dist_norm(xy(pip), palm_center, scale)

            ext_a = 0.6 * self._clamp01((pip_angle - 140.0) / 40.0) + 0.4 * self._clamp01((dip_angle - 135.0) / 35.0)
            flex_a = 0.6 * self._clamp01((140.0 - pip_angle) / 45.0) + 0.4 * self._clamp01((135.0 - dip_angle) / 40.0)
            ext_d = self._clamp01((tip_far - pip_far) / 0.10)

            ext_c = self._clamp01(0.8 * ext_a + 0.2 * ext_d)
            flex_c = self._clamp01(flex_a)

            ext_conf[name] = ext_c
            flex_conf[name] = flex_c

            if ext_c >= self.EXT_CONF_TH and flex_c <= 0.5:
                states[name] = True
            elif flex_c >= self.FLEX_CONF_TH and ext_c <= 0.5:
                states[name] = False
            else:
                states[name] = None

        # 拇指
        ang_thumb_mcp = self._angle_deg(xy(1), xy(2), xy(3))
        ang_thumb_ip  = self._angle_deg(xy(2), xy(3), xy(4))
        tip_far_thumb = self._dist_norm(xy(4), palm_center, scale)
        mcp_far_thumb = self._dist_norm(xy(2), palm_center, scale)

        ext_a_thumb = 0.6 * self._clamp01((ang_thumb_mcp - 150.0) / 35.0) + 0.4 * self._clamp01((ang_thumb_ip - 145.0) / 35.0)
        flex_a_thumb = 0.6 * self._clamp01((150.0 - ang_thumb_mcp) / 40.0) + 0.4 * self._clamp01((145.0 - ang_thumb_ip) / 40.0)
        ext_d_thumb = self._clamp01((tip_far_thumb - mcp_far_thumb) / 0.08)
        ext_c_thumb = self._clamp01(0.8 * ext_a_thumb + 0.2 * ext_d_thumb)
        flex_c_thumb = self._clamp01(flex_a_thumb)

        ext_conf['thumb'] = ext_c_thumb
        flex_conf['thumb'] = flex_c_thumb

        if ext_c_thumb >= self.EXT_CONF_TH and flex_c_thumb <= 0.5:
            states['thumb'] = True
        elif flex_c_thumb >= self.FLEX_CONF_TH and ext_c_thumb <= 0.5:
            states['thumb'] = False
        else:
            states['thumb'] = None

        return states, ext_conf, flex_conf

    # -------------------- 特殊手势判定 --------------------
    def _is_fist10(self, lms, s):
        """10：拳头（全部弯曲）"""
        return all(self._is_true(v) is False for v in s.values())

    def _is_shaka6(self, lms, s):
        """6：拇指 + 小指"""
        return self._is_true(s['thumb']) and (not self._is_true(s['index'])) and \
               (not self._is_true(s['middle'])) and (not self._is_true(s['ring'])) and \
               self._is_true(s['pinky'])

    def _is_gun8(self, lms, _s_ignored):
        """
        8：拇指 + 食指（其余不竖起）
        条件：使用 _finger_state_and_conf 的“强扩展”定义，
        仅当扩展集合 == {'thumb','index'} 时判定为 8。
        """
        states, ext_c, _ = self._finger_state_and_conf(lms)
        fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
        extended = {f for f in fingers if (states[f] is True and ext_c[f] >= self.EXT_CONF_TH)}
        return extended == {'thumb', 'index'}

    def _is_pinch7(self, lms, s):
        """
        7：拇指+食指+中指“捏合”
        - 环/小指弯曲
        - 拇指尖与食/中指尖距离都较小（按手掌尺度归一化）
        """
        if not (self._is_true(s['ring']) is False and self._is_true(s['pinky']) is False):
            return False

        scale = self._bbox_norm_size(lms)
        tip_thumb = self._xy(lms, 4)
        tip_index = self._xy(lms, 8)
        tip_middle = self._xy(lms, 12)

        d_t_i = self._dist_norm(tip_thumb, tip_index, scale)
        d_t_m = self._dist_norm(tip_thumb, tip_middle, scale)

        # 可按需求微调这两个阈值
        return (d_t_i < 0.22 and d_t_m < 0.26)

    def _is_hook9(self, lms, s):
        """
        9：食指“钩形”（放宽判定）
        - 食指在 PIP 或 DIP 弯曲到“比较平”即可：
          * PIP 角度（MCP-PIP-TIP）≤ 120°
          * 或 DIP 角度（PIP-DIP-TIP）≤ 105°
        - 其它手指弯曲（拇指可略弯）
        """
        def xy(i): return self._xy(lms, i)

        # 其它手指需弯曲（沿用原逻辑）
        others_bent = (not self._is_true(s['middle'])) and \
                      (not self._is_true(s['ring'])) and \
                      (not self._is_true(s['pinky']))

        ang_pip = self._angle_deg(xy(5), xy(6), xy(8))   # MCP-PIP-TIP
        ang_dip = self._angle_deg(xy(6), xy(7), xy(8))   # PIP-DIP-TIP

        # 放宽阈值：原为 PIP≤95 或 DIP≤85
        PIP_TH = 125.0
        DIP_TH = 110.0
        index_hook = (ang_pip <= PIP_TH) or (ang_dip <= DIP_TH)

        return others_bent and index_hook

    # -------------------- 数字手势统一判定 --------------------
    def recognize_digit(self, hand_landmarks):
        """
        返回 (digit_id, None)；若不确定 → (None, None)
        优先判定：10/6/7/8/9 → 否则按“扩展置信+他指不强直”计作 1~5
        """
        lms = hand_landmarks

        # 先用“角度粗判”做除 8 之外的特殊手势
        s_angles = self._finger_states_by_angles(lms)

        # 10：拳头
        if self.enable_special_10 and self._is_fist10(lms, s_angles):
            return 10, None

        # 6：shaka
        if self.enable_special_6 and self._is_shaka6(lms, s_angles):
            return 6, None

        # 7：捏合
        if self.enable_special_7 and self._is_pinch7(lms, s_angles):
            return 7, None

        # 8：手枪（仅拇指+食指竖起）
        if self.enable_special_8 and self._is_gun8(lms, s_angles):
            return 8, None

        # 9：食指钩形（放宽）
        if self.enable_special_9 and self._is_hook9(lms, s_angles):
            return 9, None

        # ---- 常规：按“1~5 判别逻辑（角度+距离联合置信度）”计作 1~5 ----
        states, ext_c, _flex_c = self._finger_state_and_conf(lms)
        fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']

        # 仅统计“明确扩展且扩展置信度≥阈值”的手指
        extended = [f for f in fingers if (states[f] is True and ext_c[f] >= self.EXT_CONF_TH)]
        # 其它手指需“非扩展或不确定”（不允许明显强直）
        bent_ok = all((states[f] is False) or (states[f] is None) for f in fingers if f not in extended)

        count = len(extended)
        if 1 <= count <= 5 and bent_ok:
            # 置信度合成（用于是否通过的门槛，不改变输出签名）
            if extended:
                ext_min = min(ext_c[f] for f in extended)
            else:
                ext_min = 0.0
            others_max_ext = max([ext_c[f] for f in fingers if f not in extended] + [0.0])
            conf = self._clamp01(ext_min * (1.0 - others_max_ext))
            if conf >= 0.45:
                return count, None

        return None, None

    # -------------------- 图像回调 --------------------
    def image_callback(self, cv_image):
        """
        - 明确识别到：画关键点/框，并在图像左上角叠加数字；发布 RecognizerMessage(value=数字)。
        - 不确定或未检测到：透传原图，不改动现有 pub_message。
        """
        try:
            rgb_for_infer = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_for_infer)

            drew_anything = False
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                d_id, _ = self.recognize_digit(hand_landmarks)

                if d_id is not None:
                    # 仅绘制关键点与外接框，并叠加数字标签
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
                        # 外接框
                        cv2.rectangle(cv_image, (x_min - 10, y_min - 10),
                                      (x_max + 10, y_max + 10), (0, 255, 0), 2)

                        # 左上角数字标签
                        label = f"{d_id}"
                        font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3
                        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                        x1 = max(x_min - 10, 0)
                        y1 = max(y_min - 10 - text_h - 8, 0)
                        cv2.rectangle(cv_image, (x1, y1), (x1 + text_w + 8, y1 + text_h + 8), (0, 255, 0), -1)
                        cv2.putText(cv_image, label, (x1 + 4, y1 + text_h + 2),
                                    font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

                        self.pub_image = cv_image

                    # 发布识别结果
                    self.pub_message = RecognizerMessage(value=d_id, message=None)

                    # 日志输出为 value:message 的形式（message 为空）
                    self.get_logger().info(f"{d_id}:")
                    drew_anything = True

            # 未识别或未检测：透传原图
            if not drew_anything:
                if not self.pub_image_close:
                    self.pub_image = cv_image
                # 不修改 self.pub_message（保持原值）

        except Exception as e:
            self.get_logger().error(f"图像处理错误: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = GestureRecognizerDigits()
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

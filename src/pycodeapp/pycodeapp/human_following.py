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
功能：人体跟随
描述：基于姿态识别的人体跟随，通过订阅姿态识别的相关主题可以控制OriginBot智能机器人实时跟随。
作者：pycodeworld
'''

import signal
import math
import time
import json
from typing import Optional, Tuple, Dict, Any, List

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import String
from geometry_msgs.msg import Twist


def _pick_target(items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """选择一个目标（当前简单地取第一个；可按需替换为最大框等策略）"""
    if not items:
        return None
    return items[0]


class HumanFollowing(Node):
    def __init__(self):
        super().__init__('human_following')

        # ---------------- 参数（去除图像/调试图相关）----------------
        # 话题名
        self.declare_parameter('input_topic', 'posture_recognition/message')  # 姿态识别节点发布的消息话题
        self.declare_parameter('cmd_topic', 'cmd_vel')

        # 控制目标：希望目标框高度占比（越大表示越近）
        self.declare_parameter('target_box_ratio', 0.7)     # 目标接近距离
        self.declare_parameter('too_close_ratio', 0.95)     # 过近需后退
        self.declare_parameter('too_far_ratio', 0.45)       # 太远需要前进

        # 控制增益与限幅
        self.declare_parameter('kv', 1.0)                   # 距离 -> 线速度 增益
        self.declare_parameter('kw', 2.0)                   # 水平偏差 -> 角速度 增益
        self.declare_parameter('v_max_forward', 0.45)       # 最大前进速度 m/s
        self.declare_parameter('v_max_backward', 0.30)      # 最大后退速度 m/s
        self.declare_parameter('w_max', 1.8)                # 最大角速度 rad/s

        # 死区与滤波
        self.declare_parameter('center_deadband', 0.03)     # 居中死区（相对宽度）
        self.declare_parameter('dist_deadband', 0.02)       # 距离死区（box 高度占比）
        self.declare_parameter('ema_alpha', 0.35)           # 指标指数平滑系数 0~1

        # 安全停与实时性
        self.declare_parameter('lost_timeout_sec', 0.5)     # 丢失人/非站立超时即停
        self.declare_parameter('cmd_rate_hz', 20.0)         # 指令发送频率

        # 读取参数
        recog_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        cmd_topic = self.get_parameter('cmd_topic').get_parameter_value().string_value

        self.target_box_ratio = float(self.get_parameter('target_box_ratio').value)
        self.too_close_ratio = float(self.get_parameter('too_close_ratio').value)
        self.too_far_ratio = float(self.get_parameter('too_far_ratio').value)

        self.kv = float(self.get_parameter('kv').value)
        self.kw = float(self.get_parameter('kw').value)
        self.v_max_f = float(self.get_parameter('v_max_forward').value)
        self.v_max_b = float(self.get_parameter('v_max_backward').value)
        self.w_max = float(self.get_parameter('w_max').value)

        self.center_deadband = float(self.get_parameter('center_deadband').value)
        self.dist_deadband = float(self.get_parameter('dist_deadband').value)
        self.alpha = float(self.get_parameter('ema_alpha').value)

        self.lost_timeout = float(self.get_parameter('lost_timeout_sec').value)

        # 订阅：只订阅识别节点消息
        qos_msg = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )
        self.sub_recog = self.create_subscription(
            String, recog_topic, self.recognizer_cb, qos_msg
        )

        # 发布控制
        self.pub_cmd = self.create_publisher(Twist, cmd_topic, 10)

        # 定时器：守护丢失/定频下发
        period = 1.0 / float(self.get_parameter('cmd_rate_hz').value)
        self.timer = self.create_timer(period, self.control_tick)

        # 状态
        self.last_msg_time = 0.0        # 最近一次收到识别消息的时间
        self.last_seen_time = 0.0       # 最近一次“站立”时间（只有站立才跟随）
        self.latest_cmd: Twist = Twist()

        # 平滑指标
        self._ema_center_x = None       # 相对中心 [-0.5, 0.5]
        self._ema_box_ratio = None      # 框高占比 [0, 1]

        # 退出时安全急停
        signal.signal(signal.SIGINT, self._sigint_handler)
        self.get_logger().info("HumanFollowing(消息驱动版) 已启动：仅订阅姿态识别消息，识别为“站立”即跟随。")

    # ---------------- 识别消息回调 ----------------
    def recognizer_cb(self, msg: String):
        self.last_msg_time = time.time()

        # 解析 JSON（兼容两种结构：纯数组 或 {"value":..,"message":[...]}）
        try:
            payload = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"识别消息 JSON 解析失败: {e}")
            self._safe_stop_once("bad_json")
            return

        if isinstance(payload, dict) and 'message' in payload:
            items = payload.get('message', [])
        elif isinstance(payload, list):
            items = payload
        else:
            self.get_logger().warn("识别消息格式不含 'message' 且不是数组，忽略。")
            self._safe_stop_once("bad_format")
            return

        target = _pick_target(items)
        if target is None:
            # 无目标：急停
            self._ema_center_x = None
            self._ema_box_ratio = None
            self._safe_stop_once("no_target")
            return

        # 读取字段（来自姿态识别的边界框与图像尺寸）
        try:
            name = str(target.get("name", "unknown"))
            x = int(target["left"])
            y = int(target["top"])
            w = int(target["width"])
            h = int(target["height"])
            img_w = int(target["image_width"])
            img_h = int(target["image_height"])
        except Exception as e:
            self.get_logger().warn(f"识别消息字段缺失或类型异常: {e}")
            self._safe_stop_once("bad_fields")
            return

        # 仅“站立”时跟随；其他姿态立即停
        is_standing = (name == "standing")
        if is_standing:
            self.last_seen_time = time.time()
        else:
            self._safe_stop_once(f"pose={name}")
            return

        # 计算控制指标
        cx = x + w / 2.0
        center_err = (cx - img_w / 2.0) / float(img_w)           # [-0.5, 0.5]
        box_ratio = (h / float(img_h)) if img_h > 0 else 0.0     # [0, 1]

        # EMA 平滑
        self._ema_center_x = center_err if self._ema_center_x is None \
            else (self.alpha * center_err + (1 - self.alpha) * self._ema_center_x)
        self._ema_box_ratio = box_ratio if self._ema_box_ratio is None \
            else (self.alpha * box_ratio + (1 - self.alpha) * self._ema_box_ratio)

        # 距离控制（越近 box_ratio 越大）
        dist_err = (self.target_box_ratio - self._ema_box_ratio)
        v = self.kv * dist_err
        if abs(dist_err) < self.dist_deadband:
            v = 0.0
        v = max(-self.v_max_b, min(self.v_max_f, v))

        # 角速度：让人居中
        w_cmd = self.kw * self._ema_center_x
        if abs(self._ema_center_x) < self.center_deadband:
            w_cmd = 0.0
        w_cmd = max(-self.w_max, min(self.w_max, w_cmd))

        # 保险：过近强制后退
        if self._ema_box_ratio > self.too_close_ratio:
            v = -min(self.v_max_b, self.kv * (self._ema_box_ratio - self.target_box_ratio))

        # 保险：太远强制前进
        if self._ema_box_ratio < self.too_far_ratio:
            v = min(self.v_max_f, self.kv * (self.target_box_ratio - self._ema_box_ratio))

        # 发布控制
        cmd = Twist()
        cmd.linear.x = float(v)
        # 图像坐标向右为正 -> 机器人左转为正，取负
        cmd.angular.z = float(-w_cmd)
        self.pub_cmd.publish(cmd)
        self.latest_cmd = cmd

    # ---------------- 定时器：丢失守护/心跳下发 ----------------
    def control_tick(self):
        now = time.time()
        # 超时（既没新消息，也没“站立”） -> 急停
        if now - max(self.last_msg_time, self.last_seen_time) > self.lost_timeout:
            self._safe_stop_once("timeout")
        # 定频把 last cmd 再下发（部分底盘会超时停）
        self.pub_cmd.publish(self.latest_cmd)

    # ---------------- 安全急停 ----------------
    def _safe_stop_once(self, reason: str = ""):
        stop = Twist()  # 全零
        self.pub_cmd.publish(stop)
        self.latest_cmd = stop
        if reason:
            self.get_logger().debug(f"STOP ({reason})")

    def _sigint_handler(self, sig, frame):
        self.get_logger().info("SIGINT 收到，急停并退出。")
        try:
            for _ in range(5):  # 多发几次确保底盘收到
                self.pub_cmd.publish(Twist())
                time.sleep(0.02)
        finally:
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = HumanFollowing()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("节点被用户终止")
    except Exception as e:
        node.get_logger().error(f"异常：{e}")
    finally:
        # 退出前必须停止底盘
        try:
            for _ in range(8):
                node.pub_cmd.publish(Twist())
                time.sleep(0.02)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

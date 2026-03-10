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
功能：手势控制
描述：基于食指方向识别的手势控制，通过订阅手势识别——食指方向识别的相关主题可以控制OriginBot智能机器人按手势运动：
  - 上：前进
  - 下：后退
  - 左：原地左转
  - 右：原地右转
  - 未识别：停止
作者：pycodeworld
"""

import json
import signal
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import String
from geometry_msgs.msg import Twist


class IndexDirectionController(Node):
    def __init__(self):
        super().__init__('index_direction_controller')

        # ---------------- 参数 ----------------
        # 话题：只订阅“食指方向识别”节点消息
        self.declare_parameter('input_topic', 'gesture_recognition_index_direction/message')
        # 底盘控制话题
        self.declare_parameter('cmd_topic', 'cmd_vel')

        # 速度参数（可根据底盘调节）
        self.declare_parameter('v_forward', 0.05)    # m/s
        self.declare_parameter('v_backward', 0.05)   # m/s（正数，内部取负）
        self.declare_parameter('w_turn', 0.80)       # rad/s（左转为正、右转为负）

        # 发布频率与超时
        self.declare_parameter('cmd_rate_hz', 20.0)      # 指令发布心跳频率
        self.declare_parameter('lost_timeout_sec', 0.35) # 超时视为“未识别到” -> 停止

        # 读取参数
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        cmd_topic = self.get_parameter('cmd_topic').get_parameter_value().string_value
        self.v_forward = float(self.get_parameter('v_forward').value)
        self.v_backward = float(self.get_parameter('v_backward').value)
        self.w_turn = float(self.get_parameter('w_turn').value)
        self.cmd_rate_hz = float(self.get_parameter('cmd_rate_hz').value)
        self.lost_timeout = float(self.get_parameter('lost_timeout_sec').value)

        # QoS
        qos_msg = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )

        # 订阅：仅订阅识别节点消息
        self.sub_recog = self.create_subscription(String, input_topic, self._gesture_cb, qos_msg)

        # 发布：底盘控制
        self.pub_cmd = self.create_publisher(Twist, cmd_topic, 10)

        # 定时器：心跳与超时急停
        period = 1.0 / max(1e-6, self.cmd_rate_hz)
        self.timer = self.create_timer(period, self._tick)

        # 状态
        self._last_msg_time: float = 0.0
        self._latest_cmd: Twist = Twist()
        self._stopped: bool = True

        # 退出信号：安全急停
        signal.signal(signal.SIGINT, self._sigint_handler)

        self.get_logger().info(
            f"IndexDirectionController started | sub='{input_topic}' -> pub='{cmd_topic}' | "
            f"v_fwd={self.v_forward:.2f} v_back={self.v_backward:.2f} w_turn={self.w_turn:.2f} | "
            f"timeout={self.lost_timeout:.2f}s rate={self.cmd_rate_hz:.1f}Hz"
        )

    # ---------------- 工具：从 JSON(value) 提取手势 ----------------
    @staticmethod
    def _normalize_value(val: str) -> Optional[str]:
        """
        将 value 规范化为 'UP'/'DOWN'/'LEFT'/'RIGHT' 之一；无法识别返回 None。
        支持中英文：上/下/左/右。
        """
        if not isinstance(val, str):
            return None
        s = val.strip()
        if not s:
            return None

        # 中文直接映射
        zh_map = {"上": "UP", "下": "DOWN", "左": "LEFT", "右": "RIGHT"}
        if s in zh_map:
            return zh_map[s]

        # 英文大小写不敏感
        s_up = s.upper()
        if s_up in {"UP", "DOWN", "LEFT", "RIGHT"}:
            return s_up
        return None

    def _parse_value_from_msg(self, data: str) -> Optional[str]:
        """
        只接受 JSON 且结构为 {"value": "..."}，否则返回 None。
        """
        try:
            obj = json.loads(data.strip())
        except Exception:
            return None

        if not isinstance(obj, dict) or "value" not in obj:
            return None
        return self._normalize_value(obj["value"])

    # ---------------- 订阅回调 ----------------
    def _gesture_cb(self, msg: String):
        now = time.time()
        g = self._parse_value_from_msg(msg.data)
        self._last_msg_time = now

        if g is None:
            # 收到该识别节点但不是期望结构或值 -> 按“未识别到”处理：停
            self._apply_stop_once("bad_or_missing_value")
            return

        # 根据手势映射控制
        cmd = Twist()
        if g == "UP":
            cmd.linear.x = self.v_forward
            cmd.angular.z = 0.0
            self._apply_cmd(cmd, reason='UP->forward')
        elif g == "DOWN":
            cmd.linear.x = -self.v_backward
            cmd.angular.z = 0.0
            self._apply_cmd(cmd, reason='DOWN->backward')
        elif g == "LEFT":
            # 先停再左转
            self._apply_stop_once("LEFT->stop_then_turn_left")
            time.sleep(0.02)  # 确保底盘接收到停指令（按需调整/可去掉）
            cmd.linear.x = 0.0
            cmd.angular.z = +self.w_turn
            self._apply_cmd(cmd, reason='LEFT->turn_left')
        elif g == "RIGHT":
            self._apply_stop_once("RIGHT->stop_then_turn_right")
            time.sleep(0.02)
            cmd.linear.x = 0.0
            cmd.angular.z = -self.w_turn
            self._apply_cmd(cmd, reason='RIGHT->turn_right')
        else:
            # 理论不会到此
            self._apply_stop_once("unknown_value")

    # ---------------- 定时器：心跳与超时急停 ----------------
    def _tick(self):
        now = time.time()
        # 超时 -> 视为未识别 -> 停止
        if now - self._last_msg_time > self.lost_timeout:
            self._apply_stop_once("timeout_no_message")
        # 定频心跳：重发最近一次指令
        self.pub_cmd.publish(self._latest_cmd)

    # ---------------- 发布封装 ----------------
    def _apply_cmd(self, cmd: Twist, reason: str = ""):
        self.pub_cmd.publish(cmd)
        self._latest_cmd = cmd
        if reason:
            self.get_logger().info(f"CMD: {reason} | v={cmd.linear.x:.2f} w={cmd.angular.z:.2f}")
        self._stopped = (abs(cmd.linear.x) < 1e-6 and abs(cmd.angular.z) < 1e-6)

    def _apply_stop_once(self, reason: str = ""):
        if self._stopped:
            return
        stop = Twist()
        self.pub_cmd.publish(stop)
        self._latest_cmd = stop
        self._stopped = True
        if reason:
            self.get_logger().info(f"CMD: STOP ({reason})")

    # ---------------- 信号处理 ----------------
    def _sigint_handler(self, sig, frame):
        self.get_logger().info("SIGINT 收到，急停并退出。")
        try:
            for _ in range(5):
                self.pub_cmd.publish(Twist())
                time.sleep(0.02)
        finally:
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = IndexDirectionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("节点被用户终止")
    except Exception as e:
        node.get_logger().error(f"异常：{e}")
    finally:
        # 退出前务必停止底盘
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

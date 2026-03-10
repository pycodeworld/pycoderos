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
功能：相对位姿记录
描述：
  - 订阅里程计 odom（nav_msgs/msg/Odometry）。
  - 以第一帧有效姿态为“起点”，输出并发布：
      * 位移分量：dx、dy（m，odom 坐标：x 前、y 左、z 上）
      * 位移模长：distance = sqrt(dx^2 + dy^2)（m）
      * 相对偏转角：d_yaw = wrap(yaw_now - yaw0)（rad & deg）
  - 以 **单一话题**（std_msgs/String，内容为 JSON）发布合并数据：
      * rel_pose_json：{
          "offset":{"dx","dy","distance","unit"},
          "yaw":{"rad","deg"},
          "stamp":{"sec","nanosec"}
        }
作者：pycodeworld
'''

import signal
import time
import math
import json
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data  # 统一使用传感器 QoS
from nav_msgs.msg import Odometry
from std_msgs.msg import String


def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    """从四元数计算航向角（Yaw，弧度，右手系，绕 Z 轴）。"""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _wrap_angle(a: float) -> float:
    """将角度归一化到 (-pi, pi]。"""
    return math.atan2(math.sin(a), math.cos(a))


def _to_stamp_dict(t: float) -> dict:
    """将 float 秒转为 {sec, nanosec} 字典，便于 JSON 发布。"""
    sec = int(t)
    nanosec = int(round((t - sec) * 1e9))
    if nanosec >= 1_000_000_000:  # 处理四舍五入导致的 1e9 边界
        sec += 1
        nanosec -= 1_000_000_000
    return {"sec": sec, "nanosec": nanosec}


class OdomRelPose(Node):
    """
    记录相对位移与偏转角，并以单一 JSON 字符串发布。
    """

    def __init__(self):
        super().__init__('odom_rel_pose')

        # ---------------- 参数 ----------------
        self.declare_parameter('odom_topic', 'odom')                   # 里程计话题
        # 日志/发布频率（Hz）
        self.declare_parameter('log_rate_hz', 5.0)
        # odom 超时阈值（s）
        self.declare_parameter('odom_timeout_sec', 0.6)
        self.declare_parameter(
            'use_header_time', True)                # 使用消息时间判断超时
        # 数值打印/发布小数位
        self.declare_parameter('precision', 3)
        # 新参数：合并发布的话题名
        self.declare_parameter('rel_pose_topic', 'rel_pose_json')

        odom_topic = self.get_parameter(
            'odom_topic').get_parameter_value().string_value
        self.log_rate_hz = float(self.get_parameter('log_rate_hz').value)
        self.odom_timeout = float(self.get_parameter('odom_timeout_sec').value)
        self.use_header_time = bool(
            self.get_parameter('use_header_time').value)
        self.precision = int(self.get_parameter('precision').value)
        self.rel_pose_topic = self.get_parameter(
            'rel_pose_topic').get_parameter_value().string_value

        # ---------------- 通信 ----------------
        # 订阅：使用传感器 QoS
        self.sub_odom = self.create_subscription(
            Odometry,
            odom_topic,
            self.odom_cb,
            qos_profile_sensor_data
        )

        # 发布：仅一个合并话题，统一使用传感器 QoS
        self.pub_rel = self.create_publisher(
            String, self.rel_pose_topic, qos_profile_sensor_data)

        # 固定频率输出日志 & 发布 JSON
        period = 1.0 / max(1e-6, self.log_rate_hz)
        self.timer = self.create_timer(period, self.log_tick)

        # ---------------- 状态 ----------------
        self.last_odom_stamp: Optional[float] = None  # 最近一次 odom 时间（秒）

        # 起点（用于相对位姿）
        self.x0: Optional[float] = None
        self.y0: Optional[float] = None
        self.yaw0: Optional[float] = None

        # 当前观测
        self.x_now: Optional[float] = None
        self.y_now: Optional[float] = None
        self.yaw_now: Optional[float] = None

        # 缓存计算量
        self.dx: Optional[float] = None
        self.dy: Optional[float] = None
        self.distance: Optional[float] = None
        self.d_yaw: Optional[float] = None  # (-pi, pi]

        # 退出时优雅关闭
        signal.signal(signal.SIGINT, self._sigint_handler)

        self.get_logger().info(
            f"OdomRelPoseLogger 已启动：log_rate={self.log_rate_hz:.2f}Hz, "
            f"precision={self.precision}, odom_timeout={self.odom_timeout:.2f}s, "
            f"发布单话题：{self.rel_pose_topic}，QoS=SensorData(订阅/发布)"
        )

    # ---------------- odom 回调：更新当前位姿与初始参考 ----------------
    def odom_cb(self, msg: Odometry):
        # 时间戳
        if self.use_header_time and (msg.header.stamp.sec != 0 or msg.header.stamp.nanosec != 0):
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        else:
            t = time.time()

        # 位姿
        px = float(msg.pose.pose.position.x)
        py = float(msg.pose.pose.position.y)
        qx = float(msg.pose.pose.orientation.x)
        qy = float(msg.pose.pose.orientation.y)
        qz = float(msg.pose.pose.orientation.z)
        qw = float(msg.pose.pose.orientation.w)

        # 当前航向
        yaw = None
        if not math.isnan(qx + qy + qz + qw):
            yaw = _yaw_from_quat(qx, qy, qz, qw)

        # 更新当前观测
        self.x_now, self.y_now, self.yaw_now = px, py, yaw

        # 初始化起点与初始航向
        if self.x0 is None or self.y0 is None:
            self.x0, self.y0 = px, py
        if self.yaw0 is None and (yaw is not None):
            self.yaw0 = yaw

        # 计算相对量
        if (self.x0 is not None) and (self.y0 is not None):
            self.dx = px - self.x0
            self.dy = py - self.y0
            self.distance = math.hypot(self.dx, self.dy)
        else:
            self.dx = self.dy = self.distance = None

        if (self.yaw0 is not None) and (yaw is not None):
            self.d_yaw = _wrap_angle(yaw - self.yaw0)
        else:
            self.d_yaw = None

        self.last_odom_stamp = t

    # ---------------- 定时日志输出 & JSON 发布 ----------------
    def log_tick(self):
        now = time.time()

        # 超时监测
        if self.last_odom_stamp is None or (now - self.last_odom_stamp) > self.odom_timeout:
            self.get_logger().warn("odom 超时或未收到数据，暂停输出相对位姿与发布。")
            return

        p = self.precision

        # ===== 1) 组装可读日志 =====
        if self.dx is not None and self.dy is not None and self.distance is not None:
            disp_str = f"dx={self.dx:.{p}f} m, dy={self.dy:.{p}f} m, dist={self.distance:.{p}f} m"
        else:
            disp_str = "位移: NaN"

        if self.d_yaw is not None:
            dyaw_rad = self.d_yaw
            dyaw_deg = math.degrees(dyaw_rad)
            yaw_str = f"d_yaw={dyaw_rad:.{p}f} rad ({dyaw_deg:.{p}f} deg)"
        else:
            yaw_str = "d_yaw: NaN"

        self.get_logger().info(f"[RelPose] {disp_str} | {yaw_str}")

        # ===== 2) 发布 合并 JSON（String） =====
        stamp = _to_stamp_dict(self.last_odom_stamp)

        offset = {
            "dx": round(self.dx, p) if self.dx is not None else None,
            "dy": round(self.dy, p) if self.dy is not None else None,
            "distance": round(self.distance, p) if self.distance is not None else None,
            "unit": "m",
        }

        yaw_payload = {
            "rad": round(self.d_yaw, p) if self.d_yaw is not None else None,
            "deg": round(math.degrees(self.d_yaw), p) if self.d_yaw is not None else None,
        }

        combined_payload = {
            "offset": offset,
            "yaw": yaw_payload,
            "stamp": stamp
        }

        msg = String()
        # 使用 ensure_ascii=False，中文可读；separators 压缩空白
        msg.data = json.dumps(
            combined_payload, ensure_ascii=False, separators=(",", ":"))
        self.pub_rel.publish(msg)

    # ---------------- SIGINT 处理：优雅退出 ----------------
    def _sigint_handler(self, sig, frame):
        self.get_logger().info("SIGINT 收到，准备退出。")
        try:
            time.sleep(0.05)  # 给 rclpy 一些时间处理未决回调
        finally:
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = OdomRelPose()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("节点被用户终止")
    except Exception as e:
        node.get_logger().error(f"异常：{e}")
    finally:
        try:
            time.sleep(0.02)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

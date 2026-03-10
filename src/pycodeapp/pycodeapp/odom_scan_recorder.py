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
功能：里程计与激光雷达数据记录
描述：订阅 /odom 与 /scan，将数据合并为一个 JSON 文件并每隔 1s 写入到指定文件夹
作者：pycodeworld
'''

import os
import re
import json
import math
from typing import Any, Dict, List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, qos_profile_sensor_data
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


class OdomScanRecorder(Node):
    """
    订阅 /odom (Odometry) 和 /scan (LaserScan)，每 1s 将最新消息合并写入 JSON 文件：
    文件夹：$PYCODEBOT_HOME/src/data/odom_scan_recorder/
    文件名：1.json, 2.json, ...
    """

    def __init__(self, name: str = "odom_scan_recorder"):
        super().__init__(name)

        # ========= 目录与文件编号 =========
        pycode_home = os.getenv("PYCODEBOT_HOME")
        if not pycode_home:
            # 兜底：若未设置环境变量，则放到用户家目录下（方便调试）
            pycode_home = os.path.expanduser("~")
            self.get_logger().warn(
                "环境变量 PYCODEBOT_HOME 未设置，数据将写到用户目录下！"
            )

        self.save_dir = os.path.join(pycode_home, "src", "data", name)
        os.makedirs(self.save_dir, exist_ok=True)

        self.file_index = self._init_file_index()

        # ========= 订阅者（分别设置 QoS） =========
        self.last_odom: Optional[Odometry] = None
        self.last_scan: Optional[LaserScan] = None

        # /odom：多数发布者使用 Reliable；保持可靠（如发布端为 Best Effort，可改成 qos_profile_sensor_data）
        reliable_qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RELIABLE)
        self.create_subscription(Odometry, "/odom", self._odom_cb, reliable_qos)

        # /scan：传感器数据通常为 Best Effort，使用传感器 QoS，避免 QoS 不兼容
        self.create_subscription(LaserScan, "/scan", self._scan_cb, qos_profile_sensor_data)

        # ========= 定时器（1s） =========
        self.timer_period = 1.0
        self.timer = self.create_timer(self.timer_period, self._on_timer)

        self.get_logger().info(
            f"数据记录器已启动。输出目录：{self.save_dir}；起始文件序号：{self.file_index}"
        )

    # ---------------- 工具：初始化文件序号 ----------------
    def _init_file_index(self) -> int:
        """
        扫描目录下的 *.json，按纯数字文件名取最大 + 1；缺省从 1 开始。
        """
        max_idx = 0
        number_re = re.compile(r"^(\d+)\.json$")
        try:
            for fname in os.listdir(self.save_dir):
                m = number_re.match(fname)
                if m:
                    idx = int(m.group(1))
                    if idx > max_idx:
                        max_idx = idx
        except Exception as e:
            self.get_logger().warn(f"扫描目录初始化编号失败：{e}")
        return max_idx + 1 if max_idx >= 1 else 1

    # ---------------- 回调：/odom ----------------
    def _odom_cb(self, msg: Odometry):
        self.last_odom = msg

    # ---------------- 回调：/scan ----------------
    def _scan_cb(self, msg: LaserScan):
        self.last_scan = msg

    # ---------------- 定时器：1s 写一次 ----------------
    def _on_timer(self):
        # 允许只有一类数据时也写文件；若希望“两者都到齐才写”，则改为：if not (self.last_odom and self.last_scan): return
        odom_json = self._odom_to_json(self.last_odom) if self.last_odom else None
        scan_json = self._scan_to_json(self.last_scan) if self.last_scan else None

        payload = {
            "write_time": self._now_stamp_dict(),  # 写入时刻（节点时钟）
            "odom": odom_json,                      # 可能为 None
            "scan": scan_json,                      # 可能为 None
        }

        # 为确保 JSON 合法，将 NaN/Inf 转为 None
        payload = self._sanitize(payload)

        out_path = os.path.join(self.save_dir, f"{self.file_index}.json")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            self.get_logger().info(f"[写入] {out_path}")
            self.file_index += 1
        except Exception as e:
            self.get_logger().error(f"[失败] 写入 {out_path} 出错：{e}")

    # ---------------- 序列化：时间戳 ----------------
    @staticmethod
    def _stamp_to_dict(stamp) -> Dict[str, int]:
        # stamp: builtin_interfaces/Time
        return {"sec": int(getattr(stamp, "sec", 0)), "nanosec": int(getattr(stamp, "nanosec", 0))}

    def _now_stamp_dict(self) -> Dict[str, int]:
        now = self.get_clock().now().to_msg()
        return self._stamp_to_dict(now)

    # ---------------- 序列化：Odometry -> dict ----------------
    def _odom_to_json(self, m: Odometry) -> Dict[str, Any]:
        return {
            "header": {
                "stamp": self._stamp_to_dict(m.header.stamp),
                "frame_id": m.header.frame_id,
            },
            "child_frame_id": m.child_frame_id,
            "pose": {
                "pose": {
                    "position": {
                        "x": float(m.pose.pose.position.x),
                        "y": float(m.pose.pose.position.y),
                        "z": float(m.pose.pose.position.z),
                    },
                    "orientation": {
                        "x": float(m.pose.pose.orientation.x),
                        "y": float(m.pose.pose.orientation.y),
                        "z": float(m.pose.pose.orientation.z),
                        "w": float(m.pose.pose.orientation.w),
                    },
                },
                "covariance": [float(v) for v in list(m.pose.covariance)],
            },
            "twist": {
                "twist": {
                    "linear": {
                        "x": float(m.twist.twist.linear.x),
                        "y": float(m.twist.twist.linear.y),
                        "z": float(m.twist.twist.linear.z),
                    },
                    "angular": {
                        "x": float(m.twist.twist.angular.x),
                        "y": float(m.twist.twist.angular.y),
                        "z": float(m.twist.twist.angular.z),
                    },
                },
                "covariance": [float(v) for v in list(m.twist.covariance)],
            },
        }

    # ---------------- 序列化：LaserScan -> dict ----------------
    def _scan_to_json(self, m: LaserScan) -> Dict[str, Any]:
        return {
            "header": {
                "stamp": self._stamp_to_dict(m.header.stamp),
                "frame_id": m.header.frame_id,
            },
            "angle_min": float(m.angle_min),
            "angle_max": float(m.angle_max),
            "angle_increment": float(m.angle_increment),
            "time_increment": float(m.time_increment),
            "scan_time": float(m.scan_time),
            "range_min": float(m.range_min),
            "range_max": float(m.range_max),
            # ranges / intensities 可能包含 NaN/Inf，这里先转 float，再统一在 _sanitize() 里处理
            "ranges": [float(x) for x in list(m.ranges)],
            "intensities": [float(x) for x in list(m.intensities)],
        }

    # ---------------- 工具：递归清洗 NaN / Inf ----------------
    def _sanitize(self, obj: Any) -> Any:
        """
        递归将 NaN/Inf/-Inf 转为 None，以保证严格的 JSON 合法性。
        """
        if isinstance(obj, float):
            if math.isfinite(obj):
                return obj
            return None
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._sanitize(v) for v in obj]
        return obj


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = OdomScanRecorder()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node:
            node.get_logger().info("数据记录器被用户终止")
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

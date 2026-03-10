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
功能：激光雷达避障
描述：订阅激光雷达节点/scan（sensor_msgs/msg/LaserScan），默认持续向前运动；
     当前方检测到障碍物时，自动选择更空的一侧绕行，然后继续直行。
作者：pycodeworld
'''

import signal
import math
import time
from typing import Optional, List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class ObstacleAvoidance(Node):
    """
    简单避障逻辑：
      - 正常状态：以固定线速度前进（linear.x = forward_speed）
      - 若“前方扇区”的最小距离 < stop_distance -> 进入“转向状态”
      - 转向方向：比较左右扇区的平均距离，朝更空的一侧转（angular.z = ±turn_speed）
      - 转到“前方扇区”的最小距离 > clear_distance 后，恢复直行
    角度约定：以雷达坐标系 0rad 为正前方，+逆时针（ROS 标准）
    """

    def __init__(self):
        super().__init__('obstacle_avoidance')

        # ---------------- 参数 ----------------
        # 话题
        self.declare_parameter('scan_topic', 'scan')
        self.declare_parameter('cmd_topic', 'cmd_vel')

        # 运动参数
        self.declare_parameter('forward_speed', 0.05)     # m/s 直行速度
        self.declare_parameter('turn_speed', 0.9)         # rad/s 转向角速度（恒定）
        self.declare_parameter('cmd_rate_hz', 15.0)       # 指令发布频率

        # 距离阈值
        self.declare_parameter('stop_distance', 0.20)     # 前方小于该距离则开始转向
        self.declare_parameter('clear_distance', 0.50)    # 前方大于该距离则恢复直行

        # 扇区角度（度）
        self.declare_parameter('front_sector_deg', 25.0)  # 前方检测扇区半角
        self.declare_parameter('side_sector_deg', 70.0)   # 左/右比较扇区半角（从前向两侧展开）

        # 优先转向（左右相等时的打破平手；1=左转，-1=右转）
        self.declare_parameter('tie_break_turn_left', True)

        # 读取参数
        scan_topic = self.get_parameter('scan_topic').get_parameter_value().string_value
        cmd_topic = self.get_parameter('cmd_topic').get_parameter_value().string_value

        self.forward_speed = float(self.get_parameter('forward_speed').value)
        self.turn_speed = float(self.get_parameter('turn_speed').value)
        self.cmd_rate_hz = float(self.get_parameter('cmd_rate_hz').value)

        self.stop_distance = float(self.get_parameter('stop_distance').value)
        self.clear_distance = float(self.get_parameter('clear_distance').value)

        self.front_sector = math.radians(float(self.get_parameter('front_sector_deg').value))
        self.side_sector = math.radians(float(self.get_parameter('side_sector_deg').value))

        self.tie_break_left = bool(self.get_parameter('tie_break_turn_left').value)

        # ---------------- 通信 ----------------
        # LaserScan 使用传感器 QoS
        self.sub_scan = self.create_subscription(
            LaserScan,
            scan_topic,
            self.scan_cb,
            QoSPresetProfiles.SENSOR_DATA.value
        )
        self.pub_cmd = self.create_publisher(Twist, cmd_topic, 10)

        # ---------------- 定时器 ----------------
        period = 1.0 / self.cmd_rate_hz
        self.timer = self.create_timer(period, self.control_tick)

        # ---------------- 状态 ----------------
        self._last_scan: Optional[LaserScan] = None
        self._last_scan_time: float = 0.0
        self._latest_cmd: Twist = Twist()

        # 行为状态：'FORWARD' 或 'TURN'
        self._mode: str = 'FORWARD'
        # 转向方向：+1 左转 / -1 右转
        self._turn_dir: int = 1 if self.tie_break_left else -1

        # 超时：长时间未收到扫描则停车
        self.declare_parameter('scan_timeout_sec', 0.5)
        self.scan_timeout = float(self.get_parameter('scan_timeout_sec').value)

        # 退出时安全急停
        signal.signal(signal.SIGINT, self._sigint_handler)

        self.get_logger().info(
            "ObstacleAvoidance 已启动：订阅激光雷达进行避障。"
        )

    # ---------------- LaserScan 回调 ----------------
    def scan_cb(self, scan: LaserScan):
        self._last_scan = scan
        self._last_scan_time = time.time()

    # ---------------- 计算工具 ----------------
    @staticmethod
    def _window_indices(scan: LaserScan, ang_lo: float, ang_hi: float) -> Tuple[int, int]:
        """将角度窗口映射到索引区间（包含），并裁剪到有效范围。"""
        n = len(scan.ranges)
        if n == 0 or scan.angle_increment == 0.0:
            return 0, -1  # 空区间
        a_min = scan.angle_min
        inc = scan.angle_increment

        i0 = int(math.ceil((ang_lo - a_min) / inc))
        i1 = int(math.floor((ang_hi - a_min) / inc))
        i0 = _clamp(i0, 0, n - 1)
        i1 = _clamp(i1, 0, n - 1)
        i0, i1 = int(i0), int(i1)
        if i0 > i1:
            return 0, -1
        return i0, i1

    @staticmethod
    def _valid_ranges(scan: LaserScan, i0: int, i1: int) -> List[float]:
        """提取 [i0, i1] 内的有效距离（过滤 NaN/Inf 及超出量程的值）。"""
        out: List[float] = []
        rng_min, rng_max = scan.range_min, scan.range_max
        for i in range(i0, i1 + 1):
            r = scan.ranges[i]
            if math.isfinite(r) and rng_min < r < rng_max:
                out.append(r)
        return out

    def _front_min(self, scan: LaserScan) -> Optional[float]:
        """计算前方扇区最小距离。"""
        # 前方扇区：[-front, +front]
        i0, i1 = self._window_indices(scan, -self.front_sector, +self.front_sector)
        if i1 < i0:
            return None
        vals = self._valid_ranges(scan, i0, i1)
        if not vals:
            return None
        return min(vals)

    def _side_mean(self, scan: LaserScan, left: bool) -> Optional[float]:
        """计算左/右扇区的平均距离，用于决定转向方向。"""
        if left:
            # [ +front, +side ]
            i0, i1 = self._window_indices(scan, +self.front_sector, +self.side_sector)
        else:
            # [ -side, -front ]
            i0, i1 = self._window_indices(scan, -self.side_sector, -self.front_sector)
        if i1 < i0:
            return None
        vals = self._valid_ranges(scan, i0, i1)
        if not vals:
            return None
        return sum(vals) / len(vals)

    # ---------------- 控制循环 ----------------
    def control_tick(self):
        now = time.time()

        # 超时无扫描 -> 安全急停
        if self._last_scan is None or (now - self._last_scan_time) > self.scan_timeout:
            self._safe_stop_once("scan_timeout")
            return

        scan = self._last_scan

        # 计算前方最小距离
        front_min = self._front_min(scan)
        if front_min is None:
            # 读不到有效数据：稳妥起见停下
            self._safe_stop_once("no_valid_front")
            return

        # 状态机切换
        if self._mode != 'TURN' and front_min < self.stop_distance:
            # 需要转向：决定左右
            left_mean = self._side_mean(scan, left=True)
            right_mean = self._side_mean(scan, left=False)

            if left_mean is None and right_mean is None:
                # 都没有有效值，优先按 tie-break
                self._turn_dir = 1 if self.tie_break_left else -1
            elif left_mean is None:
                self._turn_dir = -1
            elif right_mean is None:
                self._turn_dir = 1
            else:
                # 朝更空的一侧转
                if abs(left_mean - right_mean) < 1e-6:
                    self._turn_dir = 1 if self.tie_break_left else -1
                else:
                    self._turn_dir = 1 if left_mean > right_mean else -1

            self._mode = 'TURN'
            self.get_logger().debug(
                f"Obstacle ahead (min={front_min:.2f}m)<{self.stop_distance:.2f}m -> TURN "
                f"{'LEFT' if self._turn_dir>0 else 'RIGHT'}"
            )

        elif self._mode == 'TURN' and front_min > self.clear_distance:
            # 前方重新清空 -> 恢复直行
            self._mode = 'FORWARD'
            self.get_logger().debug(
                f"Path clear (min={front_min:.2f}m)>{self.clear_distance:.2f}m -> FORWARD"
            )

        # 生成控制指令
        cmd = Twist()
        if self._mode == 'FORWARD':
            cmd.linear.x = float(self.forward_speed)
            cmd.angular.z = 0.0
        else:  # TURN
            cmd.linear.x = 0.0
            cmd.angular.z = float(self.turn_speed * (1 if self._turn_dir > 0 else -1))

        self.pub_cmd.publish(cmd)
        self._latest_cmd = cmd

    # ---------------- 安全急停 ----------------
    def _safe_stop_once(self, reason: str = ""):
        stop = Twist()  # 全零
        self.pub_cmd.publish(stop)
        self._latest_cmd = stop
        if reason:
            self.get_logger().debug(f"STOP ({reason})")

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
    node = ObstacleAvoidance()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("节点被用户终止")
    except Exception as e:
        node.get_logger().error(f"异常：{e}")
    finally:
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

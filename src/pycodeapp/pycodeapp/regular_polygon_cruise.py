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
功能：正多边形巡航
描述：
  - 只订阅由 OdomRelPose 发布的**单一** JSON 字符串（std_msgs/String）：
      * 话题（默认）：rel_pose_json
      * 数据结构：
        {
          "offset":{"dx","dy","distance","unit"},
          "yaw":{"rad","deg"},
          "stamp":{"sec","nanosec"}
        }
  - 依据这些相对量进行闭环，控制 OriginBot 走一个给定边长的正多边形（默认正方形）：
      * MOVE_SIDE × N：每段累计弧长达到 side_length 即转弯
      * TURN_CORNER × N：相对航向依次对齐 0, Δ, 2Δ, …, (N-1)Δ（Δ=2π/N），之后再对齐到 0°
      * HOME_TRANSLATE：回到 (dx,dy) ≈ (0,0)
      * HOME_TURN：对齐 d_yaw ≈ 0
作者：pycodeworld
'''

import signal
import math
import time
import json
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data  # ✅ 使用传感器 QoS 预设

from std_msgs.msg import String
from geometry_msgs.msg import Twist


def _wrap_angle(a: float) -> float:
    """wrap 到 (-pi, pi]"""
    return math.atan2(math.sin(a), math.cos(a))


def _deg_norm(deg: float) -> float:
    """角度标准化到 (-180, 180]，用于日志输出"""
    r = (deg + 180.0) % 360.0 - 180.0
    # 把 -180 映射到 180，保持与 _wrap_angle 一致的右开左闭
    return 180.0 if abs(r + 180.0) < 1e-9 else r


class RegularPolygonCruise(Node):
    def __init__(self):
        super().__init__('regular_polygon_cruise')

        # ---------------- 参数 ----------------
        # 仅订阅上一段节点发布的“合并 JSON”话题
        self.declare_parameter('rel_pose_topic', 'rel_pose_json')

        # 底盘控制话题
        self.declare_parameter('cmd_topic', 'cmd_vel')

        # 形状与容差
        self.declare_parameter('side_length', 0.5)       # 每边长度（m）
        self.declare_parameter('num_sides', 4)           # 正多边形边数（默认 4 = 正方形）
        self.declare_parameter('pos_tol_m', 0.02)        # 回家位置容差（m）
        self.declare_parameter('yaw_tol_deg', 2.0)       # 角度容差（deg）

        # 速度与增益
        self.declare_parameter('v_forward', 0.1)         # 直行速度（m/s）
        self.declare_parameter('w_turn', 0.4)            # 转弯最大角速度（rad/s）
        self.declare_parameter('k_yaw_move', 1.2)        # 直行段：偏航 P 增益
        self.declare_parameter('k_turn', 2.0)            # 转弯：偏航 P 增益
        self.declare_parameter('k_home_v', 0.8)          # 回家：距离 -> 线速度 增益
        self.declare_parameter('k_home_w', 2.0)          # 回家：航向误差 -> 角速度 增益
        self.declare_parameter('v_home_max', 0.1)        # 回家最大线速度（m/s）
        self.declare_parameter('w_home_max', 0.8)        # 回家最大角速度（rad/s）

        # 频率与超时
        self.declare_parameter('control_rate_hz', 30.0)
        self.declare_parameter('input_timeout_sec', 0.8)  # 输入超时即停

        # 读取参数
        rel_pose_topic = self.get_parameter(
            'rel_pose_topic').get_parameter_value().string_value
        cmd_topic = self.get_parameter(
            'cmd_topic').get_parameter_value().string_value

        self.side_len = float(self.get_parameter('side_length').value)

        # 正多边形边数检查（至少 3）
        ns = int(self.get_parameter('num_sides').value)
        if ns < 3:
            self.get_logger().warn(f"num_sides={ns} 非法，已回退为 4（正方形）。")
            ns = 4
        self.num_sides = ns

        self.pos_tol = float(self.get_parameter('pos_tol_m').value)
        self.yaw_tol = math.radians(
            float(self.get_parameter('yaw_tol_deg').value))

        self.v_forward = float(self.get_parameter('v_forward').value)
        self.w_turn = float(self.get_parameter('w_turn').value)
        self.k_yaw_move = float(self.get_parameter('k_yaw_move').value)
        self.k_turn = float(self.get_parameter('k_turn').value)

        self.k_home_v = float(self.get_parameter('k_home_v').value)
        self.k_home_w = float(self.get_parameter('k_home_w').value)
        self.v_home_max = float(self.get_parameter('v_home_max').value)
        self.w_home_max = float(self.get_parameter('w_home_max').value)

        self.control_rate_hz = float(
            self.get_parameter('control_rate_hz').value)
        self.input_timeout = float(
            self.get_parameter('input_timeout_sec').value)

        # ---------------- 通信 ----------------
        # 订阅：统一使用 qos_profile_sensor_data（BEST_EFFORT/KEEP_LAST/10）
        self.sub_rel_pose = self.create_subscription(
            String, rel_pose_topic, self._rel_pose_cb, qos_profile_sensor_data
        )

        # 发布：控制命令通常保留可靠传输（此处用默认 depth=10）
        self.pub_cmd = self.create_publisher(Twist, cmd_topic, 10)

        # ---------------- 状态变量 ----------------
        # 最新观测（相对于起点）
        self.dx: Optional[float] = None
        self.dy: Optional[float] = None
        self.dist_from_home: Optional[float] = None
        self.d_yaw: Optional[float] = None

        # 输入时间戳（用于超时守护）
        self.last_in_stamp: float = 0.0  # 以 wall time 记

        # 段内累计弧长（由相邻 (dx,dy) 差分累计）
        self.seg_travel: float = 0.0
        self.prev_dx: Optional[float] = None
        self.prev_dy: Optional[float] = None

        # 方形/多边形段序号与目标航向（以 yaw0 为 0 基准）
        self.leg_idx: int = 0                   # 0..(N-1)
        self.turn_targets = self._build_turn_targets(self.num_sides)

        # FSM
        # WAIT_DATA -> MOVE_SIDE -> TURN_CORNER (循环) -> HOME_TRANSLATE -> HOME_TURN -> DONE
        self.state = 'WAIT_DATA'
        self._in_tol_count = 0                  # 角度容差计数，防抖
        self._yaw_hold_cycles = 3               # 至少 N 个周期满足才认为到位

        # 定时器
        period = 1.0 / max(1e-6, self.control_rate_hz)
        self.timer = self.create_timer(period, self._control_tick)

        # 优雅退出
        signal.signal(signal.SIGINT, self._sigint_handler)

        delta_deg = 360.0 / self.num_sides
        self.get_logger().info(
            f"[PolygonWalker] sides={self.num_sides}, side={self.side_len:.3f} m, v={self.v_forward:.2f} m/s, "
            f"w_turn={self.w_turn:.2f} rad/s, delta={delta_deg:.1f} deg, pos_tol={self.pos_tol:.3f} m, "
            f"yaw_tol={math.degrees(self.yaw_tol):.1f} deg; subscribe='{rel_pose_topic}'"
        )

    # --------- 生成转角序列：0, Δ, 2Δ, …, (N-1)Δ, 0 ----------
    def _build_turn_targets(self, n: int):
        delta = 2.0 * math.pi / float(n)
        targets = [k * delta for k in range(n)]
        targets.append(0.0)  # 末尾 0.0 用于最后一次“归正”转弯
        return targets

    # ---------------- 订阅回调：合并 JSON ----------------
    def _rel_pose_cb(self, msg: String):
        now_wall = time.time()
        try:
            d = json.loads(msg.data)
            disp = d.get('offset', {}) if isinstance(d, dict) else {}
            yawp = d.get('yaw', {}) if isinstance(d, dict) else {}

            dx = disp.get('dx', None)
            dy = disp.get('dy', None)
            dist = disp.get('distance', None)
            dyaw = yawp.get('rad', None)
        except Exception as e:
            self.get_logger().warn(f"rel_pose_json 解析失败：{e}")
            return

        # 字段不全直接忽略（等待下一帧）
        if dx is None or dy is None or dyaw is None:
            return

        self.dx = float(dx)
        self.dy = float(dy)
        self.dist_from_home = float(
            dist) if dist is not None else math.hypot(self.dx, self.dy)
        self.d_yaw = _wrap_angle(float(dyaw))
        self.last_in_stamp = now_wall  # 输入活跃

    # ---------------- 控制主循环 ----------------
    def _control_tick(self):
        # 输入超时保护
        if (time.time() - self.last_in_stamp) > self.input_timeout:
            self._publish_stop()
            if self.state != 'WAIT_DATA':
                self.get_logger().warn("输入超时，急停。")
            self.state = 'WAIT_DATA'
            # 清理段内累计，避免重入时距离突变
            self.prev_dx = self.dx
            self.prev_dy = self.dy
            return

        # 等待数据就绪
        if self.state == 'WAIT_DATA':
            if (self.dx is None) or (self.dy is None) or (self.d_yaw is None):
                self._publish_stop()
                return
            # 初始化段内累计
            self.prev_dx = self.dx
            self.prev_dy = self.dy
            self.seg_travel = 0.0
            self.leg_idx = 0
            self._in_tol_count = 0
            self.state = 'MOVE_SIDE'
            self.get_logger().info("起步：开始第 1 段直行。")
            return

        # 距离积分（以最近两次 (dx,dy) 差分累计弧长）
        if (self.prev_dx is not None) and (self.prev_dy is not None) and (self.dx is not None) and (self.dy is not None):
            dd = math.hypot(self.dx - self.prev_dx, self.dy - self.prev_dy)
            self.seg_travel += dd
            self.prev_dx, self.prev_dy = self.dx, self.dy

        # 状态机
        if self.state == 'MOVE_SIDE':
            self._move_side()
        elif self.state == 'TURN_CORNER':
            self._turn_corner()
        elif self.state == 'HOME_TRANSLATE':
            self._home_translate()
        elif self.state == 'HOME_TURN':
            self._home_turn()
        elif self.state == 'DONE':
            self._publish_stop()  # 保持静止
        else:
            self._publish_stop()

    # ---------------- 各状态动作 ----------------
    def _move_side(self):
        # 当前目标航向（以 yaw0=0 为基准）
        yaw_tgt = self.turn_targets[self.leg_idx]
        yaw_err = _wrap_angle(self.d_yaw - yaw_tgt)

        # 直行同时用小角速度纠偏
        v = self.v_forward
        w = -self.k_yaw_move * yaw_err
        w = max(-self.w_turn, min(self.w_turn, w))

        # 到达该段长度 -> 切换转弯
        if self.seg_travel >= self.side_len:
            # 先停一下让底盘“收敛”
            self._publish_stop()
            self.state = 'TURN_CORNER'
            self._in_tol_count = 0

            tgt_deg = _deg_norm(math.degrees(
                _wrap_angle(self.turn_targets[self.leg_idx + 1])))
            self.get_logger().info(
                f"第 {self.leg_idx + 1} 段到达，转弯至 {tgt_deg:.1f} 度。"
            )
            return

        self._publish_cmd(v, w)

    def _turn_corner(self):
        # 下一个角的目标（依次 0, Δ, 2Δ, …, (N-1)Δ, 最后回到 0）
        yaw_tgt = self.turn_targets[self.leg_idx + 1]
        yaw_err = _wrap_angle(self.d_yaw - yaw_tgt)

        # 纯转向（P 控制）
        w = -self.k_turn * yaw_err
        w = max(-self.w_turn, min(self.w_turn, w))
        v = 0.0

        if abs(yaw_err) <= self.yaw_tol:
            self._in_tol_count += 1
        else:
            self._in_tol_count = 0

        # 稳定满足容差若干周期后判定到位
        if self._in_tol_count >= self._yaw_hold_cycles:
            self._in_tol_count = 0
            self.leg_idx += 1
            self.seg_travel = 0.0
            # 角到位后刷新差分基线，避免刚启动直行时把转弯量计入弧长
            self.prev_dx, self.prev_dy = self.dx, self.dy

            if self.leg_idx <= (self.num_sides - 1):
                self.state = 'MOVE_SIDE'
                self.get_logger().info(f"转弯到位：开始第 {self.leg_idx + 1} 段直行。")
            else:
                # N 段完成，进入回家
                self.state = 'HOME_TRANSLATE'
                self.get_logger().info("多边形完成，开始回到起点。")
            self._publish_stop()
            return

        self._publish_cmd(v, w)

    def _home_translate(self):
        # 根据当前 (dx,dy) 计算“回家”目标的方位角（在 yaw0 坐标：朝向原点）
        dx, dy = self.dx, self.dy
        dist = math.hypot(dx, dy)
        if dist <= self.pos_tol:
            self._publish_stop()
            self.state = 'HOME_TURN'
            self._in_tol_count = 0
            self.get_logger().info("已回到起点附近，准备归正朝向。")
            return

        # 期望朝向：指向 (-dx, -dy)
        bearing_to_home = math.atan2(-dy, -dx)
        yaw_err = _wrap_angle(self.d_yaw - bearing_to_home)

        # 混合式 go-to-goal
        v = min(self.v_home_max, self.k_home_v * dist)
        w = -self.k_home_w * yaw_err
        w = max(-self.w_home_max, min(self.w_home_max, w))

        self._publish_cmd(v, w)

    def _home_turn(self):
        # 目标朝向：d_yaw -> 0
        yaw_err = _wrap_angle(self.d_yaw - 0.0)
        v = 0.0
        w = -self.k_turn * yaw_err
        w = max(-self.w_home_max, min(self.w_home_max, w))

        if abs(yaw_err) <= self.yaw_tol:
            self._in_tol_count += 1
        else:
            self._in_tol_count = 0

        if self._in_tol_count >= self._yaw_hold_cycles:
            self._publish_stop()
            self.state = 'DONE'
            self.get_logger().info("任务完成：已回到初始位置并归正朝向。")
            return

        self._publish_cmd(v, w)

    # ---------------- 发布器与辅助 ----------------
    def _publish_cmd(self, v: float, w: float):
        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(w)
        self.pub_cmd.publish(cmd)

    def _publish_stop(self):
        self.pub_cmd.publish(Twist())

    def _sigint_handler(self, sig, frame):
        self.get_logger().info("SIGINT 收到，急停并退出。")
        try:
            for _ in range(6):
                self.pub_cmd.publish(Twist())
                time.sleep(0.02)
        finally:
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = RegularPolygonCruise()
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

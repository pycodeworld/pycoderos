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

import rclpy,json
from rclpy.node import Node
from pycodemsg.action import CommString
from rclpy.action import ActionServer
'''
功能：行为节点功能示例程序
描述：收到Action数据后每隔1S返回一次进度，总共10次
作者：pycodeworld
'''
class HelloAction(Node):
    def __init__(self, node_name):
        super().__init__(node_name)
        self.get_logger().info(f"启动节点：{node_name}")
         
        self.action_server = ActionServer(self, CommString, node_name, self.action_callback)
        self.max_count = 10  # 最大发送次数

    async def action_callback(self, handle):
        action_handle = handle
        current_count = 0
        timer = self.create_timer(1, lambda: self.timer_callback(action_handle, current_count))
        while current_count < self.max_count:
            await rclpy.task.sleep_until(self.get_clock().now().to_msg())
            current_count += 1
        timer.cancel()
 
    def timer_callback(self, action_handle, current_count):
        res = CommString()
        res.message = f"Hello {action_handle.request.data}!"
        res.percent =  current_count*100/10
        action_handle.publish_feedback(res)
      
    def say_hello(self, msg):
        self.get_logger().info(f'收到消息: {msg.data}')
  

def main():
    rclpy.init()
    node = HelloAction('hello_action')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    if rclpy.ok():
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

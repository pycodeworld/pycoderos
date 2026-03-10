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

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

'''
功能：消息发布节点功能示例程序
描述：在定时器中向主题（topic）：'/my_name' 发布消息，每次间隔1S。
作者：pycodeworld
'''
class HelloTopicSpeaker(Node):
    def __init__(self, node_name):
        super().__init__(node_name)
        self.get_logger().info(f"启动节点：{node_name}")
        self.publisher = self.create_publisher(String, 'my_name', 10)
        self.timer = self.create_timer(1, self.timer_callback)
 
    def timer_callback(self):
        msg = String()
        msg.data = 'world'
        self.publisher.publish(msg)
        self.get_logger().info(f'发布消息: {msg.data}')


def main():
    rclpy.init()
    node = HelloTopicSpeaker('hello_topic_speaker')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    if rclpy.ok():
        node.destroy_node()
        node.timer.cancel() 
        rclpy.shutdown()


if __name__ == '__main__':
    main()

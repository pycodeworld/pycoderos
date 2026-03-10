#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2025, www.pycodeworld.com
# 在线机器人编程平台：www.pycodeworld.com/ros
#

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

'''
功能：消息监听（接收）节点功能示例程序
描述：接收主题（topic）：'/my_name' 中的消息。
作者：pycodeworld
'''


class HelloTopicListener(Node):
    def __init__(self, node_name):
        super().__init__(node_name)
        self.get_logger().info(f"启动节点：{node_name}")
        # 订阅控制命令
        self.subscription = self.create_subscription(
            String, 'my_name', self.say_hello, 10)

    def say_hello(self, msg):
        self.get_logger().info(f'Hello {msg.data}!')


def main():
    rclpy.init()
    node = HelloTopicListener('hello_topic_listener')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    if rclpy.ok():
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

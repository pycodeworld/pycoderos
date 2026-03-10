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
from pycodemsg.srv import CommString # 采用内置的服务数据类型

'''
功能：服务发布节点功能示例程序
描述：收到Service数据后发送回应消息
作者：pycodeworld
'''
class HelloService(Node):
    def __init__(self, node_name):
        super().__init__(node_name)
        self.get_logger().info(f"启动节点：{node_name}")
        self.op_srv = self.create_service(
            CommString, 
            node_name, 
            self.handle_service)
 
    def handle_service(self, request, response):
        response.success = True
        response.message = f"Hello {request.data}!"
        return response


def main():
    rclpy.init()
    node = HelloService('hello_service')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    if rclpy.ok():
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

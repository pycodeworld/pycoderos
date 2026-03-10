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
功能：节点信息发布
描述：查询配置及运行中的节点信息
作者：pycodeworld
'''
from .node_unique import unique_call
import os
import rclpy
import json
import subprocess
from rclpy.node import Node
from pycodemsg.msg import NodeEntry, CommEntry
from pycodemsg.srv import NodeList
from ament_index_python.packages import get_package_share_directory

package_share_dir = get_package_share_directory('pycodebot')


class NodeListService(Node):
    def __init__(self, node_name):
        super().__init__(node_name)
        self.srv = self.create_service(
            NodeList,  node_name, self.handle_query_request)
        self.get_logger().info(f"节点发布器已启动:{node_name}。")
        self.config_path = os.path.join(
            package_share_dir, 'resource', 'nodes_config.json')

    def get_config_node(self, node_name):
        return self.nodes_config.get(node_name)

    def load_nodes_config(self):
        try:
            with open(self.config_path, 'r') as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            self.get_logger().error(f"文件不存在：{self.config_path}")
            return None
        except json.JSONDecodeError:
            self.get_logger().error("错误: 非法JSON格式")
            return None

    def config_to_entries(self):
        node_list = []
        for node_name in self.nodes_config:
            c_node = self.nodes_config[node_name]
            node_entry = NodeEntry(name=node_name, is_running=False)
            node_entry.command = c_node.get("command")
            node_entry.alias = c_node.get("alias")
            publishers = c_node.get("publishers") or []
            for name in publishers:
                pub = publishers[name]
                node_entry.publishers.append(CommEntry(name=name, type=pub.get(
                    "type") or "", command=pub.get("command") or ""))
            subscribers = c_node.get("subscribers") or []
            for name in subscribers:
                sub = subscribers[name]
                node_entry.subscribers.append(CommEntry(name=name, type=sub.get(
                    "type") or "", command=sub.get("command") or ""))
            services = c_node.get("services") or []
            for name in services:
                srv = services[name]
                node_entry.services.append(CommEntry(name=name, type=srv.get(
                    "type") or "", command=srv.get("command") or ""))
            actions = c_node.get("actions") or []
            for name in actions:
                act = actions[name]
                node_entry.actions.append(CommEntry(name=name, type=act.get(
                    "type") or "", command=act.get("command") or ""))
            node_list.append(node_entry)
        return node_list

    def get_config_node_attr(self, node_name, attr=None, item_name=None):
        node = self.get_config_node(node_name)
        if not node:
            return ""
        if not attr:
            return node.get("command") or ""
        if attr in node and item_name in node[attr]:
            return node[attr][item_name].get("command") or ""
        else:
            return ""

    def _is_system_node(self, node_name):
        """判断节点是否为系统默认节点（如/parameter_events）"""
        system_nodes = [
            'rosbridge_websocket',
            '/parameter_events',
            '/rosout',
            'rosapi',
            '_ros2cli_',
            '/_',
        ]
        return any(name in node_name for name in system_nodes)

    def _is_system_topic(self, topic_name):
        """判断话题是否为系统默认话题（如/parameter_events）"""
        system_topics = [
            '/parameter_events',
            '/rosout',
            '/clock',  # 如果使用仿真时间
            '/tf',          # 如果使用TF2
            '/tf_static',    # 如果使用TF2
            '/_',  # 过滤内置主题
        ]
        return any(name in topic_name for name in system_topics)

    def _is_system_service(self, service_name, service_type):
        """判断是否为系统Service"""
        system_services = [
            '/rosout/get_loggers',
            '/parameter_events',
            '/get_type_description',
            '/describe_parameters',
            '/get_parameter_types',
            '/get_parameters',
            '/list_parameters',
            '/set_parameters',
            '/set_parameters_atomically',
            '/_',  # 过滤内置服务
        ]
        # 按名称过滤
        if any(sys_srv in service_name for sys_srv in system_services):
            return True
        return False

    def handle_query_request(self,  request, response):
        """处理查询请求，发布过滤后的节点信息"""
        self.nodes_config = self.load_nodes_config()
        self.get_logger().info(f"节点信息请求:{request}。")
        nodes = self.get_filtered_node_info()

        response.nodes = nodes

        self.get_logger().info(f"返回节点信息:{nodes}。")
        return response

    def get_filtered_node_info(self):
        """获取过滤后的节点信息（排除系统节点和话题）"""
        node_list = []
        node_names = []
        nodes = self.get_node_names_and_namespaces()
        config_list = self.config_to_entries()

        for node_name, space_name in nodes:
            if self._is_system_node(node_name):  # 跳过系统节点
                continue
            if space_name in ["/pycodebot"]:  # 跳过平台节点
                continue

            publishers = self.get_publisher_info(node_name, space_name)
            subscribers = self.get_subscriber_info(node_name, space_name)
            services = self.get_service_info(node_name, space_name)
            actions = []  # self.get_action_info(node_name, space_name)

            # 过滤系统话题
            publishers = [
                pub for pub in publishers if not self._is_system_topic(pub.name)]
            subscribers = [
                sub for sub in subscribers if not self._is_system_topic(sub.name)]

            node_full_name = f"/{node_name}" if space_name in [
                "/", ""] else f"{space_name}/{node_name}"  # 节点全名
            node_names.append(node_full_name)
            node_in_config = self.nodes_config.get(node_full_name)
            if publishers or subscribers or services or actions:
                node_list.append(NodeEntry(
                    name=node_full_name,
                    alias=node_in_config.get(
                        "alias") if node_in_config else "",
                    package="/" if space_name == "" else space_name,
                    publishers=publishers,
                    subscribers=subscribers,
                    services=services,
                    actions=actions,
                    is_running=True,
                    command=node_in_config.get("command") if node_in_config else ""))

        # 添加未运行但再配置中的节点

        for node in config_list:
            if node.name in node_names:
                continue
            node_list.append(node)

        return node_list

    def get_publisher_info(self, node_name, package):
        """获取节点的发布者信息（未过滤）"""
        publishers = []
        try:
            pub_in_running = self.get_publisher_names_and_types_by_node(
                node_name, package)
            for topic_name, topic_types in pub_in_running:
                command = self.get_config_node_attr(
                    node_name, "publishers", topic_name)
                publishers.append(
                    CommEntry(name=topic_name, type=topic_types[0], command=command))
        except Exception as e:
            self.get_logger().warn(
                f"Failed to get publishers for {node_name}: {e}")
        return publishers

    def get_subscriber_info(self, node_name, package):
        """获取节点的订阅者信息（未过滤）"""
        subscribers = []
        try:
            for topic_name, topic_types in self.get_subscriber_names_and_types_by_node(node_name, package):
                command = self.get_config_node_attr(
                    node_name, "subscribers", topic_name)
                subscribers.append(
                    CommEntry(name=topic_name, type=topic_types[0], command=command))
        except Exception as e:
            self.get_logger().warn(
                f"Failed to get subscribers for {node_name}: {e}")
        return subscribers

    def get_service_info(self, node_name, package):
        services = []
        try:
            for service_name, service_types in self.get_service_names_and_types_by_node(node_name, package):
                command = self.get_config_node_attr(
                    node_name, "services", service_name)
                if not self._is_system_service(service_name, service_types[0]):
                    services.append(CommEntry(name=service_name,
                                    type=service_types[0], command=command))
        except Exception as e:
            self.get_logger().warn(
                f"Failed to get services for {node_name}: {e}")
        return services

    ''' 
    def get_action_info(self, node_name, package):
        """获取节点的动作信息（通常不需要过滤）"""
        actions = []
        try:
            for action_name, action_types in self.get_action_server_names_and_types_by_node(node_name, package):

                self.get_logger().warn(f"查询节点行为{node_name}: {action_name}")
                command = self.get_config_node_attr(
                    node_name, "actions", action_name)
                actions.append(CommEntry(name=action_name,
                               type=action_types[0], command=command))
        except Exception as e:
            self.get_logger().warn(f"Failed to get actions for {node_name}: {e}")
        return actions
    '''

    def get_action_info(self, node_name, package):
        # 调用 ros2 node info <node_name>
        node_full_name = f"/{node_name}" if package == "/" else f"{package}/{node_name}"
        result = subprocess.run(
            ["ros2", "node", "info", node_full_name],
            capture_output=True,
            text=True
        )
        self.get_logger().info(
            f"执行：ros2 node info {node_full_name}")
        # 解析输出，提取 Action 信息
        actions = []
        is_action_seg = False
        for line in result.stdout.splitlines():
            if "Action Servers:" in line:
                is_action_seg = True
                continue  # 跳过标题行
            if is_action_seg:
                if line.startswith(" "*4):
                    action_info = line.strip()
                    action_name, action_type = [
                        val.strip() for val in action_info.split(":")]
                    command = self.get_config_node_attr(
                        node_name, "actions", action_name)
                    actions.append(CommEntry(name=action_name,
                                   type=action_type, command=command))
                else:
                    break
        return actions


def main():
    unique_call(NodeListService)


if __name__ == '__main__':
    main()

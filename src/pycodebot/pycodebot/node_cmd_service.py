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
功能：节点控制
描述：节点启动、停止、重启、绑定、解绑、配置等功能。
作者：pycodeworld
'''

from .node_unique import kill_process_by_id, unique_call, kill_process_by_name
import os
import json
import subprocess
from rclpy.node import Node
from pycodemsg.srv import NodeCommand
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import threading

MAX_NUM_COMMANDS = 4  # 命令池最大数量

package_share_dir = get_package_share_directory('pycodebot')


class NodeCmdService(Node):
    def __init__(self, node_name):
        super().__init__(node_name)
        # 订阅控制命令
        self.cmd_server = self.create_service(
            NodeCommand, node_name, self.command_callback)

        qos_profile = QoSProfile(
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )
        self.cmd_counter_id = 1
        self.output_topic = self.create_publisher(
            String, 'cmd_output', qos_profile=qos_profile)
        self.get_logger().info(f"节点控制器已启动:{node_name}。")
        # 运行中的节点
        self.config_path = os.path.join(
            package_share_dir, 'resource', 'nodes_config.json')
        # 当前工作目录
        working_dir = package_share_dir.split('/install/')[0]
        self.current_dir = working_dir  # 初始化当前路径

        self.commands = {}

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

    def get_config_node_command(self, node_name):
        if node_name in self.nodes_config:
            return self.nodes_config[node_name]["command"]
        return None

    def write_nodes_config(self):
        try:
            with open(self.config_path, "w") as f:
                f.write(json.dumps(self.nodes_config))
            return True
        except Exception as e:
            self.get_logger().error(f"写配置出错：{str(e)}")
            return False

    def handle_commander(self, response, command: str):
        _command = f"nohup {command} > /dev/null 2>&1 &"
        self.get_logger().info(f"执行命令：{_command}")
        os.system(_command)

    def handle_terminer_run(self,  response, command):
        if len(self.commands) >= MAX_NUM_COMMANDS:
            key = list(self.commands.keys())[0]
            self.terminer_kill(self.commands[key], key)
            self.get_logger().warning("命令池已满，删除第一个")

        cid = f"c{self.cmd_counter_id}"
        self.commands[cid] = {"cid": cid}
        self.cmd_counter_id += 1

        self.terminer_run_in_thread(command, cid)
        response.success = True
        response.message = cid
        return response

    def terminer_kill(self, cmd, cid):
        if cmd:
            process = cmd["process"]
            self.get_logger().info(f"终止执行:cid={cid}, pid={process.pid}")
            kill_process_by_id(process.pid)

    def handle_terminer_kill(self, response,  resource):
        if resource:
            cmd = self.commands.get(resource)
            self.terminer_kill(cmd, resource)
        else:
            keys = list(self.commands.keys())
            for key in keys:
                self.terminer_kill(self.commands[key], key)
            self.get_logger().info(f"终止所有执行线程")
            self.commands = {}
        response.success = True
        response.message = ""
        return response

    def parse_ros2_log_level(self, line):
        """解析ROS2日志级别并返回小写格式"""
        ros2_log_patterns = [
            '[INFO]',
            '[DEBUG]',
            '[WARN]',
            '[ERROR]',
            '[FATAL]',
        ]
        level = "info"
        for pattern in ros2_log_patterns:
            if line.startswith(pattern):
                level = pattern.strip('[]').lower()
        return level

    def pub_message(self, message):
        if not message:
            return
        level = self.parse_ros2_log_level(message)
        self.output_topic.publish(
            String(data=json.dumps({"type": level, "message": message})))

    def read_stream(self, stream, cid, stream_type):
        while True:
            line = stream.readline()  # 阻塞，直到读到一行
            if not line:  # 如果读到 EOF，说明流已关闭
                self.output_topic.publish(
                    String(data=json.dumps({"type": "end"})))
                del self.commands[cid]
                break
            self.pub_message(line)

    def terminer_run_in_thread(self,  command: str, cid):
        self.get_logger().info(f"线程中执行命令：{command}")
        proc = subprocess.Popen(command, shell=True, cwd=self.current_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, bufsize=1)
        self.commands[cid]["process"] = proc

        stdout_thread = threading.Thread(
            target=self.read_stream,
            args=(proc.stdout, cid, "STDOUT"),
            daemon=True
        )
        stderr_thread = threading.Thread(
            target=self.read_stream,
            args=(proc.stderr, cid, "STDERR"),
            daemon=True
        )

        # 启动线程
        stdout_thread.start()
        stderr_thread.start()
        return None

    def handle_process_result(self, response,  process=None):
        response.success = True if not process else (process.returncode == 0)
        self.get_logger().info(f"节点命令返回：{response.success}")
        if not response.success:
            self.get_logger().error(
                f"命令执行失败({process.returncode})： {process.stderr}")
        return response

    def handle_param_error(self, response, msg="命令参数错误"):
        response.success = False
        response.message = msg
        return response

    async def command_callback(self,  request, response):

        try:
            self.get_logger().info(
                f"收到命令:{request.action}, {request.command}, {request.alias}, {request.resource} ")
            action = request.action
            command = request.command
            alias = request.alias
            resource = request.resource

            self.nodes_config = self.load_nodes_config()
            if self.nodes_config and not command and resource:
                command = self.get_config_node_command(resource)
            if action == 'start':
                return self.start_node(request, response, command)
            elif action == "kill":
                return self.handle_terminer_kill(response, resource)
            elif action == "run":
                return self.handle_terminer_run(response, command)
            elif action == 'stop':
                return self.stop_node(response, resource)
            elif action == 'restart':
                return self.restart_node(response, resource, command)
            elif action == 'bind':
                return self.bind_node(response, resource, command, alias)
            elif action == 'unbind':
                return self.unbind_node(response, resource)
            elif action == 'message':
                return self.config_message(response, resource, command)
            else:
                self.get_logger().warn(f"未知命令: {action}")
        except Exception as e:
            self.get_logger().error(f"处理命令时出错: {str(e)}")
            return self.handle_param_error(response)

    '''启动节点'''

    def start_node(self, request, response, command):
        if not command:  # 生成默认命令
            return self.handle_param_error(response)
        process = self.handle_commander(response, command)
        return self.handle_process_result(response, process)

    '''停止节点'''

    def stop_node(self, response, name):
        if not name:
            return self.handle_param_error(response)
        kill_process_by_name(name)
        return self.handle_process_result(response)

    '''重启节点'''

    def restart_node(self, response,  resource, command):
        if not resource:
            return self.handle_param_error(response)
        kill_process_by_name(resource)
        process = self.handle_commander(response,  command)
        self.get_logger().info(f"重启节点: {resource}")
        return self.handle_process_result(response, process)

    def bind_node(self, response,  resource, command, alias):
        node_in_config = self.nodes_config.get(resource)
        if node_in_config:
            node_in_config["command"] = command
            node_in_config["alias"] = alias
        else:
            self.nodes_config[resource] = {"command": command, "alias": alias}
        self.write_nodes_config()
        return self.handle_process_result(response)

    def unbind_node(self, response,  resource):
        if resource in self.nodes_config:
            self.nodes_config.pop(resource)
            self.write_nodes_config()
        return self.handle_process_result(response)

    def config_message(self, response,  resource, command):
        res = json.loads(resource)
        node_name, attr, item = res["node"], res["attr"], res["item"]
        if node_name not in self.nodes_config:
            self.nodes_config[node_name] = {}
        if attr not in self.nodes_config[node_name]:
            self.nodes_config[node_name][attr] = {}
        self.nodes_config[node_name][attr][item] = command
        return self.handle_process_result(response)


def main():
    unique_call(NodeCmdService)


if __name__ == '__main__':
    main()

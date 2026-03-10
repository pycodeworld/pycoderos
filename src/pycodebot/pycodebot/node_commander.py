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
from .node_unique import unique_call
import os, rclpy, json
from rclpy.node import Node
from rclpy.action import ActionServer
from pycodemsg.action import CmdOperation
from ament_index_python.packages import get_package_share_directory
package_share_dir = get_package_share_directory('pycodebot')

class NodeCommander(Node):
    def __init__(self, node_name):
        super().__init__(node_name)
        # 订阅控制命令
        self.action_server =  ActionServer(self, CmdOperation, node_name, self.command_callback)
        self.get_logger().info(f"节点控制器已启动:{node_name}。")
        # 运行中的节点
        self.config_path = os.path.join(
            package_share_dir, 'resource', 'nodes_config.json')


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
        if node_name in  self.nodes_config:
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


    def kill_node(self, node_name):
        os.system(f'kill $(pgrep -f "{node_name}") 2>&1 &')

 
    def handle_commander(self, handle, command:str):
        _command = f"{command} > /dev/null 2>&1 &"
        self.get_logger().info(f"执行命令：{_command}")
        os.system(_command)
        return None
    
    
    def handle_process_result(self, handle, process=None):
        # 返回最终结果
        result = CmdOperation.Result()
        result.success = True if not process else  (process.returncode == 0)
        self.get_logger().info(f"节点命令返回：{result.success}")
        if result.success:
            handle.succeed()
        else:
            self.get_logger().error(f"命令执行失败({process.returncode})： {process.stderr}")
            handle.abort()
        return result
    

    def handle_param_error(self, handle, msg="命令参数错误"):
        result = CmdOperation.Result()
        result.success = False
        result.message = msg
        handle.abort()
        return result


    async def command_callback(self, handle):
        action = handle.request.action
        command = handle.request.command
        resource = handle.request.resource
        try:
            self.get_logger().info(f"收到命令: {action} {resource} {command}")
            self.nodes_config = self.load_nodes_config()
            if self.nodes_config and not command:
                command = self.get_config_node_command(resource)
            
            self.get_logger().info(f"节点命令: {command}")
            if action == 'start' : 
                return self.start_node(handle, command)
            elif action == "run":
                process = self.handle_commander(handle, command)
                return self.handle_process_result(handle, process)
            elif action == 'stop':
                return self.stop_node(handle, resource)
            elif action == 'restart':
                return self.restart_node(handle, resource, command)
            elif action == 'bind':
                return self.bind_node(handle, resource, command)
            elif action == 'unbind':
                return self.unbind_node(handle, resource)
            elif action == 'message':
                return self.config_message(handle, resource, command)
            else:
                self.get_logger().warn(f"未知命令: {action}")
        except Exception as e:
            self.get_logger().error(f"处理命令时出错: {str(e)}")
            return self.handle_param_error(handle) 
 

    '''启动节点'''
    def start_node(self, handle, command):
        if not command:
            return self.handle_param_error(handle)
        process = self.handle_commander(handle, command)
        return self.handle_process_result(handle, process)


    '''停止节点'''
    def stop_node(self, handle, name):
        if not name:
            return self.handle_param_error(handle)
        self.kill_node(name)
        return self.handle_process_result(handle)

 
    '''重启节点'''
    def restart_node(self, handle,  resource, command):
        if not resource:
            return self.handle_param_error(handle)
        self.kill_node(resource)
        process = self.handle_commander(handle,  command)
        self.get_logger().info(f"重启节点: {resource}")
        return self.handle_process_result(handle, process)


    def bind_node(self, handle,  resource, command):
        node_in_config = self.nodes_config.get(resource)
        if node_in_config:
            node_in_config["command"] = command
        else:
            self.nodes_config[resource] = {"command":command}
        self.write_nodes_config()
        return self.handle_process_result(handle)

    
    def unbind_node(self, handle,  resource):
        if resource in self.nodes_config:
            self.nodes_config.pop(resource)
            self.write_nodes_config()
        return self.handle_param_error(handle)
    

    def config_message(self, handle,  resource, command):
        res = json.loads(resource)
        node_name, attr, item = res["node"], res["attr"], res["item"]
        if node_name not in self.nodes_config:
            self.nodes_config[node_name] = {}
        if attr not in self.nodes_config[node_name]:
            self.nodes_config[node_name][attr] = {}
        self.nodes_config[node_name][attr][item] = command
        return self.handle_param_error(handle)


def main():
    unique_call(NodeCommander)


if __name__ == '__main__':
    main()

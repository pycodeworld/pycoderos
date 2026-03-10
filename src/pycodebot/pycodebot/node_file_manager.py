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
功能：文件管理
描述：对工程文件进行读写、新增、删除、移动、复制等操作。并可创建节点node文件及功能包pacakge
作者：pycodeworld
'''
from .node_unique import unique_call
import os
import shutil
import re
import rclpy
import json
from rclpy.node import Node
from pycodemsg.srv import FileOperation, FileList
from pycodemsg.msg import FileEntry
from ament_index_python.packages import get_package_share_directory
from datetime import datetime

# 排除编译或平台依赖文件，避免误操作
EXCLUDED_FILES = ["/build","/log", "/install", "/pycodebot", "/pycodemsg", 
                 "/pycodebot.env",  "/pycodebot.service",  "/pycodebot.sh"]

class NodeFileManager(Node):
    def __init__(self, node_name):
        super().__init__(node_name)

        package_share_dir = get_package_share_directory('pycodebot')
        # 当前工作目录
        self.working_dir = package_share_dir.split('/install/')[0]

        self.get_logger().info(
            f"文件管理器已启动:{node_name}，工作目录: {self.working_dir}")

        self.op_srv = self.create_service(
            FileOperation,
            'file_operation_service',
            self.handle_operation)

        self.list_srv = self.create_service(
            FileList,
            "file_list_service",
            self.handle_list)
 

    def get_full_path(self, relative_path):
        if relative_path:
            path = os.path.join(self.working_dir, relative_path)
            return os.path.abspath(path)
        else:
            return self.working_dir+"/"

    def get_package_name(self, relative_path):
        return relative_path.split("/")[0]

    def handle_operation(self, request, response):
        operation = request.operation
        # self.get_logger().error(f"{str(request)}")

        try:
            if operation == 'pkg':
                response = self.handle_create_pkg(request, response)
                return response
            if not self.check_file_path(request, response):
                return response
            if operation in ['rb']:  # 读操作
                response = self.handle_read(request, response)
            elif operation in ['wb', 'ab']:  # 写操作
                response = self.handle_write(request, response)
            elif operation == "create":  # 新建文件夹
                response = self.handle_create_dir(request, response)
            elif operation == 'delete':  # 删除操作
                response = self.handle_delete(request, response)
            elif operation == 'copy':  # 拷贝文件
                response = self.handle_copy(request, response)
            elif operation == 'move':  # 移动
                response = self.handle_move(request, response)
            elif operation == "node":
                response = self.handle_create_node(request, response)
            else:
                response.success = False
                response.message = f"未知操作: {operation}"
        except Exception as e:
            response.success = False
            response.message = f"发生异常: {str(e)}"
        return response

    def handle_read(self, request, response):
        """处理文件读取"""
        file_path = self.src_full_path
        if not os.path.exists(file_path):
            response.success = False
            response.message = f"文件不能存在: {request.path}"
            return response
        try:
            with open(file_path, request.operation) as f:
                f.seek(request.pos)
                response.data = list(f.read(2*1024*1024))
                self.get_logger().info(
                    f"从 {request.path} 读取了 {len(response.data)} 字节，从位置 {request.pos}读取")
            response.success = True
            response.message = f"读取文件成功: {request.path}"
        except Exception as e:
            response.success = False
            response.message = f"读取文件错误: {str(e)}"
        return response

    def handle_write(self, request, response):
        file_path = self.src_full_path
        try:
            with open(file_path, request.operation) as f:
                f.write(request.data)
            response.success = True
            response.message = f"写文件成功: {request.path}"
        except Exception as e:
            response.success = False
            response.message = f"写文件失败: {str(e)}"
        return response

    def handle_delete(self,  request, response):
        file_path = self.src_full_path
        if not os.path.exists(file_path):
            response.success = False
            response.message = f"文件或文件夹不存在: {request.path}"
            return response
        try:
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)
            response.success = True
            response.message = f"删除成功: {request.path}"
        except Exception as e:
            response.success = False
            response.message = f"删除失败: {str(e)}"
        return response

    def scan_directory(self, path, files):
        if not os.path.exists(path):
            return
        with os.scandir(path) as it:
            for fd in it:
                # 排除指定目录和文件
                fd_path = fd.path.removeprefix(self.working_dir) 
                if fd_path in EXCLUDED_FILES:
                    continue
                entry = FileEntry()
                entry.is_directory = False
                entry.name = fd.name
                entry.path = fd.path.removeprefix(self.working_dir)
                entry.size = 0
                if fd.is_dir():
                    entry.is_directory = True
                    self.scan_directory(fd.path, files)
                else:
                    stat_info = fd.stat()
                    entry.size = stat_info.st_size
                    entry.mtime = datetime.fromtimestamp(
                        stat_info.st_mtime).strftime('%Y-%m-%d %H:%M:%S')

                files.append(entry)

   

    def check_file_path(self, request, response):
        full_path = self.get_full_path(request.path)
        dest_path = self.get_full_path(request.dest_path) if hasattr(
            request, "dest_path") else None
        # 访问不在工作路径下的非法路径
        if not full_path.startswith(self.working_dir) :
            response.success = False
            response.message = f"源路径不在工程目录下: {request.path}。"
            return False
        if dest_path and not dest_path.startswith(self.working_dir):
            response.success = False
            response.message = f"目标路径不在工程目录下: {request.dest_path}。"
            return False
        self.src_full_path = full_path
        self.dest_full_path = dest_path
        return True

    def handle_list(self, request, response):
        full_path = self.get_full_path(request.path)
        if not full_path.startswith(self.working_dir):
            response.success = False
            response.message = f"源路径不在工程目录下: {request.path}。"

        self.get_logger().info(f"查询目录: {full_path}")
        if not os.path.isdir(full_path):
            response.success = False
            response.message = f"路径不是文件夹：{request.path}"
            return response
        files = []
        try:
            self.scan_directory(full_path, files)
            response.files = files
            response.success = True
            response.message = f"获取文件列表成功：{request.path}"
        except Exception as e:
            response.success = False
            response.message = f"获取文件列表失败: {str(e)}"
        return response

    def handle_copy(self, request, response):
        src_path = self.src_full_path
        dest_path = self.dest_full_path
        try:
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dest_path)
            else:
                shutil.copy(src_path, dest_path)
            response.success = True
            response.message = f"文件拷贝成功：{request.path} => {request.dest_path}"
        except Exception as e:
            response.success = False
            response.message = f"文件拷贝失败：{str(e)}"
        return response

    def handle_move(self, request, response):
        src_path = self.src_full_path
        dest_path = self.dest_full_path
        try:
            shutil.move(src_path, dest_path)
            response.success = True
            response.message = "文件移动成功：{request.path} => {request.dest_path}"
        except Exception as e:
            response.success = False
            response.message = f"文件移动失败：{str(e)}"
        return response

    def handle_create_dir(self, request, response):
        src_path = self.src_full_path
        try:
            os.makedirs(src_path, exist_ok=True)
            response.success = True
            response.message = f"创建文件夹成功：{request.path}"
        except Exception as e:
            response.success = False
            response.message = f"创建文件夹失败：{str(e)}"
        return response

    def handle_create_node(self, request, response):
        src_path = self.src_full_path
        strs = request.path.split("/")

        if len(strs) != 3:
            response.success = False
            response.message = f"创建节点参数错误: {request.path}"
            return
        package_path, package_name, node_name = strs
        node_name: str = node_name.removesuffix(".py")
        class_name = ''.join(part.title() for part in node_name.split("_"))
        content = f'''
import rclpy
from rclpy.node import Node


class {class_name}(Node):
    def __init__(self, node_name):
        super().__init__('{node_name}')


def main():
    rclpy.init()
    node = {class_name}('{node_name}')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    if rclpy.ok():
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
'''
        try:
            with open(src_path, 'w') as f:
                f.write(content)
                f.close()
            response.success = True
            response.message = f"创建节点文件成功： {request.path}"
        except Exception as e:
            response.success = False
            response.message = f"创建节点文件失败：{str(e)}"
        return response

    def handle_create_pkg(self, request, response):
        try:
            os.chdir(self.working_dir)
            command = f'ros2 pkg create {request.path} --build-type ament_python --license Apache-2.0  --dependencies rclpy std_msgs'
            res = os.system(command)
            if res == 0:
                response.success = True
                response.message = f"创建功能包成功： {request.path}"
            else:
                response.success = False
                response.message = f"创建功能包失败：command={command}, working_dir={self.working_dir}"
        except Exception as e:
            error = f"创建功能包失败：{str(e)}"
            self.get_logger().error(error)
            response.success = False
            response.message = error
        return response


def main():
    unique_call(NodeFileManager)


if __name__ == '__main__':
    main()

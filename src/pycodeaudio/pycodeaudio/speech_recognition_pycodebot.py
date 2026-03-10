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

"""
功能：语音识别Pycodebot指令
描述：指令集定义在文件 src/model/speech_recognition_pycodebot/pycodebot.json中
     1. 加载识别指令集中所有的词汇； 2. 解析语音结果中指令； 3.做出响应的动作
作者：pycodeworld
"""
import rclpy
import vosk
import json
import os
from .speech_recognition_model import SpeechRecognitionModel
from .util.util_pycode_json import PycodeJson
from pycodemsg.srv import NodeCommand


class SpeechRecognitionPycodebot(SpeechRecognitionModel):

    def __init__(self, node_name):
        self.client = None
        super().__init__(node_name)

    # 初始化识别器，加载配置文件
    def init_recognizer(self):
        pycode_home = os.getenv('PYCODEBOT_HOME')
        json_file = os.path.join(
            pycode_home, "src", "model", "speech_recognition_pycodebot", "pycodebot.json")

        self.pycoder = PycodeJson(json_file)
        try:
            self.recognizer = vosk.KaldiRecognizer(
                self.model, self.sample_rate, self.pycoder.get_all_keywords())
        except Exception as e:
            self.get_logger().error(f"初始化语音识别器失败: {e}")
            raise

    # 处理语音识别文本
    def process_command(self, text):
        result = self.pycoder.parse(text)
        if not result:
            return

        self.get_logger().info(f"解析命令: {json.dumps(result)}")
        if result["action"] == "compile":
            request.action = "run"
            request.command = f"colcon build --symlink-install --packages-select {result["object"]}"
        else:
            request = NodeCommand.Request()
            request.action = result["action"]
            request.resource = "/" + result["object"]

        if not self.client:
            self.client = self.create_client(
                NodeCommand, '/pycodebot/node_cmd_service')

        future = self.client.call_async(request)
        future.add_done_callback(self.service_callback)

    def service_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info(
                f'服务响应: {response.success}, 消息: {response.message}')
        except Exception as e:
            self.get_logger().error(f'服务调用失败: {e}')


def main(args=None):
    rclpy.init(args=args)
    try:
        node = SpeechRecognitionPycodebot("speech_recognition_pycodebot")
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("节点被用户终止")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

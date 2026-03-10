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
功能：离线语音识别基类
描述：语音识别默认模型：src/model/speech_recognition_model/vosk-model-small-cn-0.22 
作者：pycodeworld
"""

import os
import sys
import json
import queue
import pyaudio
import vosk
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from abc import abstractmethod


class SpeechRecognitionModel(Node):
    def __init__(self, node_name):
        super().__init__(node_name)
        pycode_home = os.getenv('PYCODEBOT_HOME')
        model_parent = f"{pycode_home}/src/model/speech_recognition_model"
        # 参数配置
        self.declare_parameter('model_path',  f"vosk-model-small-cn-0.22")
        self.declare_parameter('sample_rate', 16000)
        # 设置唤醒词，设为空，则一直保持唤醒状态
        self.declare_parameter('wake_words', "")
        # 设置唤醒词情况下保持唤醒时间秒数
        self.declare_parameter('wake_timeout', 60)

        # 获取参数
        model_path = self.get_parameter('model_path').value
        self.sample_rate = self.get_parameter('sample_rate').value
        self.wake_words = self.get_parameter('wake_words').value
        self.wake_timeout = self.get_parameter('wake_timeout').value

        if not model_path.startswith("/"):
            model_path = os.path.join(model_parent, model_path)

        if not os.path.isdir(model_path):
            self.get_logger().error(f"语音识别模型文件不存在{model_path}")
            sys.exit(1)

        self.is_wake = False
        self.last_wake_time = 0
        # 不设置唤醒词则一直保持唤醒状态
        if not self.wake_words:
            self.is_wake = True

        self.get_logger().info("语音识别节点已启动")

        self.speech_pub = self.create_publisher(
            String, f'{node_name}/message', 10)

        self.model = vosk.Model(model_path)
        self.init_recognizer()
        self.start_listening()

    def init_recognizer(self):
        # 初始化Vosk识别器
        try:
            self.recognizer = vosk.KaldiRecognizer(
                self.model, self.sample_rate)
        except Exception as e:
            self.get_logger().error(f"初始化语音识别器失败: {e}")
            raise

    def show_audio_devices(self):
        """获取可用的音频设备"""
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                self.get_logger().info(json.dumps(device_info))

    def audio_callback(self, in_data, frame_count, time_info, status):
        self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)

    def start_listening(self):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_listening = False
        self.audio_queue = queue.Queue()
        self.show_audio_devices()

        """开始语音识别"""
        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=4096,
                stream_callback=self.audio_callback
            )

            self.is_listening = True
            # self._processing_thread = threading.Thread(
            #    target=self.process_audio, daemon=True)
            # self._processing_thread.start()

            self.stream.start_stream()
            self.process_audio()
            self.get_logger().info("开始语音识别...")

        except Exception as e:
            self.get_logger().error(f"启动音频流失败: {e}")

    def stop_listening(self):
        """停止语音识别"""
        self.is_listening = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        self.get_logger().info("停止语音识别")

    def process_audio(self):
        """处理音频数据"""
        while self.is_listening:
            if not rclpy.ok():
                break
            try:
                data = self.audio_queue.get(timeout=1.0)
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get('text', '').strip()
                    if text:
                        self.get_logger().info(f"识别内容：{text}")
                        self._process_command(text)
            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"音频处理错误: {e}")

    def _process_command(self, text):
        current_time = self.get_clock().now()
        if self.wake_words:
            if self.wake_words in text or (current_time - self.last_wake_time).nanoseconds / 1e9 < self.wake_timeout:
                self.last_wake_time = current_time
                self.process_command(text)
        else:
            self.process_command(text)

    def destroy_node(self):
        self.stop_listening()
        super().destroy_node()

    def process_command(self, text):
        self.speech_pub.publish(String(data=text))


def main(args=None):
    rclpy.init(args=args)
    try:
        node = SpeechRecognitionModel("speech_recognition_model")
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

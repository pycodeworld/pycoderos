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
功能：语音识别配置文件解析
描述：获取配置文件中的词汇，注册后让语言识别仅识别配置中的词汇；
     语音识别后的文字解析匹配出对应的动作和对象。
作者：pycodeworld
"""

import json


class PycodeJson:
    def __init__(self, file):
        self.keywords = []
        self.data = None
        with open(file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def find_values(self, data, target_key):
        results = []
        if isinstance(data, dict):
            for key, value in data.items():
                if key == target_key:
                    if isinstance(value, list):
                        results.extend(value)
                    else:
                        results.append(value)
                results.extend(self.find_values(value, target_key))
        elif isinstance(data, list):
            for item in data:
                results.extend(self.find_values(item, target_key))
        return results

    def keywords_list(self, keywords):
        results = []
        if isinstance(keywords, list):
            for word in keywords:
                results.extend(self.keywords_list(word))
        elif isinstance(keywords, str):
            results.extend(keywords.split())
        return results

    def get_all_keywords(self):
        keywords = self.find_values(self.data, "keywords")
        results = self.keywords_list(keywords)
        return json.dumps(results, ensure_ascii=False)

    def keywords_match_text(self, keywords, text):
        if isinstance(keywords, str):
            ws = keywords.split()
            # 必须有一个词在识别结果中
            for w in ws:
                if w in text:
                    return True
            return False
        elif not isinstance(keywords, list):
            return False
        for word in keywords:
            # 必须所有的词在识别结果中出现
            res = self.keywords_match_text(word, text)
            if not res:
                return res
        return True

    def parse(self, text):
        commands = self.data["commands"]
        if not commands:
            print("图片格式错误")
        for command in commands:
            result = {"action": None, "object": None}
            for act in command["actions"]:
                if self.keywords_match_text(act["keywords"], text):
                    result["action"] = act["action"]
            if not result["action"]:
                continue
            for obj in command["objects"]:
                if self.keywords_match_text(obj["keywords"], text):
                    result["object"] = obj["object"]
            if result["object"]:
                return result
        return None

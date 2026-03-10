import os
import json
import re
from typing import List, Dict, Set


class VocabularyGrammarCommands:
    """词汇管理器类，从语法规则自动生成命令"""
    def __init__(self, config_dir: str):
        self.config_dir = config_dir
        self.commands: Set[str] = set()
        self.grammar_rules: Dict[str, str] = {}
        self.defined_rules: Set[str] = set()

    def load_all_vocabulary(self) -> List[str]:
        """加载所有词汇并生成命令"""
        self.load_grammar_rules()
        self.expand_all_rules()

        return list(self.commands)

    def load_grammar_rules(self, filename: str = 'grammar_rules.txt') -> None:
        """加载语法规则文件"""
        file_path = os.path.join(self.config_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"语法规则文件不存在: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                # 跳过空行和注释
                if not line or line.startswith('#'):
                    continue

                # 解析规则定义
                if '=' in line:
                    try:
                        rule_name, rule_content = line.split('=', 1)
                        rule_name = rule_name.strip()
                        rule_content = rule_content.strip()

                        if rule_name in self.grammar_rules:
                            print(f"警告: 规则 '{rule_name}' 被重复定义")

                        self.grammar_rules[rule_name] = rule_content
                        self.defined_rules.add(rule_name)

                    except ValueError:
                        print(f"语法错误第{line_num}行: {line}")

    def expand_all_rules(self) -> None:
        """扩展所有语法规则生成具体命令"""
        for rule_name, rule_content in self.grammar_rules.items():
            if not rule_content.startswith('<'):
                # 直接扩展非引用规则
                self.expand_rule(rule_name, rule_content)

    def expand_rule(self, rule_name: str, rule_content: str, depth: int = 0) -> List[str]:
        """递归扩展语法规则"""
        if depth > 10:  # 防止无限递归
            print(f"警告: 规则 '{rule_name}' 可能包含循环引用")
            return []

        # 处理规则引用
        rule_refs = re.findall(r'<(\w+)>', rule_content)
        if rule_refs:
            # 先扩展引用的规则
            expanded_content = rule_content
            for ref in rule_refs:
                if ref in self.grammar_rules:
                    ref_expansions = self.expand_rule(
                        ref, self.grammar_rules[ref], depth + 1)
                    if ref_expansions:
                        # 用引用规则的扩展替换引用
                        ref_pattern = f'<{ref}>'
                        expanded_content = expanded_content.replace(
                            ref_pattern,
                            f"({' | '.join(ref_expansions)})"
                        )
                else:
                    print(f"警告: 未定义的规则引用: <{ref}>")

            # 递归处理扩展后的内容
            return self.expand_rule(rule_name, expanded_content, depth + 1)

        # 处理可选元素 [xxx]
        if '[' in rule_content and ']' in rule_content:
            optional_pattern = r'\[([^]]+)\]'
            optional_matches = re.findall(optional_pattern, rule_content)
            for match in optional_matches:
                # 生成带可选和不带可选的版本
                with_optional = match
                without_optional = ""
                rule_content = rule_content.replace(
                    f'[{match}]', f'({with_optional} | {without_optional})')

        # 处理选择项 (a | b | c)
        if '(' in rule_content and ')' in rule_content:
            return self.expand_alternatives(rule_name, rule_content)

        # 基本字符串，直接添加
        if not any(c in rule_content for c in '()|[]<>'):
            self.commands.add(rule_content)
            return [rule_content]

        return []

    def expand_alternatives(self, rule_name: str, rule_content: str) -> List[str]:
        """扩展选择项语法 (a | b | c)"""
        # 找到最外层的括号
        stack = []
        alternatives = []
        current = []

        for char in rule_content:
            if char == '(':
                if stack:
                    current.append(char)
                stack.append(char)
            elif char == ')':
                if stack:
                    stack.pop()
                    if not stack:
                        alternatives.append(''.join(current))
                        current = []
                    else:
                        current.append(char)
                else:
                    current.append(char)
            elif char == '|' and not stack:
                if current:
                    alternatives.append(''.join(current).strip())
                    current = []
            else:
                current.append(char)

        if current:
            alternatives.append(''.join(current).strip())

        # 处理每个选择项
        all_expansions = []
        for alt in alternatives:
            if '(' in alt and ')' in alt:
                # 嵌套括号，递归处理
                nested_expansions = self.expand_alternatives(rule_name, alt)
                all_expansions.extend(nested_expansions)
            elif '|' in alt:
                # 内部还有选择项
                sub_alts = [a.strip() for a in alt.split('|')]
                all_expansions.extend(sub_alts)
            else:
                all_expansions.append(alt.strip())

        # 去重并添加到命令集
        unique_expansions = list(set(all_expansions))
        for expansion in unique_expansions:
            if expansion:  # 跳过空字符串
                self.commands.add(expansion)

        return unique_expansions

    def get_all_commands(self) -> List[str]:
        """获取所有生成的命令"""
        return sorted(list(self.commands))

    def create_vosk_grammar(self) -> str:
        """创建Vosk语法JSON"""
        all_commands = self.get_all_commands()
        grammar = all_commands
        return json.dumps(grammar, ensure_ascii=False)

    def save_generated_commands(self, filename: str = 'generated_commands.txt') -> None:
        """保存生成的命令到文件（用于调试）"""
        file_path = os.path.join(self.config_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            for cmd in self.get_all_commands():
                f.write(cmd + '\n')

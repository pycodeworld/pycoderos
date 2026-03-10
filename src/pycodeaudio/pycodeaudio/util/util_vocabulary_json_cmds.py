import os
import json

'''配置文件中加载语言识别命令字'''


class VocabularyJsonCommands:
    def __init__(self, config_dir: str, file: str):
        self.config_dir = config_dir
        self.file = file

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

    def get_commands(self) -> None:
        file_path = os.path.join(self.config_dir, self.file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"配置文件不存在: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            cmds = self.find_values(data, "keywords")
            return json.dumps(cmds, ensure_ascii=False)

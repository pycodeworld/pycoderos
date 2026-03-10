#!/usr/bin/env python3
''' 
import importlib
import rclpy
from rclpy.node import Node
import os
 

def load_nodes_from_directory(node_dir):
    """
    自动加载指定目录下的所有 Python 节点模块
    """
    nodes = []

    # 遍历目录中的所有.py文件
    for filename in os.listdir(node_dir):
        if filename.endswith('.py') and not filename.startswith('_'):
            module_name = filename[:-3]  # 去掉.py后缀

            try:
                # 动态导入模块
                module = importlib.import_module(
                    f"{os.path.basename(node_dir)}.{module_name}")
     
                # 查找模块中的节点类（继承自rclpy.node.Node的类）
                for name, obj in module.__dict__.items():
                    if isinstance(obj, type) and issubclass(obj, Node) and obj != Node:
                        nodes.append(obj(module_name))
            except Exception as e:
                print(f"Failed to load node from {filename}: {str(e)}")

    return nodes


def main(args=None):
    rclpy.init()

    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 加载所有节点
    nodes = load_nodes_from_directory(current_dir)
    executor = rclpy.executors.MultiThreadedExecutor()

    for node in nodes:
        print(f"noode: {node}")
        executor.add_node(node)
    try:
        executor.spin()
    finally:
        for node in nodes:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
'''
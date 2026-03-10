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
功能：物体识别（基于YOLO模型）
描述：识别到物体后，在图像上绘制【中文类别 + 置信度】并发布结构化识别结果
作者：pycodeworld
"""

import os
import sys
import rclpy
import cv2
import numpy as np
from collections import Counter
from ultralytics import YOLO
from .abs_base_recognition import AbsBaseRecognizer, RecognizerMessage


class ObjectRecognizer(AbsBaseRecognizer):
    """
    多物体识别（支持 ONNX 与 PyTorch .pt）
    - model_path 从 ROS2 参数中读取（参数名：'model_path'）
    - 若为相对路径：按 PYCODEBOT_HOME/src/model/object_recognition_model 进行拼接
    - __init__ 仅保留 name（保留原样默认值）
    - 其它推理相关超参作为类属性（class variables）
    """

    # ====== 统一默认配置（类属性，可在外部统一覆盖）======
    CONF_THRES = 0.25
    IOU_THRES = 0.60
    IMGSZ = 640
    DEVICE = 'cpu'      # 'cpu' 或 'cuda:0'
    FP16 = False        # 仅 PT+CUDA 时有效

    MAX_DET = 200
    MIN_BOX_AREA_RATIO = 0.001  # 过滤过小框

    def __init__(self, name: str = 'object_recognition'):
        super().__init__(name)

        pycode_home = os.getenv('PYCODEBOT_HOME')
        model_parent = None
        if pycode_home:
            model_parent = os.path.join(
                pycode_home, 'src', 'model', 'object_recognition')

        # 声明并获取模型路径参数（默认文件名，按需在 launch/命令行覆盖）
        # 注意：这里默认给出文件名，而不是带目录；若是相对路径，会与 model_parent 拼接
        self.declare_parameter('model_path', 'yolov12x.pt')
        model_path_param = self.get_parameter('model_path').value

        # 相对路径则基于 model_parent 拼接；若未设置 PYCODEBOT_HOME，则保持原样
        if not os.path.isabs(model_path_param) and model_parent:
            resolved_model_path = os.path.join(model_parent, model_path_param)
        else:
            resolved_model_path = model_path_param

        # 文件存在性检查（YOLO 模型是“文件”，不是目录）
        if not os.path.isfile(resolved_model_path):
            self.get_logger().error(f"物体识别模型文件不存在：{resolved_model_path}")
            sys.exit(1)

        self.model_path = resolved_model_path

        # 记录模型类型
        lower = self.model_path.lower()
        self.is_onnx = lower.endswith('.onnx')
        self.is_pt = lower.endswith('.pt') or lower.endswith('.pth')

        # 1) 加载 YOLO 模型（Ultralytics 会根据后缀选择后端）
        self.model = YOLO(self.model_path)

        # 2) 推理与可视化参数 —— 来自类属性
        self.conf_thres = float(self.CONF_THRES)
        self.iou_thres = float(self.IOU_THRES)
        self.imgsz = int(self.IMGSZ)
        self.device = str(self.DEVICE)
        self.fp16 = bool(self.FP16)

        self.max_det = int(self.MAX_DET)
        self.min_box_area_ratio = float(self.MIN_BOX_AREA_RATIO)

        # 优先从模型读 names；若无则用 COCO80 兜底
        self.names = getattr(self.model, "names", None) or {
            0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus", 6: "train", 7: "truck", 8: "boat",
            9: "traffic light", 10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird", 15: "cat",
            16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
            25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee", 30: "skis", 31: "snowboard", 32: "sports ball",
            33: "kite", 34: "baseball bat", 35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket",
            39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl", 46: "banana", 47: "apple",
            48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake",
            56: "chair", 57: "couch", 58: "potted plant", 59: "bed", 60: "dining table", 61: "toilet", 62: "tv", 63: "laptop",
            64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave", 69: "oven", 70: "toaster", 71: "sink",
            72: "refrigerator", 73: "book", 74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush"
        }

        # COCO->中文映射（未列到的回退英文）
        self.cn_map = {
            'person': '人', 'bicycle': '自行车', 'car': '汽车', 'motorcycle': '摩托车', 'airplane': '飞机',
            'bus': '公交车', 'train': '火车', 'truck': '卡车', 'boat': '船', 'traffic light': '红绿灯',
            'fire hydrant': '消防栓', 'stop sign': '停车标志', 'parking meter': '停车计时器', 'bench': '长椅',
            'bird': '鸟', 'cat': '猫', 'dog': '狗', 'horse': '马', 'sheep': '羊', 'cow': '牛',
            'elephant': '大象', 'bear': '熊', 'zebra': '斑马', 'giraffe': '长颈鹿',
            'backpack': '背包', 'umbrella': '雨伞', 'handbag': '手提包', 'tie': '领带', 'suitcase': '行李箱',
            'frisbee': '飞盘', 'skis': '滑雪板', 'snowboard': '单板滑雪', 'sports ball': '球', 'kite': '风筝',
            'baseball bat': '棒球棒', 'baseball glove': '棒球手套', 'skateboard': '滑板', 'surfboard': '冲浪板',
            'tennis racket': '网球拍', 'bottle': '瓶子', 'wine glass': '酒杯', 'cup': '杯子', 'fork': '叉子',
            'knife': '刀', 'spoon': '勺子', 'bowl': '碗', 'banana': '香蕉', 'apple': '苹果',
            'sandwich': '三明治', 'orange': '橙子', 'broccoli': '西兰花', 'carrot': '胡萝卜', 'hot dog': '热狗',
            'pizza': '披萨', 'donut': '甜甜圈', 'cake': '蛋糕', 'chair': '椅子', 'couch': '沙发',
            'potted plant': '盆栽', 'bed': '床', 'dining table': '餐桌', 'toilet': '马桶',
            'tv': '电视', 'laptop': '笔记本电脑', 'mouse': '鼠标', 'remote': '遥控器', 'keyboard': '键盘',
            'cell phone': '手机', 'microwave': '微波炉', 'oven': '烤箱', 'toaster': '烤面包机', 'sink': '水槽',
            'refrigerator': '冰箱', 'book': '书', 'clock': '时钟', 'vase': '花瓶', 'scissors': '剪刀',
            'teddy bear': '泰迪熊', 'hair drier': '吹风机', 'toothbrush': '牙刷'
        }

        backend = "ONNXRuntime" if self.is_onnx else (
            "PyTorch" if self.is_pt else "Auto")
        self.get_logger().info(
            f"多物体识别（{backend}）已启动：model={self.model_path}, conf={self.conf_thres}, "
            f"iou={self.iou_thres}, imgsz={self.imgsz}, device={self.device}, fp16={self.fp16}"
        )

        # 标记首帧是否已做过“模型类型检查”
        self._checked_model_kind = False

    # 为类别生成稳定颜色（BGR）
    def _color_for_class(self, cls_id: int):
        hue = (int(cls_id) * 15) % 180
        hsv = np.uint8([[[hue, 200, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        return int(bgr[0]), int(bgr[1]), int(bgr[2])

    def _en_label(self, cls_id: int):
        if isinstance(self.names, dict):
            return self.names.get(int(cls_id), str(cls_id))
        else:
            try:
                return self.names[int(cls_id)]
            except Exception:
                return str(cls_id)

    def _cn_label(self, cls_id: int):
        en = self._en_label(cls_id)
        return self.cn_map.get(en, en)

    def image_callback(self, cv_image):
        try:
            h, w = cv_image.shape[:2]

            # half（fp16）策略：仅 PT 且启用 FP16 时生效
            half_flag = (self.fp16 and not self.is_onnx)

            # 推理
            results = self.model(
                source=cv_image,
                verbose=False,
                conf=self.conf_thres,
                iou=self.iou_thres,
                imgsz=self.imgsz,
                device=self.device,
                half=half_flag,
                agnostic_nms=True,
                max_det=self.max_det
            )[0]

            # 首帧类型检查：防止加载“分类版”模型
            if not self._checked_model_kind:
                self._checked_model_kind = True
                if (getattr(results, "boxes", None) is None or len(results.boxes) == 0) and getattr(results, "probs", None) is not None:
                    raise RuntimeError(
                        "当前加载的是【分类版】YOLO 模型（出现 probs 无 boxes）。请改用【检测版】模型（*.onnx 或 *.pt 均可）。"
                    )

            drew_anything = False
            counts = Counter()
            message_list = []  # <- 用于 RecognizerMessage.message

            if results and results.boxes is not None and len(results.boxes) > 0:
                xyxy = results.boxes.xyxy.cpu().numpy()
                cls = results.boxes.cls.cpu().numpy().astype(int)
                conf = results.boxes.conf.cpu().numpy()

                for i in range(len(cls)):
                    x1, y1, x2, y2 = xyxy[i].astype(int)
                    # 边界裁剪
                    x1 = max(0, min(x1, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    x2 = max(0, min(x2, w - 1))
                    y2 = max(0, min(y2, h - 1))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    # 过滤过小框
                    box_w, box_h = (x2 - x1), (y2 - y1)
                    area_ratio = (box_w * box_h) / float(h * w + 1e-6)
                    if area_ratio < self.min_box_area_ratio:
                        continue

                    en_label = self._en_label(cls[i])
                    cn_label = self._cn_label(cls[i])
                    counts[cn_label] += 1
                    drew_anything = True

                    # 绘制
                    if not self.pub_image_close:
                        color = self._color_for_class(cls[i])
                        cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 2)
                        text = f"{cn_label} {conf[i]*100:.1f}%"
                        cv_image = self.draw_chinese_text(
                            cv_image, text, (x1, max(0, y1 - 32)), color=color
                        )

                    # 追加一条检测记录到 message_list（严格按需求字段命名）
                    message_list.append({
                        "name": en_label,          # 英文类别名
                        "cn_name": cn_label,       # 中文类别名
                        "left": int(x1),
                        "top": int(y1),
                        "height": int(box_h),      # 框高
                        "width": int(box_w),       # 框宽
                        "image_height": int(h),    # 图像高
                        "image_width": int(w)      # 图像宽
                    })

                # 发布图像
                if not self.pub_image_close:
                    self.pub_image = cv_image

                # --- 发布结构化消息 ---
                # value = 识别结果数量；message = 按顺序排列的结果数组
                value = len(message_list)
                self.pub_message = RecognizerMessage(value=value, message=message_list)

                # 可读摘要
                parts = [f"{k}×{v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])]
                self.get_logger().info(f"识别结果摘要: {', '.join(parts) if parts else '无'}；总计{value}")

                # 按要求打印 value:message
                self.get_logger().info(f"{self.pub_message.value}:{self.pub_message.message}")

            # 未检测到：透传原图并发布空结果
            if not drew_anything:
                if not self.pub_image_close:
                    self.pub_image = cv_image
                self.pub_message = RecognizerMessage(value=0, message=[])
                self.get_logger().info(f"{self.pub_message.value}:{self.pub_message.message}")

        except Exception as e:
            self.get_logger().error(f"图像处理错误: {e}")
            try:
                # 失败时 value<0，message 为错误描述字符串
                self.pub_message = RecognizerMessage(value=-1, message=str(e))
                self.get_logger().info(f"{self.pub_message.value}:{self.pub_message.message}")
                if not self.pub_image_close:
                    self.pub_image = cv_image
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ObjectRecognizer()  # model_path 通过 ROS2 参数注入
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node:
            node.get_logger().info("节点被用户终止")
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

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
功能：车牌识别（YOLO 检测 + HyperLPR3 读牌）
描述：检测车牌位置，裁剪后进行 OCR（HyperLPR3），图像上叠加【原始车牌文本 + 置信度】并发布结构化结果
作者：pycodeworld
"""

import os
import sys
import rclpy
import cv2
import numpy as np
from collections import Counter
from typing import Tuple, Optional
import re

# --- OCR 依赖：仅 HyperLPR3 ---
try:
    import hyperlpr3 as lpr3
except Exception:
    lpr3 = None

from ultralytics import YOLO
from .abs_base_recognition import AbsBaseRecognizer, RecognizerMessage


class PlateRecognizer(AbsBaseRecognizer):
    """
    车牌识别节点（支持 ONNX 与 PyTorch .pt/.pth）
    - det_model_path 从 ROS2 参数读取；相对路径基于 PYCODEBOT_HOME 拼接
    - OCR 仅使用 HyperLPR3（若未安装或初始化失败，则禁用 OCR，仅输出检测框）
    """

    # ====== YOLO 检测相关（类属性，可统一覆盖）======
    CONF_THRES = 0.25
    IOU_THRES = 0.60
    IMGSZ = 640
    DEVICE = 'cpu'      # 'cpu' 或 'cuda:0'
    FP16 = False        # 仅 PT+CUDA 时有效
    MAX_DET = 200
    MIN_BOX_AREA_RATIO = 0.0005  # 过滤过小框（相对整图面积）

    # ====== 可视化 ======
    LABEL_LINE_THICKNESS = 2
    BOX_THICKNESS = 2

    # ====== 预处理与融合策略 ======
    CROP_EXPAND_RATIO = 0.10        # OCR 裁剪扩边
    FUSE_VALID_BOOST = 1.05         # 通过车牌格式校验时，融合置信轻微加成
    FUSE_INVALID_PENALTY = 0.85     # 未通过时，融合置信轻微惩罚

    # ====== 中文车牌正则（常见形态） ======
    RE_CIVIL = re.compile(r"^[京沪津渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼港澳][A-Z][A-Z0-9挂学警使领]{5}$")
    RE_NEW_ENERGY = re.compile(r"^[京沪津渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼][A-Z][A-HJ-NP-Z0-9]{5,6}$")
    RE_POLICE = re.compile(r".*警")
    RE_LEARN  = re.compile(r".*学")
    RE_DIPLO  = re.compile(r".*[使领]")  # 外交/领事
    RE_HKMO   = re.compile(r".*[港澳]")  # 港澳
    RE_TMP    = re.compile(r".*[临试挂]") # 临牌/试挂

    # 易混字符规范化（仅用于结构化 normalized，不影响叠字显示）
    AMBIGUOUS_MAP = str.maketrans({
        'O': '0', 'o': '0',
        'I': '1', 'l': '1',
        'Z': '2', 'z': '2',
        'S': '5', 's': '5',
        'B': '8',
        'Q': '0',
        'D': '0'
    })

    def __init__(self, name: str = 'license_plate_recognition'):
        super().__init__(name)

        # 模型父目录（相对路径时使用）
        pycode_home = os.getenv('PYCODEBOT_HOME')
        model_parent = None
        if pycode_home:
            model_parent = os.path.join(pycode_home, 'src', 'model', 'license_plate_recognition')

        # ========== 车牌检测模型路径 ==========
        self.declare_parameter('det_model_path', 'best.pt')
        det_model_path_param = self.get_parameter('det_model_path').value

        if not os.path.isabs(det_model_path_param) and model_parent:
            resolved_det_path = os.path.join(model_parent, det_model_path_param)
        else:
            resolved_det_path = det_model_path_param

        if not os.path.isfile(resolved_det_path):
            self.get_logger().error(f"车牌检测模型文件不存在：{resolved_det_path}")
            sys.exit(1)

        self.det_model_path = resolved_det_path

        # 记录模型类型
        lower = self.det_model_path.lower()
        self.is_onnx = lower.endswith('.onnx')
        self.is_pt = lower.endswith('.pt') or lower.endswith('.pth')

        # 1) 加载 YOLO 车牌检测模型
        self.det_model = YOLO(self.det_model_path)

        # 2) 推理与可视化参数 —— 来自类属性
        self.conf_thres = float(self.CONF_THRES)
        self.iou_thres = float(self.IOU_THRES)
        self.imgsz = int(self.IMGSZ)
        self.device = str(self.DEVICE)
        self.fp16 = bool(self.FP16)
        self.max_det = int(self.MAX_DET)
        self.min_box_area_ratio = float(self.MIN_BOX_AREA_RATIO)
        self.box_thickness = int(self.BOX_THICKNESS)
        self.label_line_thickness = int(self.LABEL_LINE_THICKNESS)

        # ========== OCR：仅 HyperLPR3 ==========
        self.hyperlpr = None
        if lpr3 is None:
            self.get_logger().error("未检测到 hyperlpr3，请先安装：pip install hyperlpr3 onnxruntime；OCR 将被禁用，仅输出检测框。")
        else:
            try:
                # HyperLPR3 捕获器（我们主要对裁剪小图做识别）
                try:
                    det_level = getattr(lpr3, 'DETECT_LEVEL_LOW', 0)
                    self.hyperlpr = lpr3.LicensePlateCatcher(detect_level=det_level)
                except Exception:
                    self.hyperlpr = lpr3.LicensePlateCatcher()
                self.get_logger().info("HyperLPR3 识别器已启用。")
            except Exception as e:
                self.get_logger().error(f"HyperLPR3 初始化失败：{e}；OCR 将被禁用。")
                self.hyperlpr = None

        # 类别名（若模型自带 names 优先用；否则兜底为 'plate'）
        self.names = getattr(self.det_model, "names", None) or {0: "plate"}
        self._checked_model_kind = False

        backend = "ONNXRuntime" if self.is_onnx else ("PyTorch" if self.is_pt else "Auto")
        ocr_mode = "hyperlpr3" if self.hyperlpr is not None else "disabled"
        self.get_logger().info(
            f"车牌识别（{backend}）已启动：det_model={self.det_model_path}, conf={self.conf_thres}, "
            f"iou={self.iou_thres}, imgsz={self.imgsz}, device={self.device}, fp16={self.fp16}, "
            f"ocr={ocr_mode}"
        )

    # ====== 工具函数 ======
    def _en_label(self, cls_id: int):
        if isinstance(self.names, dict):
            return self.names.get(int(cls_id), str(cls_id))
        else:
            try:
                return self.names[int(cls_id)]
            except Exception:
                return str(cls_id)

    def _color_for_plate(self):
        # 统一使用蓝色系（BGR）
        return (255, 128, 0)

    def _expand_and_clip(self, x1, y1, x2, y2, w, h, ratio=None) -> Tuple[int, int, int, int]:
        """
        在裁剪 OCR 区域时适当扩边，避免 cut 掉字符
        """
        if ratio is None:
            ratio = self.CROP_EXPAND_RATIO
        bw = x2 - x1
        bh = y2 - y1
        dx = int(bw * ratio)
        dy = int(bh * ratio)
        nx1 = max(0, x1 - dx)
        ny1 = max(0, y1 - dy)
        nx2 = min(w - 1, x2 + dx)
        ny2 = min(h - 1, y2 + dy)
        return nx1, ny1, nx2, ny2

    def _normalize_plate_candidate(self, text: str) -> str:
        """
        轻量规范化（仅用于结构化）：保留省份汉字，其余位置执行易混字符替换
        """
        if not text:
            return text
        t = text.strip().replace("·", "").replace("•", "").replace(" ", "").replace("-", "")
        if len(t) <= 1:
            return t
        head = t[0]
        tail = t[1:].translate(self.AMBIGUOUS_MAP)
        return head + tail

    def _classify_plate_pattern(self, t_norm: str) -> Tuple[bool, str]:
        """
        返回 (是否有效, 模式名称)
        """
        if not t_norm:
            return False, ""
        # 港澳/警学领使/临试挂 优先标注
        if self.RE_HKMO.search(t_norm):
            return True, "hk_macao"
        if self.RE_POLICE.search(t_norm):
            return True, "police"
        if self.RE_LEARN.search(t_norm):
            return True, "learn"
        if self.RE_DIPLO.search(t_norm):
            return True, "diplomatic"
        if self.RE_TMP.search(t_norm):
            return True, "temp_or_hanging"

        # 民用/新能源
        if self.RE_CIVIL.match(t_norm):
            return True, "civil"
        if self.RE_NEW_ENERGY.match(t_norm) and ("D" in t_norm or "F" in t_norm):
            return True, "new_energy"

        return False, ""

    def _best_ocr_text(self, crop: np.ndarray) -> Tuple[Optional[str], float, Optional[str], bool, str]:
        """
        使用 OCR 读取文本（仅 HyperLPR3）
        返回：原始文本、平均置信度、轻量规范化文本、是否符合车牌格式、匹配模式
        - 不对叠字内容做清洗；仅规范化用于结构化输出
        """
        if self.hyperlpr is None or crop is None or crop.size == 0:
            return None, 0.0, None, False, ""
        try:
            results = self.hyperlpr(crop)  # [(code, conf, type_idx, box), ...]
            if not results:
                return None, 0.0, None, False, ""
            # 取最高置信度结果
            best = max(results, key=lambda r: float(r[1]) if len(r) > 1 else 0.0)
            code = str(best[0]).strip() if len(best) > 0 else ""
            conf_val = float(best[1]) if len(best) > 1 else 0.0
            if code == "":
                return None, 0.0, None, False, ""
            t_norm = self._normalize_plate_candidate(code)
            is_valid, pattern = self._classify_plate_pattern(t_norm)
            return code, conf_val, t_norm, is_valid, pattern
        except Exception as e:
            self.get_logger().warning(f"HyperLPR3 OCR 失败：{e}")
            return None, 0.0, None, False, ""

    # ====== 回调主逻辑 ======
    def image_callback(self, cv_image):
        try:
            h, w = cv_image.shape[:2]
            half_flag = (self.fp16 and not self.is_onnx)

            # 车牌检测
            det = self.det_model(
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

            # 首帧模型型别检查（防止误用分类模型）
            if not self._checked_model_kind:
                self._checked_model_kind = True
                if (getattr(det, "boxes", None) is None or len(det.boxes) == 0) and getattr(det, "probs", None) is not None:
                    raise RuntimeError(
                        "当前加载的是【分类版】YOLO 模型（出现 probs 无 boxes）。请改用【检测版】模型（*.onnx 或 *.pt 均可）。"
                    )

            drew_anything = False
            counts = Counter()
            message_list = []

            if det and det.boxes is not None and len(det.boxes) > 0:
                xyxy = det.boxes.xyxy.cpu().numpy()
                cls = det.boxes.cls.cpu().numpy().astype(int)
                conf = det.boxes.conf.cpu().numpy()

                for i in range(len(cls)):
                    x1, y1, x2, y2 = xyxy[i].astype(int)
                    # 边界裁剪
                    x1 = max(0, min(x1, w - 1)); y1 = max(0, min(y1, h - 1))
                    x2 = max(0, min(x2, w - 1)); y2 = max(0, min(y2, h - 1))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    # 过滤过小框
                    box_w, box_h = (x2 - x1), (y2 - y1)
                    area_ratio = (box_w * box_h) / float(h * w + 1e-6)
                    if area_ratio < self.min_box_area_ratio:
                        continue

                    # 类别名（多数车牌模型只有一个类）
                    en_label = self._en_label(cls[i])
                    counts['车牌'] += 1
                    drew_anything = True

                    # ====== OCR ======
                    nx1, ny1, nx2, ny2 = self._expand_and_clip(x1, y1, x2, y2, w, h, ratio=self.CROP_EXPAND_RATIO)
                    crop = cv_image[ny1:ny2, nx1:nx2]
                    plate_text, ocr_conf, t_norm, is_valid_plate, plate_pattern = self._best_ocr_text(crop)

                    # 融合置信：检测置信 * OCR 置信（无 OCR 时仅用检测置信），再按格式校验轻微调整
                    base_fused = float(conf[i]) * float(ocr_conf if ocr_conf > 0 else 1.0)
                    if plate_text:
                        fused_conf = base_fused * (self.FUSE_VALID_BOOST if is_valid_plate else self.FUSE_INVALID_PENALTY)
                    else:
                        fused_conf = base_fused
                    fused_conf = max(0.0, min(1.0, fused_conf))

                    # ====== 绘制 ======
                    if not self.pub_image_close:
                        color = self._color_for_plate()
                        cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, self.box_thickness)

                        # 有 OCR 文本：显示【原文 + OCR 置信度】；否则显示“车牌 + 检测置信度”
                        if plate_text:
                            text = f"{plate_text} {ocr_conf*100:.1f}%"
                        else:
                            text = f"车牌 {conf[i]*100:.1f}%"

                        cv_image = self.draw_chinese_text(
                            cv_image,
                            text,
                            (x1, max(0, y1 - 32)),
                            color=color
                        )

                    # ====== 结构化消息 ======
                    message = {
                        "name": en_label,                 # YOLO 类别（如 'plate'）
                        "cn_name": "车牌",
                        "left": int(x1),
                        "top": int(y1),
                        "width": int(box_w),
                        "height": int(box_h),
                        "area_ratio": float(area_ratio),
                        "det_confidence": float(conf[i]),
                        "ocr_text": plate_text if plate_text else "",
                        "ocr_confidence": float(ocr_conf),
                        "fused_confidence": float(fused_conf),
                    }
                    if t_norm is not None:
                        message["ocr_text_normalized"] = t_norm
                        message["is_valid_plate"] = bool(is_valid_plate)
                        if plate_pattern:
                            message["plate_pattern"] = plate_pattern

                    message_list.append(message)

                # 发布图像
                if not self.pub_image_close:
                    self.pub_image = cv_image

                # 发布结构化消息
                value = len(message_list)  # 检测框数量
                self.pub_message = RecognizerMessage(value=value, message=message_list)

                # 数量摘要日志
                parts = [f"{k}×{v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])]
                self.get_logger().info(f"识别结果摘要: {', '.join(parts) if parts else '无'}；总计{value}")
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
                if not self.pub_image_close:
                    self.pub_image = cv_image
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PlateRecognizer()  # det_model_path 通过 ROS2 参数注入
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

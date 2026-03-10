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
功能：人脸识别 + 稀疏特征点可视化
描述：动态检测人脸并识别，支持自动注册新人脸；在图像上“少量标注”关键点，
作者：pycodeworld
'''

import rclpy
import cv2
import numpy as np
import face_recognition
import os
import glob
import json
from typing import Dict, List, Tuple

from .abs_base_recognition import AbsBaseRecognizer, RecognizerMessage


class FaceRecognizer(AbsBaseRecognizer):
    def __init__(self, name: str = 'face_recognition'):
        super().__init__(name)

        pycode_home = os.getenv('PYCODEBOT_HOME')
        data_path = os.path.join(pycode_home, "src", "data", name)
        os.makedirs(data_path, exist_ok=True)

        # ================== 识别参数与状态 ==================
        self.known_face_encodings: List[np.ndarray] = []
        self.known_face_labels: List[str] = []       # 不含扩展名
        self.known_face_filenames: List[str] = []    # 含扩展名
        self.face_count = 0

        self.distance_threshold = 0.6
        self.save_dir = data_path

        # ================== 可视化开关与样式 ==================
        # 稀疏标注：点更少、更显脸。可按需打开/关闭。
        self.draw_landmark_points = True
        self.draw_landmark_lines = True
        self.draw_landmark_mesh = False   # ← 默认关闭网格，保证更清爽
        # 样式
        self.landmark_point_radius = 3
        self.landmark_thickness = 1
        self.landmark_color = (255, 255, 255)  # 白色
        self.mesh_color = (255, 255, 255)      # 白色
        # ====================================================

        self._load_known_faces_from_dir()
        self.get_logger().info(f"人脸识别节点已启动，样本目录：{data_path}")

    # ========== 启动时读取 resource/<节点名> 文件夹 ==========
    def _load_known_faces_from_dir(self):
        patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
        img_paths = []
        for p in patterns:
            img_paths.extend(glob.glob(os.path.join(self.save_dir, p)))

        loaded_labels = []
        for path in sorted(img_paths):
            base_with_ext = os.path.basename(path)
            label = os.path.splitext(base_with_ext)[0]
            try:
                image = face_recognition.load_image_file(path)
                boxes = face_recognition.face_locations(image, model="cnn")
                if not boxes:
                    self.get_logger().warn(f"[跳过] {path} 未检测到人脸，未加入特征库")
                    continue
                encodings = face_recognition.face_encodings(image, boxes)
                if len(encodings) == 0:
                    self.get_logger().warn(f"[跳过] {path} 未提取到人脸特征，未加入特征库")
                    continue

                self.known_face_encodings.append(encodings[0])
                self.known_face_labels.append(label)
                self.known_face_filenames.append(base_with_ext)
                loaded_labels.append(label)
            except Exception as e:
                self.get_logger().error(f"[失败] 载入 {path} 出错: {e}")

        self.face_count = len(self.known_face_labels)

        if loaded_labels:
            self.get_logger().info("已载入人脸样本：")
            for lb in loaded_labels:
                self.get_logger().info(f"  - {lb}")
        else:
            self.get_logger().info("样本目录暂无可用人脸，将在识别到新人脸时自动注册并保存。")

    # ========== 关键：将 68 点“稀疏化”为更少的锚点 ==========
    def _reduce_landmarks(
        self,
        landmarks: Dict[str, List[Tuple[int, int]]]
    ) -> Dict[str, List[Tuple[int, int]]]:
        """
        从 face_recognition 的 68 点中挑选关键点，返回更稀疏的五官点集。
        可按需修改索引表以控制“少/多”。
        """
        sel = {}

        # 下颌线（原 17 点：0~16） -> 取 5 个主要轮廓点
        chin = landmarks.get("chin", [])
        chin_idx = [0, 4, 8, 12, 16]  # 两侧角 + 下巴 + 两侧过渡
        sel["chin"] = [chin[i] for i in chin_idx if i < len(chin)]

        # 眉毛（各 5 点） -> 左右各 3 个（外端、中心、内端）
        left_eb = landmarks.get("left_eyebrow", [])
        right_eb = landmarks.get("right_eyebrow", [])
        eb_idx = [0, 2, 4]
        sel["left_eyebrow"] = [left_eb[i] for i in eb_idx if i < len(left_eb)]
        sel["right_eyebrow"] = [right_eb[i] for i in eb_idx if i < len(right_eb)]

        # 鼻梁（4 点） -> 取上端与下端，再加过渡点
        nb = landmarks.get("nose_bridge", [])
        nb_idx = [0, 2, 3]  # 0:山根 3:鼻梁底
        sel["nose_bridge"] = [nb[i] for i in nb_idx if i < len(nb)]

        # 鼻尖（5 点） -> 左、中、右
        nt = landmarks.get("nose_tip", [])
        nt_idx = [0, 2, 4]
        sel["nose_tip"] = [nt[i] for i in nt_idx if i < len(nt)]

        # 眼睛（各 6 点） -> 左右各 4 个（左右眼角 + 上/下中点）
        le = landmarks.get("left_eye", [])
        re = landmarks.get("right_eye", [])
        eye_idx = [0, 2, 3, 5]  # 0/3:左右眼角，2/5:上/下
        sel["left_eye"] = [le[i] for i in eye_idx if i < len(le)]
        sel["right_eye"] = [re[i] for i in eye_idx if i < len(re)]

        # 嘴（各 12 点上唇/下唇） -> 取左右嘴角 + 上中/下中 + 两个过渡点（共 8）
        tl = landmarks.get("top_lip", [])
        bl = landmarks.get("bottom_lip", [])
        # 这些索引在 face_recognition 的顺序里分布较均匀
        lip_idx_top = [0, 3, 6, 9]     # 左角、上中、右角、下中邻位
        lip_idx_bot = [0, 3, 6, 9]     # 左角、下中、右角、上中邻位
        top_sel = [tl[i] for i in lip_idx_top if i < len(tl)]
        bot_sel = [bl[i] for i in lip_idx_bot if i < len(bl)]
        # 合并并做去重（可能包含重复的左右嘴角）
        merge = top_sel + bot_sel
        uniq = []
        seen = set()
        for (x, y) in merge:
            k = (int(x), int(y))
            if k not in seen:
                seen.add(k)
                uniq.append((x, y))
        sel["mouth"] = uniq

        return sel

    # ========== 绘制稀疏五官特征点与简单轮廓 ==========
    def _draw_sparse_features(
        self,
        img: np.ndarray,
        reduced: Dict[str, List[Tuple[int, int]]]
    ) -> List[Tuple[int, int]]:
        """
        在图像上绘制稀疏点与简洁折线；返回所有点坐标（若要网格用）。
        """
        all_pts: List[Tuple[int, int]] = []

        # 下颌线：开放折线
        if "chin" in reduced and len(reduced["chin"]) >= 2:
            pts = np.array(reduced["chin"], dtype=np.int32)
            if self.draw_landmark_lines:
                cv2.polylines(img, [pts], isClosed=False,
                              color=self.landmark_color,
                              thickness=self.landmark_thickness,
                              lineType=cv2.LINE_AA)
            if self.draw_landmark_points:
                for p in pts:
                    cv2.circle(img, tuple(p), self.landmark_point_radius,
                               self.landmark_color, -1, cv2.LINE_AA)
            all_pts.extend(reduced["chin"])

        # 眉毛、眼睛、鼻、嘴：按组画折线（嘴画闭合略显重，这里只连关键点）
        for key in ["left_eyebrow", "right_eyebrow",
                    "nose_bridge", "nose_tip",
                    "left_eye", "right_eye", "mouth"]:
            pts_list = reduced.get(key, [])
            if len(pts_list) == 0:
                continue

            pts = np.array(pts_list, dtype=np.int32)
            # 眼睛/眉毛/鼻梁：用开放折线；鼻尖/嘴：仅画点与少量直连
            if key in ["left_eye", "right_eye", "left_eyebrow", "right_eyebrow", "nose_bridge"]:
                if self.draw_landmark_lines and len(pts) >= 2:
                    cv2.polylines(img, [pts], isClosed=False,
                                  color=self.landmark_color,
                                  thickness=self.landmark_thickness,
                                  lineType=cv2.LINE_AA)
            elif key in ["mouth", "nose_tip"]:
                # 只画点；为了保持轻量不闭合
                pass

            if self.draw_landmark_points:
                for p in pts:
                    cv2.circle(img, tuple(p), self.landmark_point_radius,
                               self.landmark_color, -1, cv2.LINE_AA)

            all_pts.extend(pts_list)

        return all_pts

    # ========== 可选：Delaunay 三角网格（默认关闭） ==========
    def _draw_landmark_mesh(
        self,
        img: np.ndarray,
        pts: List[Tuple[int, int]],
        bbox: Tuple[int, int, int, int]
    ):
        if not self.draw_landmark_mesh or len(pts) < 3:
            return
        h, w = img.shape[:2]
        subdiv = cv2.Subdiv2D((0, 0, w, h))
        for p in pts:
            if 0 <= p[0] < w and 0 <= p[1] < h:
                subdiv.insert((float(p[0]), float(p[1])))

        triangles = subdiv.getTriangleList()
        x1, y1, x2, y2 = bbox
        margin = 5

        def inside(p):
            return (x1 - margin <= p[0] <= x2 + margin and
                    y1 - margin <= p[1] <= y2 + margin)

        for t in triangles:
            p1 = (int(t[0]), int(t[1]))
            p2 = (int(t[2]), int(t[3]))
            p3 = (int(t[4]), int(t[5]))
            if inside(p1) and inside(p2) and inside(p3):
                cv2.line(img, p1, p2, self.mesh_color,
                         self.landmark_thickness, cv2.LINE_AA)
                cv2.line(img, p2, p3, self.mesh_color,
                         self.landmark_thickness, cv2.LINE_AA)
                cv2.line(img, p3, p1, self.mesh_color,
                         self.landmark_thickness, cv2.LINE_AA)

    # ========== 处理图像回调 ==========
    def image_callback(self, cv_image):
        """
        接收BGR图像，进行人脸检测和识别，并发布结果。
        【统一约定】检测到 >=1 张人脸：
            value = 人脸数量
            message = [{ name(不带后缀), area_ratio, left, top, height, width }, ...]
        未检测到/异常：value < 0，message 为文字说明。
        日志：仅在成功检测时打印一条“数量:JSON”，失败为“-1.000:未检测到人脸”，异常为“-2.000:处理错误”。
        """
        try:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(
                rgb_image, model="cnn")

            faces_info = []
            img_h, img_w = cv_image.shape[:2]
            img_area = max(1, img_h * img_w)

            if face_locations:
                face_encodings = face_recognition.face_encodings(
                    rgb_image, face_locations)

                # 全量 68 点（随后做稀疏化）
                face_landmarks_list = face_recognition.face_landmarks(
                    rgb_image, face_locations
                )

                for (top, right, bottom, left), face_encoding, landmarks in zip(
                        face_locations, face_encodings, face_landmarks_list):

                    # —— 匹配/注册（与原逻辑一致）——
                    if self.known_face_encodings:
                        distances = face_recognition.face_distance(
                            self.known_face_encodings, face_encoding)
                        min_index = int(np.argmin(distances))
                        min_distance = float(distances[min_index])

                        if min_distance < self.distance_threshold:
                            msg_filename = os.path.splitext(
                                self.known_face_filenames[min_index])[0]
                        else:
                            self.face_count += 1
                            msg_filename = f"人脸{self.face_count}"
                            self.known_face_encodings.append(face_encoding)
                            self.known_face_labels.append(msg_filename)
                            self.known_face_filenames.append(
                                msg_filename + ".jpg")
                            face_crop = cv_image[max(top, 0):max(bottom, 0),
                                                 max(left, 0):max(right, 0)]
                            if face_crop.size > 0:
                                try:
                                    cv2.imwrite(os.path.join(
                                        self.save_dir, msg_filename + ".jpg"), face_crop)
                                except Exception:
                                    pass
                    else:
                        self.face_count += 1
                        msg_filename = f"人脸{self.face_count}"
                        self.known_face_encodings.append(face_encoding)
                        self.known_face_labels.append(msg_filename)
                        self.known_face_filenames.append(msg_filename + ".jpg")
                        face_crop = cv_image[max(top, 0):max(bottom, 0),
                                             max(left, 0):max(right, 0)]
                        if face_crop.size > 0:
                            try:
                                cv2.imwrite(os.path.join(
                                    self.save_dir, msg_filename + ".jpg"), face_crop)
                            except Exception:
                                pass

                    width = max(0, int(right - left))
                    height = max(0, int(bottom - top))
                    area_ratio = (width * height) / float(img_area)

                    faces_info.append({
                        "name": msg_filename,
                        "area_ratio": float(area_ratio),
                        "left": int(left),
                        "top": int(top),
                        "height": int(height),
                        "width": int(width),
                    })

                    # —— 绘制：人脸框 + 稀疏特征 —— 
                    if not self.pub_image_close:
                        h, w = cv_image.shape[:2]
                        x1 = max(left - 10, 0)
                        y1 = max(top - 10, 0)
                        x2 = min(right + 10, w - 1)
                        y2 = min(bottom + 10, h - 1)

                        # 人脸框
                        cv2.rectangle(cv_image, (x1, y1),
                                      (x2, y2), (0, 255, 0), 2)

                        # 稀疏化并绘制
                        reduced = self._reduce_landmarks(landmarks)
                        all_pts = self._draw_sparse_features(cv_image, reduced)

                        # 如需网格，可将 self.draw_landmark_mesh 设为 True
                        self._draw_landmark_mesh(cv_image, all_pts, (x1, y1, x2, y2))

                        # 名称
                        cv_image = self.draw_chinese_text(
                            cv_image,
                            msg_filename,
                            (x1, max(y1 - 40, 0)),
                            color=(0, 255, 0)
                        )
                        self.pub_image = cv_image

                # —— 仅保留这一条运行日志 —— 
                self.pub_message = RecognizerMessage(
                    value=len(faces_info), message=faces_info)
                self.get_logger().info(
                    f"{len(faces_info)}:{json.dumps(faces_info, ensure_ascii=False)}")

            else:
                self.pub_message = RecognizerMessage(
                    value=-1.0, message="未检测到人脸")
                self.pub_image = None
                self.get_logger().info(f"-1.000:未检测到人脸")

        except Exception as e:
            self.get_logger().error(f"图像处理错误: {e}")
            self.pub_message = RecognizerMessage(value=-2.0, message="处理错误")
            self.get_logger().info(f"-2.000:处理错误")

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = FaceRecognizer()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node:
            node.get_logger().info("人脸识别节点被用户终止")
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

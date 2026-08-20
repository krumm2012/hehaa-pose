# pose_estimator_yolo26.py
# YOLO26-pose 姿态估计器 (支持 Core ML 模型)

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional

try:
    import coremltools as ct
except ModuleNotFoundError:
    ct = None

# COCO Keypoint indices (YOLO26-pose 使用标准 COCO 17 关键点)
# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
# 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
# 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
# 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

class PoseEstimatorYOLO26:
    """YOLO26-pose 姿态估计器"""

    def __init__(self, model_path: str, config: dict, roi_manager=None):
        """
        初始化 YOLO26-pose 姿态估计器

        Args:
            model_path: Core ML 模型路径 (.mlpackage)
            config: 配置字典
            roi_manager: ROI 管理器 (可选)
        """
        print(f"🤖 加载 YOLO26-pose 模型: {model_path}")
        if ct is None:
            raise ImportError(
                "coremltools is required to initialize PoseEstimatorYOLO26. "
                "Install coremltools or run tests that do not instantiate the CoreML model."
            )

        # 获取计算单元配置
        pose_compute_units_str = config.get('pose_compute_units', 'ALL')
        compute_units_map = {
            'CPU': ct.ComputeUnit.CPU_ONLY,
            'GPU': ct.ComputeUnit.CPU_AND_GPU,
            'ANE': ct.ComputeUnit.CPU_AND_NE,
            'ALL': ct.ComputeUnit.ALL
        }
        compute_units = compute_units_map.get(pose_compute_units_str.upper(), ct.ComputeUnit.ALL)

        print(f"   Pose 计算单元: {pose_compute_units_str}")

        # 加载 Core ML 模型
        try:
            self.model = ct.models.MLModel(model_path, compute_units=compute_units)
            print("✅ YOLO26-pose 模型加载成功")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise

        self.config = config
        self.roi_manager = roi_manager

        # 标准 COCO 17 关键点名称
        self.keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]

        # 定义骨架连接
        self.skeleton = [
            ["right_shoulder", "right_elbow"],  # 右上臂
            ["right_elbow", "right_wrist"],     # 右前臂
            ["left_shoulder", "left_elbow"],    # 左上臂
            ["left_elbow", "left_wrist"],       # 左前臂
            ["right_shoulder", "left_shoulder"], # 肩膀连线
            ["right_hip", "left_hip"],          # 髋部连线
            ["right_shoulder", "right_hip"],    # 右侧躯干
            ["left_shoulder", "left_hip"],      # 左侧躯干
            ["right_hip", "right_knee"],        # 右大腿
            ["right_knee", "right_ankle"],      # 右小腿
            ["left_hip", "left_knee"],          # 左大腿
            ["left_knee", "left_ankle"],        # 左小腿
        ]

        # 骨架颜色
        self.colors = {
            "right_arm": (255, 140, 0),    # 橙色 - 右臂
            "left_arm": (135, 206, 235),   # 天蓝色 - 左臂
            "torso": (75, 0, 130),         # 靛蓝色 - 躯干
            "legs": (50, 205, 50)          # 绿色 - 腿部
        }

        # 获取模型输入规格
        self.input_size = self._get_input_size()
        print(f"📐 模型输入尺寸: {self.input_size}")

        # 置信度阈值
        self.confidence_threshold = config.get('pose_confidence_threshold', 0.5)
        self.keypoint_confidence = config.get('pose_keypoint_confidence', 0.3)

        # 时序平滑配置（抑制骨骼节点跳动）
        self.pose_smoothing_enabled = bool(config.get('pose_smoothing_enabled', True))
        self.pose_smoothing_alpha = float(config.get('pose_smoothing_alpha', 0.45))
        self.pose_smoothing_alpha = max(0.05, min(1.0, self.pose_smoothing_alpha))
        self.pose_smoothing_max_jump_px = float(config.get('pose_smoothing_max_jump_px', 90.0))
        self.pose_smoothing_max_jump_px = max(10.0, self.pose_smoothing_max_jump_px)
        self.pose_smoothing_hold_missing_frames = int(config.get('pose_smoothing_hold_missing_frames', 2))
        self.pose_smoothing_hold_missing_frames = max(0, self.pose_smoothing_hold_missing_frames)
        self.pose_smoothing_reset_frames = int(config.get('pose_smoothing_reset_frames', 8))
        self.pose_smoothing_reset_frames = max(1, self.pose_smoothing_reset_frames)
        self.pose_smoothing_min_valid_points = int(config.get('pose_smoothing_min_valid_points', 5))
        self.pose_smoothing_min_valid_points = max(1, self.pose_smoothing_min_valid_points)

        # 平滑状态缓存（仅跟踪主人物）
        self._smoothed_keypoints: Dict[str, Optional[Tuple[int, int]]] = {}
        self._missing_counts: Dict[str, int] = {name: 0 for name in self.keypoint_names}
        self._no_person_frames = 0

    def _get_input_size(self) -> Tuple[int, int]:
        """获取模型输入尺寸"""
        try:
            spec = self.model.get_spec()
            # 通常 YOLO 模型输入是 [1, 3, H, W]
            input_desc = spec.description.input[0]
            if hasattr(input_desc.type, 'imageType'):
                height = input_desc.type.imageType.height
                width = input_desc.type.imageType.width
                return (width, height)
            elif hasattr(input_desc.type, 'multiArrayType'):
                shape = input_desc.type.multiArrayType.shape
                # 假设格式是 [batch, channels, height, width]
                if len(shape) >= 4:
                    return (int(shape[3]), int(shape[2]))
        except Exception as e:
            print(f"⚠️ 无法自动获取输入尺寸，使用默认值 640x640: {e}")

        return (640, 640)  # 默认 YOLO 输入尺寸

    def _preprocess_frame(self, frame: np.ndarray):
        """
        预处理输入帧，转换为 PIL.Image 格式

        Args:
            frame: BGR 格式的输入帧 (numpy array)

        Returns:
            PIL.Image 格式的图像
        """
        from PIL import Image

        # 保存原始尺寸用于后处理
        self.original_size = (frame.shape[1], frame.shape[0])

        # 调整大小到模型输入尺寸
        resized = cv2.resize(frame, self.input_size)

        # BGR 转 RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # 转换为 PIL.Image (Core ML 需要)
        pil_image = Image.fromarray(rgb)

        return pil_image

    def _postprocess_keypoints(self, keypoints: np.ndarray, confidences: np.ndarray) -> List[Dict]:
        """
        后处理关键点，将坐标映射回原始图像尺寸

        Args:
            keypoints: 关键点坐标 [num_persons, num_keypoints, 2]
            confidences: 关键点置信度 [num_persons, num_keypoints]

        Returns:
            关键点字典列表
        """
        person_keypoints_list = []

        # 计算缩放比例
        scale_x = self.original_size[0] / self.input_size[0]
        scale_y = self.original_size[1] / self.input_size[1]

        for person_idx in range(keypoints.shape[0]):
            valid_kpts = {}

            for kp_idx, name in enumerate(self.keypoint_names):
                conf = confidences[person_idx, kp_idx]

                if conf > self.keypoint_confidence:
                    # 映射回原始尺寸
                    x = int(keypoints[person_idx, kp_idx, 0] * scale_x)
                    y = int(keypoints[person_idx, kp_idx, 1] * scale_y)
                    valid_kpts[name] = (x, y)
                else:
                    valid_kpts[name] = None

            person_keypoints_list.append(valid_kpts)

        return person_keypoints_list

    def get_keypoints(self, frame: np.ndarray) -> List[Dict]:
        """
        检测关键点

        Args:
            frame: 输入帧 (BGR 格式)

        Returns:
            关键点字典列表，每个字典对应一个人
        """
        # 预处理
        preprocessed = self._preprocess_frame(frame)

        try:
            # Core ML 推理
            prediction = self.model.predict({'image': preprocessed})

            # 仅首次打印输出键（调试用）
            if not hasattr(self, '_output_keys_printed'):
                # print(f"🔍 模型输出键: {list(prediction.keys())}")  # 生产环境关闭
                self._output_keys_printed = True

            # 解析输出
            # YOLO26-pose 的输出格式可能是 [batch, num_detections, data]
            # data 包含: [x1, y1, x2, y2, conf, class, ...keypoints...]

            # 尝试获取主要输出
            output_key = list(prediction.keys())[0]  # 通常是第一个输出
            output = prediction[output_key]

            # 输出格式分析
            if isinstance(output, np.ndarray):
                # print(f"📊 输出形状: {output.shape}")  # 生产环境关闭

                # YOLO 格式通常是 [1, num_detections, data_size]
                if len(output.shape) == 3:
                    batch_size, num_detections, data_size = output.shape
                    # print(f"   检测数量: {num_detections}, 数据维度: {data_size}")  # 生产环境关闭

                    # 解析每个检测
                    person_keypoints_list = []

                    for i in range(num_detections):
                        detection = output[0, i, :]  # 取第一个 batch

                        # YOLO pose 格式: [x1, y1, x2, y2, conf, class_id, kp1_x, kp1_y, kp1_conf, ...]
                        # 前6个是边界框和类别信息
                        if data_size < 6:
                            continue

                        conf = detection[4]

                        # 过滤低置信度检测
                        if conf < self.confidence_threshold:
                            continue

                        # 提取关键点 (从第6个元素开始，每3个为一组: x, y, conf)
                        keypoints_data = detection[6:]
                        num_keypoints = len(keypoints_data) // 3

                        if num_keypoints != 17:
                            print(f"⚠️ 关键点数量不匹配: {num_keypoints}, 预期 17")
                            continue

                        # 解析关键点
                        valid_kpts = {}
                        for kp_idx in range(17):
                            base_idx = kp_idx * 3
                            if base_idx + 2 < len(keypoints_data):
                                kp_x = keypoints_data[base_idx]
                                kp_y = keypoints_data[base_idx + 1]
                                kp_conf = keypoints_data[base_idx + 2]

                                name = self.keypoint_names[kp_idx]

                                if kp_conf > self.keypoint_confidence:
                                    # 坐标映射回原始尺寸
                                    scale_x = self.original_size[0] / self.input_size[0]
                                    scale_y = self.original_size[1] / self.input_size[1]
                                    x = int(kp_x * scale_x)
                                    y = int(kp_y * scale_y)
                                    valid_kpts[name] = (x, y)
                                else:
                                    valid_kpts[name] = None

                        person_keypoints_list.append(valid_kpts)

                    # 如果没有检测到，返回空列表
                    if not person_keypoints_list:
                        return []

                else:
                    print(f"⚠️ 未知的输出格式: {output.shape}")
                    return []
            else:
                print(f"⚠️ 输出不是 numpy 数组: {type(output)}")
                return []

        except Exception as e:
            print(f"❌ 推理失败: {e}")
            import traceback
            traceback.print_exc()
            return []

        # 应用 ROI 过滤
        if self.roi_manager and self.roi_manager.is_roi_set:
            filtered_keypoints = []
            for keypoints in person_keypoints_list:
                filtered_keypoints_dict = self.roi_manager.filter_detections_by_roi([keypoints], "pose")
                filtered_keypoints.extend(filtered_keypoints_dict)
            person_keypoints_list = filtered_keypoints

        # 时序平滑（在 ROI 过滤后做，保证输出稳定）
        person_keypoints_list = self._apply_temporal_smoothing(person_keypoints_list)

        return person_keypoints_list

    def _apply_temporal_smoothing(self, person_keypoints_list: List[Dict]) -> List[Dict]:
        """Temporal smoothing for the primary person keypoints."""
        if not self.pose_smoothing_enabled:
            return person_keypoints_list

        if not person_keypoints_list:
            self._no_person_frames += 1
            if self._no_person_frames >= self.pose_smoothing_reset_frames:
                self._smoothed_keypoints = {}
                self._missing_counts = {name: 0 for name in self.keypoint_names}
            return person_keypoints_list

        self._no_person_frames = 0
        primary = person_keypoints_list[0] or {}
        valid_points = sum(1 for name in self.keypoint_names if primary.get(name) is not None)
        if valid_points < self.pose_smoothing_min_valid_points:
            return person_keypoints_list

        if not self._smoothed_keypoints:
            initialized = {}
            for name in self.keypoint_names:
                pt = primary.get(name)
                initialized[name] = (int(pt[0]), int(pt[1])) if pt is not None else None
            self._smoothed_keypoints = initialized
            self._missing_counts = {name: 0 for name in self.keypoint_names}
            smoothed_primary = dict(initialized)
            return [smoothed_primary] + person_keypoints_list[1:]

        smoothed_primary: Dict[str, Optional[Tuple[int, int]]] = {}
        alpha = self.pose_smoothing_alpha
        max_jump = self.pose_smoothing_max_jump_px

        for name in self.keypoint_names:
            prev = self._smoothed_keypoints.get(name)
            cur = primary.get(name)

            if cur is None:
                self._missing_counts[name] = self._missing_counts.get(name, 0) + 1
                if prev is not None and self._missing_counts[name] <= self.pose_smoothing_hold_missing_frames:
                    smoothed_primary[name] = prev
                else:
                    smoothed_primary[name] = None
                continue

            self._missing_counts[name] = 0
            cur_xy = np.array([float(cur[0]), float(cur[1])], dtype=float)
            if prev is None:
                smoothed_primary[name] = (int(round(cur_xy[0])), int(round(cur_xy[1])))
                continue

            prev_xy = np.array([float(prev[0]), float(prev[1])], dtype=float)
            delta = cur_xy - prev_xy
            dist = float(np.linalg.norm(delta))

            # Jump clamp: prevent single-frame outliers from snapping skeleton.
            if dist > max_jump and dist > 1e-6:
                cur_xy = prev_xy + delta * (max_jump / dist)

            smooth_xy = prev_xy * (1.0 - alpha) + cur_xy * alpha
            smoothed_primary[name] = (int(round(smooth_xy[0])), int(round(smooth_xy[1])))

        self._smoothed_keypoints = dict(smoothed_primary)
        return [smoothed_primary] + person_keypoints_list[1:]

    def classify_swing(self, keypoints_dict: Dict[int, Dict[str, List[float]]]) -> str:
        """实例方法映射到静态方法"""
        if not keypoints_dict: return "No Person"
        return self.classify_swing_static(keypoints_dict[0], self.config)

    @staticmethod
    def classify_swing_static(kpts: Dict[str, List[float]], config: Dict) -> str:
        """
        判断正手、反手或双反 (静态方法，不依赖模型实例)
        """
        # 必需的关键点
        lw = kpts.get("left_wrist")
        rw = kpts.get("right_wrist")
        ls = kpts.get("left_shoulder")
        rs = kpts.get("right_shoulder")

        if not all([lw, rw, ls, rs]):
            return "Incomplete Pose"

        # 获取配置 (解析字典)
        swing_config = config.get('swing_analysis', {})
        is_mirror = swing_config.get('mirror_view', False)
        dominant_hand = swing_config.get('dominant_hand', 'right')

        # 获取身体中心参考 X
        body_center_x = (ls[0] + rs[0]) / 2

        # 双手反手判断
        wrist_dist = np.linalg.norm(np.array(lw) - np.array(rw))
        if wrist_dist < swing_config.get('two_hand_wrist_distance_max_px', 60):
            avg_wrist_x = (lw[0] + rw[0]) / 2

            if not is_mirror:
                is_backhand_side = (dominant_hand == "right" and avg_wrist_x < body_center_x) or \
                                 (dominant_hand == "left" and avg_wrist_x > body_center_x)
            else:
                is_backhand_side = (dominant_hand == "right" and avg_wrist_x > body_center_x) or \
                                 (dominant_hand == "left" and avg_wrist_x < body_center_x)

            if is_backhand_side:
                return "Two-Handed Backhand"

        # 单手判断
        if dominant_hand == "right":
            active_wrist, active_shoulder = rw, rs
        else:
            active_wrist, active_shoulder = lw, ls

        # 判断正反手
        if not is_mirror:
            if (dominant_hand == "right" and active_wrist[0] < active_shoulder[0]) or \
               (dominant_hand == "left" and active_wrist[0] > active_shoulder[0]):
                return "Backhand"
        else:
            if (dominant_hand == "right" and active_wrist[0] > active_shoulder[0]) or \
               (dominant_hand == "left" and active_wrist[0] < active_shoulder[0]):
                return "Backhand"

        return "Forehand"

    def calculate_angle(self, p1: Tuple, p2: Tuple, p3: Tuple) -> float:
        """
        计算三点之间的角度 (p2 是顶点)

        Args:
            p1, p2, p3: 三个点的坐标

        Returns:
            角度 (度)
        """
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)
        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm_product == 0:
            return 0.0
        angle = np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0))
        return np.degrees(angle)

    def draw_keypoints(self, frame: np.ndarray, person_keypoints_list: List[Dict]) -> np.ndarray:
        """
        绘制关键点和骨架

        Args:
            frame: 输入帧
            person_keypoints_list: 关键点字典列表

        Returns:
            绘制后的帧
        """
        if not person_keypoints_list:
            return frame

        # 绘制第一个检测到的人
        keypoints = person_keypoints_list[0]

        # 头部关键点（跳过绘制）
        head_keypoints = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]

        # 获取人体框范围（排除头部）
        valid_pts = [pt for name, pt in keypoints.items() if pt is not None and name not in head_keypoints]
        if not valid_pts:
            return frame

        x_coords = [pt[0] for pt in valid_pts]
        y_coords = [pt[1] for pt in valid_pts]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # 添加边距
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(frame.shape[1], x_max + padding)
        y_max = min(frame.shape[0], y_max + padding)

        # 绘制标签
        cv2.putText(frame, "tennis player", (x_max - 120, y_max),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 绘制骨架连接线
        for connection in self.skeleton:
            name_a, name_b = connection

            # 跳过头部连接
            if name_a in head_keypoints or name_b in head_keypoints:
                continue

            pt_a, pt_b = keypoints.get(name_a), keypoints.get(name_b)

            if pt_a and pt_b:
                # 选择颜色
                if "arm" in name_a or "arm" in name_b or "wrist" in name_a or "wrist" in name_b or "elbow" in name_a or "elbow" in name_b:
                    if "right" in name_a or "right" in name_b:
                        color = self.colors["right_arm"]
                    else:
                        color = self.colors["left_arm"]
                elif "hip" in name_a or "hip" in name_b or "shoulder" in name_a or "shoulder" in name_b:
                    color = self.colors["torso"]
                else:
                    color = self.colors["legs"]

                cv2.line(frame, pt_a, pt_b, color, 2)

        # 绘制关键点
        for name, pt in keypoints.items():
            if name in head_keypoints or pt is None:
                continue

            # 选择颜色
            if "wrist" in name or "elbow" in name:
                if "right" in name:
                    color = self.colors["right_arm"]
                else:
                    color = self.colors["left_arm"]
            elif "shoulder" in name or "hip" in name:
                color = self.colors["torso"]
            elif "knee" in name or "ankle" in name:
                color = self.colors["legs"]
            else:
                color = (255, 0, 255)

            # 绘制圆点
            cv2.circle(frame, pt, 5, color, -1)

            # 显示编号
            point_id = self.keypoint_names.index(name)
            cv2.putText(frame, str(point_id), (pt[0] + 5, pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    @staticmethod
    def draw_keypoints_static(frame: np.ndarray, person_keypoints_list: List[Dict]) -> np.ndarray:
        from pose_renderer import draw_pose_keypoints

        return draw_pose_keypoints(frame, person_keypoints_list)


# 为了保持向后兼容，创建别名
PoseEstimator = PoseEstimatorYOLO26

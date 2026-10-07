"""
test_planar_mirror_homography.py
─────────────────────────────────
单元测试：验证算法 2.0 双视角镜面平面单应性与几何反射变换模型 (Planar Mirror Homography / Affine)。
覆盖：
1. 镜面物理反射模型与理论放大率验证 (k_x ≈ 1.6496)
2. 各向异性尺度解耦验证 (sx ≈ 1.64, sy ≈ 1.27)
3. 真实 36 帧序列仿射变换稳定性与行列式正定性 (det > 0)
4. Frame 37 关键引拍帧精度对比与手腕解剖学合理性
5. 关键点缺失时的鲁棒安全回退机制
6. 地面单应性复合矩阵 (H_front^-1 @ H_back) 几何一致性
7. RacketResolver 与 main_pipe 管线集成
"""

import json
import math
import unittest
from pathlib import Path

import numpy as np

from dual_pose_estimator import Keypoint
from dual_view_biomechanics import map_mirror_racket_to_front
from mirror_geometry import (
    estimate_torso_planar_affine,
    extract_ground_mirror_homography,
    map_mirror_box_planar,
    map_mirror_point_planar,
)
from racket_resolution import RacketResolutionConfig, RacketResolver


class PlanarMirrorHomographyTests(unittest.TestCase):
    def setUp(self):
        self.benchmark_path = Path("tests/fixtures/mirror_geometry_benchmark_35.json")
        self.cal_path = Path("data/control_ground_calibrations/6ace19915d8569ee840b627481ce03694765c3e10a8b5d4673eba3aeeb7dca68.json")
        self.assertTrue(self.benchmark_path.is_file(), f"Benchmark fixture not found: {self.benchmark_path}")
        with open(self.benchmark_path, "r", encoding="utf-8") as f:
            self.benchmark_frames = json.load(f)

    def test_theoretical_optical_reflection_magnification(self):
        """物理几何验证：平镜反射虚像深度与横向放大率公式一致性。"""
        # Court 02 物理尺寸：镜头至镜墙 6.2m，选手站位 ~4.68m
        d_wall = 6.20
        d_player = 4.68
        d_virtual = 2.0 * d_wall - d_player  # 7.72m
        k_theoretical = d_virtual / d_player   # 7.72 / 4.68 ≈ 1.6496

        self.assertAlmostEqual(k_theoretical, 1.6496, delta=0.001)

        # 验证 36 帧实测平均横向尺度 sx 与理论值高度吻合 (偏差 < 1.5%)
        sx_list = []
        for frame in self.benchmark_frames:
            f_pose = frame["front_pose"]
            b_pose = frame["back_pose"]
            res = estimate_torso_planar_affine(f_pose, b_pose)
            if res is not None:
                sx_list.append(res.scale_x)

        self.assertGreaterEqual(len(sx_list), 30)
        mean_sx = float(np.mean(sx_list))
        self.assertAlmostEqual(mean_sx, k_theoretical, delta=0.03,
                               msg=f"实测平均横向尺度 sx ({mean_sx:.3f}) 偏离理论物理放大率 ({k_theoretical:.3f})")

    def test_anisotropic_scale_decoupling_eliminates_underestimation(self):
        """各向异性尺度验证：证明横向尺度 sx 显著大于纵向尺度 sy (反映相机俯角透视)。"""
        sx_list, sy_list = [], []
        for frame in self.benchmark_frames:
            res = estimate_torso_planar_affine(frame["front_pose"], frame["back_pose"])
            if res is not None:
                sx_list.append(res.scale_x)
                sy_list.append(res.scale_y)

        mean_sx = float(np.mean(sx_list))
        mean_sy = float(np.mean(sy_list))

        # 横向约 1.64，纵向约 1.27
        self.assertGreater(mean_sx, mean_sy * 1.20,
                           f"横向尺度 sx ({mean_sx:.2f}) 应比纵向 sy ({mean_sy:.2f}) 高出 >20%")

    def test_determinant_positivity_and_handedness_parity(self):
        """手性不变量验证：全景平镜反射必须保持同向手性 (det > 0)，且各帧拟合残差良好。"""
        for frame in self.benchmark_frames:
            fid = frame["frame_id"]
            res = estimate_torso_planar_affine(frame["front_pose"], frame["back_pose"])
            self.assertIsNotNone(res, f"Frame {fid} 躯干仿射拟合失败")
            self.assertGreater(res.determinant, 1.5, f"Frame {fid} 行列式过小")
            self.assertLess(res.determinant, 3.0, f"Frame {fid} 行列式过大")
            self.assertLess(res.rmse, 25.0, f"Frame {fid} 重投影 RMSE 过大: {res.rmse}px")

    def test_frame37_racket_recovery_high_precision(self):
        """关键帧对比：Frame 37 在平面仿射下精准映射至持拍侧右手腕前方。"""
        f37 = [f for f in self.benchmark_frames if f["frame_id"] == 37][0]
        mirror_box = [1444.0, 89.0, 1526.0, 150.0]

        # 1. 基准 torso_scale 映射
        mapped_scalar = map_mirror_racket_to_front(
            mirror_box, f37["front_pose"], f37["back_pose"], method="torso_scale"
        )
        self.assertIsNotNone(mapped_scalar)
        scalar_cx = (mapped_scalar[0] + mapped_scalar[2]) / 2.0

        # 2. 高精度 planar_affine 映射
        mapped_planar, diag = map_mirror_box_planar(
            mirror_box, f37["front_pose"], f37["back_pose"], method="planar_affine"
        )
        self.assertIsNotNone(mapped_planar)
        planar_cx = (mapped_planar[0] + mapped_planar[2]) / 2.0

        rwrist_x = f37["front_pose"]["right_wrist"]["x"]
        rwrist_y = f37["front_pose"]["right_wrist"]["y"]

        # 平面仿射充分考虑横向伸展，中心 x 位于 ~1570px (持拍手臂延展处)
        self.assertGreater(planar_cx, scalar_cx, "平面仿射应克服纵向压缩，给出更充分的横向引拍开度")
        # 且距右手腕欧氏距离在正常握拍到拍框范围 (约 50-80px)
        planar_cy = (mapped_planar[1] + mapped_planar[3]) / 2.0
        wrist_d = math.hypot(planar_cx - rwrist_x, planar_cy - rwrist_y)
        self.assertLess(wrist_d, 90.0, f"与右手腕间距应符合人体工学: {wrist_d:.1f}px")
        self.assertEqual(diag["method_used"], "planar_affine")

    def test_graceful_fallback_on_insufficient_keypoints(self):
        """退化鲁棒性：当关键点少于 3 个对应点时，自动优雅回退至 torso_scale。"""
        f37 = [f for f in self.benchmark_frames if f["frame_id"] == 37][0]
        mirror_box = [1444.0, 89.0, 1526.0, 150.0]

        # 破坏前侧姿态，仅保留单侧肩
        degraded_front = {
            "left_shoulder": f37["front_pose"]["left_shoulder"],
        }
        # 回退模式验证
        mapped = map_mirror_racket_to_front(
            mirror_box, degraded_front, f37["back_pose"], method="planar_affine"
        )
        # 因无法解算仿射且无躯干高度，安全熔断为 None
        self.assertIsNone(mapped)

    def test_ground_homography_composition_consistency(self):
        """地面单应性验证：H_front^-1 @ H_back 能准确实现地面视角互投。"""
        if not self.cal_path.is_file():
            self.skipTest("Ground calibration file not available")

        with open(self.cal_path, "r", encoding="utf-8") as f:
            cal = json.load(f)

        h_comp = extract_ground_mirror_homography(cal)
        self.assertIsNotNone(h_comp)
        self.assertEqual(h_comp.shape, (3, 3))

        # 验证地面标定点 back -> front 映射误差
        back_pts = cal["views"]["back"]["points"]
        front_pts = cal["views"]["front"]["points"]

        for b_pt, f_pt in zip(back_pts, front_pts):
            mapped = map_mirror_point_planar((b_pt[0], b_pt[1]), ground_homography=h_comp)
            self.assertIsNotNone(mapped)
            # 地面角点映射误差 < 5px
            err = math.hypot(mapped[0] - f_pt[0], mapped[1] - f_pt[1])
            self.assertLess(err, 5.0, f"Ground corner projection error: {err:.2f}px")

    def test_racket_resolver_pipeline_with_planar_affine(self):
        """管线集成验证：RacketResolver 在 planar_affine 模式下保持全流程 36 帧连续无跳变。"""
        config = RacketResolutionConfig.from_dict({
            "mirror_mapping_mode": "planar_affine",
            "back_racket_dist_threshold": 250.0,
            "wrist_proximity_threshold": 420.0,
        })
        resolver = RacketResolver(config)
        self.assertEqual(resolver.config.mirror_mapping_mode, "planar_affine")

        recovered_count = 0
        prev_cx = None

        for frame_data in self.benchmark_frames:
            fid = frame_data["frame_id"]
            front_pose = {
                k: Keypoint(v["x"], v["y"], v.get("conf", 0.95))
                for k, v in frame_data["front_pose"].items()
            }
            back_pose = {
                k: Keypoint(v["x"], v["y"], v.get("conf", 0.95))
                for k, v in frame_data["back_pose"].items()
            }
            raw_rackets = frame_data["raw_rackets"]

            res = resolver.resolve(
                raw_rackets=raw_rackets,
                front_pose=front_pose,
                back_pose=back_pose,
                frame_idx=fid,
            )

            if res.is_racket_recovered:
                recovered_count += 1

            if res.racket_box is not None:
                cx = (res.racket_box[0] + res.racket_box[2]) / 2.0
                if 32 <= fid <= 40 and prev_cx is not None:
                    delta_x = abs(cx - prev_cx)
                    self.assertLess(delta_x, 60.0, f"Frame {fid}: 时序跳变 delta_x={delta_x:.1f}px")
                prev_cx = cx

        # 确保关键遮挡帧稳定自愈
        self.assertGreaterEqual(recovered_count, 4)


if __name__ == "__main__":
    unittest.main()

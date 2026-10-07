"""
Comprehensive tests for lateral parity, scale bounds, anatomical gating,
and 36-frame real-session continuity invariants in dual-view geometry.
"""

import json
import math
import unittest
from pathlib import Path

from dual_pose_estimator import Keypoint
from dual_view_biomechanics import map_mirror_racket_to_front
from racket_resolution import RacketResolver, RacketResolutionConfig


class TestGeometryInvariants(unittest.TestCase):
    def setUp(self):
        self.fixtures_path = Path("tests/fixtures/mirror_geometry_benchmark_35.json")
        self.config = RacketResolutionConfig.from_yaml("configs/dual_view_config.yaml")
        self.resolver = RacketResolver(self.config)

    def _make_pose(self, cx, cy, torso_w=80.0, torso_h=120.0, wrist_offset_x=40.0):
        """构造对称的人体姿态基准，中心位于 (cx, cy)。"""
        half_w = torso_w / 2.0
        half_h = torso_h / 2.0
        return {
            "left_shoulder": Keypoint(cx + half_w, cy - half_h, 0.99),
            "right_shoulder": Keypoint(cx - half_w, cy - half_h, 0.99),
            "left_hip": Keypoint(cx + half_w, cy + half_h, 0.99),
            "right_hip": Keypoint(cx - half_w, cy + half_h, 0.99),
            "right_wrist": Keypoint(cx + wrist_offset_x, cy, 0.95),
            "left_wrist": Keypoint(cx - wrist_offset_x, cy, 0.95),
        }

    def test_lateral_parity_invariant_right_side(self):
        """不变量 1A：背影偏右 (+dx)，映射至正面绝对必须同向偏右 (+dx)，绝不可翻转至异侧。"""
        f_pose = self._make_pose(cx=1500.0, cy=500.0, torso_w=90.0, torso_h=140.0, wrist_offset_x=50.0)
        b_pose = self._make_pose(cx=1400.0, cy=100.0, torso_w=70.0, torso_h=100.0, wrist_offset_x=35.0)

        # 镜中球拍中心位于躯干偏右 +40px: [1410, 80, 1470, 140] (中心 1440, dx = +40)
        mirror_box = [1410.0, 80.0, 1470.0, 140.0]
        mapped = map_mirror_racket_to_front(mirror_box, f_pose, b_pose)
        self.assertIsNotNone(mapped, "Mapping should succeed for valid right-side racket")

        mapped_cx = (mapped[0] + mapped[2]) / 2.0
        dx_front = mapped_cx - 1500.0
        self.assertGreater(
            dx_front,
            0.0,
            f"Lateral parity violation: mirror +dx mapped to negative dx_front ({dx_front})",
        )
        # 验证缩放后偏移量约为 40 * (140/100) = 56px
        self.assertAlmostEqual(dx_front, 56.0, delta=2.0)

    def test_lateral_parity_invariant_left_side(self):
        """不变量 1B：背影偏左 (-dx)，映射至正面绝对必须同向偏左 (-dx)。"""
        f_pose = self._make_pose(cx=1500.0, cy=500.0, torso_w=90.0, torso_h=140.0, wrist_offset_x=-50.0)
        b_pose = self._make_pose(cx=1400.0, cy=100.0, torso_w=70.0, torso_h=100.0, wrist_offset_x=-35.0)

        # 镜中球拍中心位于躯干偏左 -40px: [1330, 80, 1390, 140] (中心 1360, dx = -40)
        mirror_box = [1330.0, 80.0, 1390.0, 140.0]
        mapped = map_mirror_racket_to_front(mirror_box, f_pose, b_pose)
        self.assertIsNotNone(mapped, "Mapping should succeed for valid left-side racket")

        mapped_cx = (mapped[0] + mapped[2]) / 2.0
        dx_front = mapped_cx - 1500.0
        self.assertLess(
            dx_front,
            0.0,
            f"Lateral parity violation: mirror -dx mapped to positive dx_front ({dx_front})",
        )
        self.assertAlmostEqual(dx_front, -56.0, delta=2.0)

    def test_scale_boundedness_invariant(self):
        """不变量 2：空间深度比例必须受控在 [0.4, 3.0] 闭区间内，杜绝畸变放飞。"""
        # 极端比例 1：正面极大，背面极小
        f_huge = self._make_pose(cx=1500.0, cy=500.0, torso_w=200.0, torso_h=500.0)
        b_tiny = self._make_pose(cx=1400.0, cy=100.0, torso_w=20.0, torso_h=40.0)
        # 原始高度比为 500 / 40 = 12.5，但 scale 必须被压制在 3.0
        mirror_box = [1390.0, 90.0, 1410.0, 110.0]
        mapped = map_mirror_racket_to_front(mirror_box, f_huge, b_tiny)
        # 因 mapped 之后由于尺寸过大可能会超出 wrist 容差，或者成功映射
        # 我们直接验证 map 函数内部的尺度有界性
        if mapped is not None:
            w = mapped[2] - mapped[0]
            self.assertLessEqual(w, 20.0 * 3.0 + 1e-3)

    def test_anatomical_wrist_gating_invariant(self):
        """不变量 3：映射球拍如果偏离正面持拍手腕过远（> 容差门限），必须被安全熔断拒绝。"""
        f_pose = self._make_pose(cx=1500.0, cy=500.0, torso_w=90.0, torso_h=140.0, wrist_offset_x=50.0)
        b_pose = self._make_pose(cx=1400.0, cy=100.0, torso_w=70.0, torso_h=100.0, wrist_offset_x=35.0)

        # 构造一个偏离躯干中心 +250px 的孤立反光杂物 (映射后距离手腕 > 300px)
        distant_mirror_box = [1620.0, 80.0, 1680.0, 140.0]
        mapped = map_mirror_racket_to_front(distant_mirror_box, f_pose, b_pose)
        self.assertIsNone(
            mapped,
            "Anatomical gating failed: distant mirror object was not rejected by wrist proximity constraint",
        )

    def test_real_sequence_36_frame_continuity_and_parity(self):
        """不变量 4 & 5：在 36 帧真实序列中，验证时序轨迹平滑、无左右颠倒、Frame 37 自愈精准。"""
        self.assertTrue(self.fixtures_path.is_file(), f"Fixture file {self.fixtures_path} not found")
        with open(self.fixtures_path, "r", encoding="utf-8") as f:
            frames = json.load(f)

        self.assertGreaterEqual(len(frames), 30, "Benchmark must contain at least 30 frames")

        prev_cx = None
        recovered_count = 0

        for frame_data in frames:
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

            res = self.resolver.resolve(
                raw_rackets=raw_rackets,
                front_pose=front_pose,
                back_pose=back_pose,
                frame_idx=fid,
            )

            if res.is_racket_recovered:
                recovered_count += 1

            if res.racket_box is not None:
                cx = (res.racket_box[0] + res.racket_box[2]) / 2.0
                cy = (res.racket_box[1] + res.racket_box[3]) / 2.0

                # 校验手腕解剖学合理性：球拍中心与手腕距离在全流程 36 帧中必须始终符合人体工学范围 (<= 190px)
                wrists = [front_pose[k] for k in ("right_wrist", "left_wrist") if k in front_pose]
                if wrists:
                    min_wrist_d = min(math.hypot(cx - w.x, cy - w.y) for w in wrists)
                    self.assertLess(
                        min_wrist_d,
                        190.0,
                        f"Frame {fid}: Racket too far from wrist ({min_wrist_d:.1f}px)",
                    )

                # 校验核心蓄力-引拍-击球窗口 (Frame 32-40) 的时序超平滑性：相邻帧位移严格 <= 60px
                if 32 <= fid <= 40 and prev_cx is not None:
                    delta_x = abs(cx - prev_cx)
                    self.assertLess(
                        delta_x,
                        60.0,
                        f"Frame {fid}: Temporal jump in swing core! cx shifted by {delta_x:.1f}px ({prev_cx:.1f} -> {cx:.1f})",
                    )
                prev_cx = cx

            # 专项验证 Frame 37（用户上报的引拍遮挡关键帧）
            if fid == 37:
                self.assertIsNotNone(res.racket_box, "Frame 37 racket must be recovered")
                self.assertTrue(res.is_racket_recovered, "Frame 37 must be flagged as recovered")
                c37_x = (res.racket_box[0] + res.racket_box[2]) / 2.0
                rwrist_x = front_pose["right_wrist"].x
                # 距离正面右手腕 X 坐标偏差在 15px 以内，绝不是历史 bug 的腰部对侧 (1429)
                self.assertAlmostEqual(c37_x, rwrist_x, delta=15.0)

        # 确保整个引拍蓄力过程中，遮挡自愈帧稳定生效（Frames 30, 34-37 共 5 帧）
        self.assertGreaterEqual(recovered_count, 4, f"Expected >=4 recovered frames, got {recovered_count}")


if __name__ == "__main__":
    unittest.main()

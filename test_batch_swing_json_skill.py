"""Unit tests for the project-local batch-swing-json skill and script."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent / ".agents" / "skills" / "batch-swing-json"
SKILL_MD = SKILL_DIR / "SKILL.md"
SCRIPT_PATH = SKILL_DIR / "scripts" / "batch_swing_json.py"


def _load_skill_module():
    spec = importlib.util.spec_from_file_location("batch_swing_json", str(SCRIPT_PATH))
    module = importlib.util.module_from_spec(spec)
    sys.modules["batch_swing_json"] = module
    spec.loader.exec_module(module)
    return module


batch_mod = _load_skill_module()


class TestBatchSwingJsonSkill(unittest.TestCase):
    def test_skill_markdown_frontmatter_and_layout(self):
        self.assertTrue(SKILL_MD.is_file(), f"Missing {SKILL_MD}")
        self.assertTrue(SCRIPT_PATH.is_file(), f"Missing {SCRIPT_PATH}")
        text = SKILL_MD.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("---\n"))
        parts = text.split("---\n", 2)
        self.assertEqual(len(parts), 3)
        frontmatter = parts[1]
        self.assertIn("name: batch-swing-json", frontmatter)
        self.assertIn("description:", frontmatter)
        self.assertIn("batch_swing_json.py", parts[2])

    def test_discover_videos_recursive_non_recursive_and_hidden_filtering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            top_mp4 = root / "a.mp4"
            top_mov = root / "b.MOV"
            apple_double = root / "._a.mp4"
            hidden_dir_vid = root / ".hidden" / "secret.mp4"
            nested_mkv = root / "sub" / "nested.mkv"
            ignored_txt = root / "notes.txt"

            hidden_dir_vid.parent.mkdir(parents=True)
            nested_mkv.parent.mkdir(parents=True)
            for p in (top_mp4, top_mov, apple_double, hidden_dir_vid, nested_mkv, ignored_txt):
                p.write_bytes(b"x")

            rec = batch_mod.discover_videos(root, recursive=True)
            self.assertEqual(rec, sorted([Path(top_mp4), Path(top_mov), Path(nested_mkv)]))

            flat = batch_mod.discover_videos(root, recursive=False)
            self.assertEqual(flat, sorted([Path(top_mp4), Path(top_mov)]))

            single = batch_mod.discover_videos(nested_mkv)
            self.assertEqual(single, [Path(nested_mkv)])

    def test_build_main_pipe_command_matches_headless_offline_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            video = tmp_dir / "clip.mp4"
            cmd = batch_mod.build_main_pipe_command(
                video,
                tmp_dir,
                stream_id="court01-main",
                det_workers=2,
                shm_slots=8,
                max_frames=100,
            )
            self.assertIn("--algo2-dual-view", cmd)
            self.assertIn("--processing-mode", cmd)
            self.assertEqual(cmd[cmd.index("--processing-mode") + 1], "offline")
            self.assertIn("--headless", cmd)
            self.assertEqual(cmd[cmd.index("--racket-pose-mode") + 1], "legacy")
            self.assertEqual(cmd[cmd.index("--pose-model") + 1], "yolo26n-pose.mlpackage")
            self.assertIn("--analyze-swings", cmd)
            self.assertEqual(
                cmd[cmd.index("--swing-output-json") + 1],
                str(tmp_dir / "clip_swing_events.json"),
            )
            self.assertEqual(cmd[cmd.index("--stream-id") + 1], "court01-main")
            self.assertEqual(cmd[cmd.index("--det-workers") + 1], "2")
            self.assertEqual(cmd[cmd.index("--shm-slots") + 1], "8")
            self.assertEqual(cmd[cmd.index("--max-frames") + 1], "100")

    def test_process_video_outputs_only_swing_json_skips_and_handles_force_and_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "match.mp4"
            video.write_bytes(b"video")
            seen_cmds = []

            def fake_ok_runner(cmd, **kwargs):
                seen_cmds.append(cmd)
                swing_out = Path(cmd[cmd.index("--swing-output-json") + 1])
                mp4_out = Path(cmd[cmd.index("--output") + 1])
                # Simulate sidecars in temp dir that should NOT leak to `root`
                mp4_out.with_suffix(".json").write_text("{}", encoding="utf-8")
                swing_out.with_suffix(".csv").write_text("a,b\n", encoding="utf-8")
                swing_out.write_text(
                    json.dumps({"events": [{"event_id": 1}, {"event_id": 2}]}),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(cmd, 0)

            # Dry-run does not execute runner or create files
            res_dry = batch_mod.process_video(video, dry_run=True, runner=fake_ok_runner)
            self.assertEqual(res_dry.status, "dry_run")
            self.assertEqual(len(seen_cmds), 0)
            self.assertFalse((root / "match_swing_events.json").exists())

            # Normal run writes only match_swing_events.json
            res_ok = batch_mod.process_video(video, runner=fake_ok_runner)
            self.assertEqual(res_ok.status, "ok")
            self.assertEqual(res_ok.events_count, 2)
            self.assertEqual(sorted(p.name for p in root.iterdir()), ["match.mp4", "match_swing_events.json"])

            # Existing valid file is skipped when force=False
            res_skip = batch_mod.process_video(video, runner=fake_ok_runner)
            self.assertEqual(res_skip.status, "skipped")
            self.assertEqual(res_skip.events_count, 2)
            self.assertEqual(len(seen_cmds), 1)

            # force=True reruns
            res_force = batch_mod.process_video(video, force=True, runner=fake_ok_runner)
            self.assertEqual(res_force.status, "ok")
            self.assertEqual(len(seen_cmds), 2)

            # Failed runner preserves no partial file on fresh video
            bad_video = root / "bad.mp4"
            bad_video.write_bytes(b"video")

            def fake_fail_runner(cmd, **kwargs):
                return subprocess.CompletedProcess(cmd, 1, stdout="boom")

            res_fail = batch_mod.process_video(bad_video, quiet=True, runner=fake_fail_runner)
            self.assertEqual(res_fail.status, "failed")
            self.assertIn("boom", res_fail.error)
            self.assertFalse((root / "bad_swing_events.json").exists())


if __name__ == "__main__":
    unittest.main()

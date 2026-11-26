"""Dedicated stateless detection worker pool for multi-process ANE saturation."""

from __future__ import annotations

import multiprocessing as mp
import queue
import signal
import time
import traceback
from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def build_roi_manager(roi_profile: Any, config: Dict[str, Any], frame_size: Tuple[int, int], algo2_dual_view: bool):
    """Build ROIManager consistently across main inference and det worker processes."""
    if not getattr(roi_profile, "enabled", False):
        return None
    from roi_manager import ROIManager

    candidate = ROIManager(config)
    roi_manager = None
    if candidate.set_roi_points(roi_profile.points):
        if getattr(roi_profile, "has_mirror_view", False) and getattr(roi_profile, "mirror_polygon", None):
            mirror_px = roi_profile.get_mirror_polygon_pixels(frame_size)
            if mirror_px:
                candidate.add_exclusion_polygon(mirror_px, label="mirror_zone")
        for ex_poly in getattr(roi_profile, "exclusion_polygons", None) or ():
            candidate.add_exclusion_polygon(ex_poly, label="custom_exclusion")
        roi_manager = candidate

    if (
        roi_manager is not None
        and algo2_dual_view
        and config.get("roi_settings", {}).get("preserve_upper_airspace", True)
    ):
        roi_manager.configure_ball_airspace(frame_size)
    return roi_manager


def _det_worker_main(
    rank: int,
    cfg_dict: Dict[str, Any],
    roi_profile: Any,
    frame_size: Tuple[int, int],
    algo2_dual_view: bool,
    shm_names: List[str],
    retain_racket_candidates: bool,
    warmup: bool,
    opencv_threads: Optional[int],
    in_q: Any,
    out_q: Any,
    ready_q: Any,
) -> None:
    """Spawn-safe detection worker entrypoint reading source frames from shared memory."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if opencv_threads is not None:
        cv2.setNumThreads(max(0, int(opencv_threads)))

    shms: List[shared_memory.SharedMemory] = []
    try:
        from yolo26n_unified_detector import YOLO26nUnifiedDetector

        width, height = int(frame_size[0]), int(frame_size[1])
        roi_manager = build_roi_manager(roi_profile, cfg_dict, (width, height), algo2_dual_view)
        det_cfg = cfg_dict["unified_detection"]
        detector = YOLO26nUnifiedDetector(
            det_cfg["model_path"],
            det_cfg,
            roi_manager=roi_manager,
        )
        raw_detector = detector.candidate_inference_view()

        shms = [shared_memory.SharedMemory(name=name) for name in shm_names]
        shared_frames = [
            np.ndarray((height, width, 3), dtype=np.uint8, buffer=shm.buf)
            for shm in shms
        ]

        if warmup:
            from pipeline_startup import warm_coreml_wrappers

            warm_coreml_wrappers([detector])

        ready_q.put(("ready", rank))
    except BaseException:
        for shm in shms:
            try:
                shm.close()
            except Exception:
                pass
        ready_q.put(("error", rank, traceback.format_exc()))
        return

    try:
        while True:
            item = in_q.get()
            if item is None:
                break
            idx, slot = item
            started_unix_ns = time.time_ns()
            t0 = time.perf_counter()
            packet = raw_detector.external_candidate_packet(
                shared_frames[slot],
                idx,
                (width, height),
                (0, 0),
                retain_racket_candidates=retain_racket_candidates,
                racket_source="legacy",
                copy_packet=False,
            )
            detection_ms = (time.perf_counter() - t0) * 1000.0
            out_q.put(("ok", idx, packet, detection_ms, started_unix_ns))
    except BaseException:
        try:
            out_q.put(("error", -1, traceback.format_exc()))
        except Exception:
            pass
    finally:
        for shm in shms:
            try:
                shm.close()
            except Exception:
                pass


class DetWorkerPool:
    """Manages N stateless detection subprocesses and reorders results by frame_id."""

    def __init__(
        self,
        num_workers: int,
        cfg_dict: Dict[str, Any],
        roi_profile: Any,
        frame_size: Tuple[int, int],
        algo2_dual_view: bool,
        shm_names: List[str],
        retain_racket_candidates: bool,
        warmup: bool = True,
        opencv_threads: Optional[int] = None,
    ) -> None:
        if int(num_workers) < 1:
            raise ValueError("DetWorkerPool requires num_workers >= 1")
        self.num_workers = int(num_workers)
        self._ctx = mp.get_context("spawn")
        self._in_q = self._ctx.Queue()
        self._out_q = self._ctx.Queue()
        self._ready_q = self._ctx.Queue()
        self._ready: Dict[int, Tuple[Dict[str, Any], float, int]] = {}
        self._procs: List[Any] = []
        self._launched = False
        self._closed = False
        self._worker_args = (
            cfg_dict,
            roi_profile,
            (int(frame_size[0]), int(frame_size[1])),
            bool(algo2_dual_view),
            list(shm_names),
            bool(retain_racket_candidates),
            bool(warmup),
            opencv_threads,
        )

    def launch(self) -> None:
        """Spawn all detection worker processes without waiting for Core ML readiness."""
        if self._launched or self._closed:
            return
        self._launched = True
        (
            cfg_dict,
            roi_profile,
            frame_size,
            algo2_dual_view,
            shm_names,
            retain_racket_candidates,
            warmup,
            opencv_threads,
        ) = self._worker_args
        for rank in range(self.num_workers):
            proc = self._ctx.Process(
                target=_det_worker_main,
                args=(
                    rank,
                    cfg_dict,
                    roi_profile,
                    frame_size,
                    algo2_dual_view,
                    shm_names,
                    retain_racket_candidates,
                    warmup,
                    opencv_threads,
                    self._in_q,
                    self._out_q,
                    self._ready_q,
                ),
                name=f"tennis-det-{rank}",
            )
            proc.start()
            self._procs.append(proc)

    def wait_ready(self, timeout: float = 60.0) -> None:
        """Wait until all launched detection workers report Core ML readiness."""
        if not self._launched:
            self.launch()
        deadline = time.monotonic() + max(1.0, float(timeout))
        ready_count = 0
        while ready_count < self.num_workers:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.close()
                raise TimeoutError(f"DetWorkerPool timed out waiting for {self.num_workers} workers")
            try:
                msg = self._ready_q.get(timeout=min(0.25, remaining))
            except queue.Empty:
                dead = [p for p in self._procs if not p.is_alive()]
                if dead:
                    self.close()
                    raise RuntimeError(f"Detection worker exited unexpectedly during startup (code={dead[0].exitcode})")
                continue
            status = msg[0]
            if status == "error":
                _, rank, tb = msg
                self.close()
                raise RuntimeError(f"Detection worker #{rank} startup failed:\n{tb}")
            if status == "ready":
                ready_count += 1

    def start(self, timeout: float = 60.0) -> None:
        """Spawn all detection worker processes and wait until Core ML models are warmed up."""
        self.launch()
        self.wait_ready(timeout=timeout)

    def submit(self, idx: int, slot: int) -> None:
        """Enqueue a shared-memory slot for stateless detection."""
        self._in_q.put((int(idx), int(slot)))

    def get_result(self, idx: int, stop_event: Any = None) -> Tuple[Dict[str, Any], float, int]:
        """Retrieve detection result for exact frame_id `idx`, buffering out-of-order arrivals."""
        cached = self._ready.pop(idx, None)
        if cached is not None:
            return cached

        while True:
            if stop_event is not None and stop_event.is_set():
                raise RuntimeError("Detection wait aborted by stop_event")
            try:
                msg = self._out_q.get(timeout=0.1)
            except queue.Empty:
                dead = [p for p in self._procs if not p.is_alive()]
                if dead:
                    raise RuntimeError(f"Detection worker exited unexpectedly (code={dead[0].exitcode})")
                continue

            status = msg[0]
            if status == "error":
                _, err_idx, tb = msg
                raise RuntimeError(f"Detection worker failed on frame {err_idx}:\n{tb}")
            _, res_idx, packet, detection_ms, started_unix_ns = msg
            if res_idx == idx:
                return packet, detection_ms, started_unix_ns
            self._ready[res_idx] = (packet, detection_ms, started_unix_ns)

    def close(self) -> None:
        """Shut down all detection workers and close IPC queues."""
        if self._closed:
            return
        self._closed = True
        for proc in self._procs:
            if proc.is_alive():
                try:
                    self._in_q.put_nowait(None)
                except Exception:
                    pass
        deadline = time.monotonic() + 3.0
        for proc in self._procs:
            proc.join(timeout=max(0.0, deadline - time.monotonic()))
        for proc in self._procs:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)
        for q in (self._in_q, self._out_q, self._ready_q):
            try:
                q.close()
                q.join_thread()
            except Exception:
                pass

import argparse
import csv
import os
import time
from pathlib import Path
from statistics import mean, stdev

import sys
from pathlib import Path

# Add project root (parent of the MetroPaper package) to sys.path so imports like
# `from MetroPaper.config import ...` work when running the file directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import torch

from MetroPaper.config import TEST_VIDEO_PATH
from MetroPaper.models.od_model import ODModel, ODResult
from MetroPaper.models.ss_model import SegformerGuidance
from MetroPaper.models.ocr_model import OCRModel, OCRResult
from MetroPaper.core.fusion_state import NavigationFusion
from MetroPaper.utils.clock_sync import anchor_line_geometry


class NullSpeaker:
    def say(self, text: str, force: bool = False):
        pass

    def stop(self):
        pass


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def ms_since(start_time):
    return (time.perf_counter() - start_time) * 1000.0


def load_video_frames(video_path, max_frames=0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)

        if max_frames > 0 and len(frames) >= max_frames:
            break

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames loaded from: {video_path}")

    return frames


def summarize_times(name, input_size, times_ms):
    if not times_ms:
        return {
            "component": name,
            "input_size": input_size,
            "mean_latency_ms": 0.0,
            "std_latency_ms": 0.0,
            "fps_equivalent": 0.0,
            "num_frames": 0,
        }

    avg_ms = mean(times_ms)
    std_ms = stdev(times_ms) if len(times_ms) > 1 else 0.0
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    return {
        "component": name,
        "input_size": input_size,
        "mean_latency_ms": avg_ms,
        "std_latency_ms": std_ms,
        "fps_equivalent": fps,
        "num_frames": len(times_ms),
    }


def benchmark_od_only(frames, warmup, od_model):
    times = []

    H, W = frames[0].shape[:2]
    _, _, root_xy, _ = anchor_line_geometry(W, H)

    for i, frame in enumerate(frames):
        cuda_sync()
        t0 = time.perf_counter()
        _ = od_model.infer(frame, root_xy=root_xy)
        cuda_sync()

        if i >= warmup:
            times.append(ms_since(t0))

    return times


def benchmark_ocr_only(frames, warmup, ocr_model):
    times = []

    for i, frame in enumerate(frames):
        cuda_sync()
        t0 = time.perf_counter()
        _ = ocr_model.infer(frame)
        cuda_sync()

        if i >= warmup:
            times.append(ms_since(t0))

    return times


def benchmark_ss_only(frames, warmup, ss_model):
    times = []

    for i, frame in enumerate(frames):
        cuda_sync()
        t0 = time.perf_counter()

        # make_overlay=False is important for fair segmentation-module timing.
        # It avoids measuring visualization blending.
        _ = ss_model.update(
            frame,
            enabled=True,
            phase=None,
            make_overlay=False,
        )

        cuda_sync()

        if i >= warmup:
            times.append(ms_since(t0))

    return times


def benchmark_fusion_only(frames, warmup, od_model, ss_model, ocr_model):
    """
    Fusion-only timing needs valid OD/SS/OCR results.
    We first generate cached model outputs, then time only fusion.update().
    """

    cached_inputs = []

    fusion_for_cache = NavigationFusion(NullSpeaker())

    for frame in frames:
        ss_res = ss_model.update(
            frame,
            enabled=fusion_for_cache.ss_enabled,
            phase=fusion_for_cache.phase,
            make_overlay=False,
        )
        od_res = od_model.infer(frame, root_xy=ss_res.root_xy)
        ocr_res = ocr_model.infer(frame)

        cached_inputs.append((od_res, ss_res, ocr_res))

        # Update cache-fusion phase to keep representative states
        _ = fusion_for_cache.update(od_res, ss_res, ocr_res)

    fusion = NavigationFusion(NullSpeaker())
    times = []

    for i, (od_res, ss_res, ocr_res) in enumerate(cached_inputs):
        t0 = time.perf_counter()
        _ = fusion.update(od_res, ss_res, ocr_res)

        if i >= warmup:
            times.append(ms_since(t0))

    return times


def benchmark_full_pipeline(frames, warmup, od_model, ss_model, ocr_model, ocr_every):
    fusion = NavigationFusion(NullSpeaker())

    last_ocr_res = OCRResult(
        platforms=[],
        has_ticket_sign=False,
        has_gate=False,
        has_ben_thanh=False,
    )

    times = []
    od_times = []
    ss_times = []
    ocr_times = []
    fusion_times = []

    for i, frame in enumerate(frames):
        cuda_sync()
        frame_t0 = time.perf_counter()

        # SegFormer
        cuda_sync()
        ss_t0 = time.perf_counter()
        ss_res = ss_model.update(
            frame,
            enabled=fusion.ss_enabled,
            phase=fusion.phase,
            make_overlay=False,
        )
        cuda_sync()
        ss_ms = ms_since(ss_t0)

        # YOLO OD
        cuda_sync()
        od_t0 = time.perf_counter()
        od_res = od_model.infer(frame, root_xy=ss_res.root_xy)
        cuda_sync()
        od_ms = ms_since(od_t0)

        # OCR/sign YOLO, amortized by running every N frames
        run_ocr = (ocr_every <= 1) or (i % ocr_every == 0)
        ocr_ms = 0.0

        if run_ocr:
            cuda_sync()
            ocr_t0 = time.perf_counter()
            last_ocr_res = ocr_model.infer(frame)
            cuda_sync()
            ocr_ms = ms_since(ocr_t0)

        ocr_res = last_ocr_res

        # Fusion
        fusion_t0 = time.perf_counter()
        _ = fusion.update(od_res, ss_res, ocr_res)
        fusion_ms = ms_since(fusion_t0)

        cuda_sync()
        frame_ms = ms_since(frame_t0)

        if i >= warmup:
            times.append(frame_ms)
            od_times.append(od_ms)
            ss_times.append(ss_ms)
            ocr_times.append(ocr_ms)
            fusion_times.append(fusion_ms)

    return times, od_times, ss_times, ocr_times, fusion_times


def save_summary_csv(path, rows):
    fieldnames = [
        "component",
        "input_size",
        "mean_latency_ms",
        "std_latency_ms",
        "fps_equivalent",
        "num_frames",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_raw_csv(path, raw_rows):
    fieldnames = [
        "component",
        "frame_index",
        "latency_ms",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in raw_rows:
            writer.writerow(row)


def add_raw_rows(raw_rows, component, times):
    for i, t in enumerate(times):
        raw_rows.append({
            "component": component,
            "frame_index": i,
            "latency_ms": t,
        })


def print_summary_table(rows):
    print("\n========== Component Benchmark Summary ==========")
    print(f"{'Component':35s} {'Input':12s} {'Latency (ms)':>15s} {'FPS':>10s}")
    print("-" * 78)

    for r in rows:
        print(
            f"{r['component']:35s} "
            f"{r['input_size']:12s} "
            f"{r['mean_latency_ms']:15.2f} "
            f"{r['fps_equivalent']:10.2f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default=TEST_VIDEO_PATH)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--ocr-every", type=int, default=8)
    parser.add_argument("--out-dir", type=str, default="outputs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    summary_path = Path(args.out_dir) / "component_benchmark_summary.csv"
    raw_path = Path(args.out_dir) / "component_benchmark_raw.csv"

    print("[INFO] Loading video frames...")
    frames = load_video_frames(args.source, max_frames=args.max_frames)
    print(f"[INFO] Loaded {len(frames)} frames from: {args.source}")

    if len(frames) <= args.warmup:
        raise RuntimeError(
            f"Video has {len(frames)} frames, but warmup={args.warmup}. "
            f"Use a smaller warmup or a longer video."
        )

    print("[INFO] Loading models...")
    od_model = ODModel()
    ss_model = SegformerGuidance()
    ocr_model = OCRModel()

    summary_rows = []
    raw_rows = []

    print("[INFO] Benchmarking YOLOv11 OD inference only...")
    od_times = benchmark_od_only(frames, args.warmup, od_model)
    summary_rows.append(
        summarize_times("YOLOv11 object detection", "640x640", od_times)
    )
    add_raw_rows(raw_rows, "YOLOv11 object detection", od_times)

    print("[INFO] Benchmarking YOLOv11 OCR/sign detection only...")
    ocr_times = benchmark_ocr_only(frames, args.warmup, ocr_model)
    summary_rows.append(
        summarize_times("YOLOv11 OCR/sign detection", "640x640", ocr_times)
    )
    add_raw_rows(raw_rows, "YOLOv11 OCR/sign detection", ocr_times)

    print("[INFO] Benchmarking SegFormer-B0 segmentation module...")
    ss_times = benchmark_ss_only(frames, args.warmup, ss_model)
    summary_rows.append(
        summarize_times("SegFormer-B0 segmentation module", "512x512", ss_times)
    )
    add_raw_rows(raw_rows, "SegFormer-B0 segmentation module", ss_times)

    print("[INFO] Benchmarking fusion + guidance logic only...")
    fusion_times = benchmark_fusion_only(
        frames,
        args.warmup,
        od_model,
        ss_model,
        ocr_model,
    )
    fusion_summary = summarize_times(
        "Fusion + guidance logic",
        "-",
        fusion_times,
    )

    # Fusion FPS is not very meaningful, but we keep it in CSV.
    summary_rows.append(fusion_summary)
    add_raw_rows(raw_rows, "Fusion + guidance logic", fusion_times)

    print("[INFO] Benchmarking full online pipeline...")
    (
        full_times,
        full_od_times,
        full_ss_times,
        full_ocr_times,
        full_fusion_times,
    ) = benchmark_full_pipeline(
        frames,
        args.warmup,
        od_model,
        ss_model,
        ocr_model,
        ocr_every=args.ocr_every,
    )

    summary_rows.append(
        summarize_times("Full online pipeline", "mixed", full_times)
    )
    add_raw_rows(raw_rows, "Full online pipeline", full_times)

    save_summary_csv(summary_path, summary_rows)
    save_raw_csv(raw_path, raw_rows)

    print_summary_table(summary_rows)

    print("\n========== Notes ==========")
    print(f"Warm-up frames excluded: {args.warmup}")
    print(f"OCR interval in full pipeline: every {args.ocr_every} frame(s)")
    print("Display, video writing, and TTS are disabled.")
    print(f"Summary CSV: {summary_path}")
    print(f"Raw CSV:     {raw_path}")


if __name__ == "__main__":
    main()
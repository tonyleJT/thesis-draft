import argparse
import csv
import os
import time
from pathlib import Path

import cv2
import torch


import sys
from pathlib import Path

# Add project root (parent of the MetroPaper package) to sys.path so imports like
# `from MetroPaper.config import ...` work when running the file directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from MetroPaper.config import (
    TEST_VIDEO_PATH,
    OCR_EVERY_N_FRAMES,
    OD_EVERY_N_FRAMES,
    SS_EVERY_N_FRAMES,
)
from MetroPaper.models.od_model import ODModel, ODResult
from MetroPaper.models.ss_model import SegformerGuidance
from MetroPaper.models.ocr_model import OCRModel, OCRResult
from MetroPaper.core.fusion_state import NavigationFusion
from MetroPaper.utils.speaker import Speaker


OD_DRAW_CLASSES = {"stair node", "escalator entry node", "ticket booth"}


class NullSpeaker:
    def say(self, text: str, force: bool = False):
        pass

    def stop(self):
        pass


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def draw_arrow(vis, root_xy, target_xy, color=(255, 255, 255), arrow_len_px=140):
    xr, yr = root_xy
    xt, yt = target_xy
    dx = xt - xr
    dy = yt - yr
    length = (dx * dx + dy * dy) ** 0.5 + 1e-6
    scale = arrow_len_px / length
    end_x = int(xr + dx * scale)
    end_y = int(yr + dy * scale)
    cv2.arrowedLine(vis, (xr, yr), (end_x, end_y), color, 2, tipLength=0.2)


def draw_announcement_text(vis, text: str):
    if not text:
        return

    x, y = 20, 45
    cv2.putText(
        vis, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA
    )
    cv2.putText(
        vis, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA
    )


def draw_od_overlays(vis, od_res):
    for d in od_res.detections:
        if d.class_name not in OD_DRAW_CLASSES:
            continue

        x1, y1, x2, y2 = d.bbox
        color = (0, 255, 0)
        if d.class_name == "escalator entry node":
            color = (0, 0, 255)

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis, d.class_name, (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA
        )


def draw_ocr_overlays(vis, ocr_res):
    for p in ocr_res.platforms:
        px1, py1, px2, py2 = p.bbox
        cv2.rectangle(vis, (px1, py1), (px2, py2), (255, 255, 255), 2)
        cv2.putText(
            vis, "platform", (px1, max(0, py1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )

        for c in p.components:
            x1, y1, x2, y2 = c.bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
            label = f"{c.class_name} {c.conf:.2f}"
            cv2.putText(
                vis, label, (x1, max(0, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA
            )

        for pair in p.pairs:
            lx, ly = map(int, pair.left_center)
            rx, ry = map(int, pair.right_center)

            cv2.circle(vis, (lx, ly), 4, (255, 255, 255), -1)
            cv2.circle(vis, (rx, ry), 4, (255, 255, 255), -1)

            cv2.putText(
                vis, "L", (lx - 6, ly - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
            )
            cv2.putText(
                vis, "R", (rx - 6, ry - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
            )


def draw_full_overlay(frame, ss_res, od_res, ocr_res, decision):
    if ss_res.overlay is not None:
        overlay = ss_res.overlay.copy()
    else:
        overlay = frame.copy()

    draw_od_overlays(overlay, od_res)
    draw_ocr_overlays(overlay, ocr_res)

    cv2.line(
        overlay,
        ss_res.line_p1,
        ss_res.line_p2,
        (0, 255, 0),
        int(ss_res.line_thick_px)
    )
    cv2.circle(overlay, ss_res.root_xy, 6, (0, 255, 0), -1)

    if decision.target_pt is not None:
        cv2.circle(overlay, decision.target_pt, 6, (255, 255, 255), -1)
        draw_arrow(overlay, ss_res.root_xy, decision.target_pt)

    draw_announcement_text(overlay, decision.ui_text)

    return overlay


def build_video_writer(output_path, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default=TEST_VIDEO_PATH)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--save-samples", type=int, default=0,
                        help="Save one annotated frame every N frames. 0 disables.")
    parser.add_argument("--output", type=str, default="outputs/metro_guidance_output.mp4")
    parser.add_argument("--log-csv", type=str, default="outputs/runtime_log.csv")
    parser.add_argument("--ocr-every", type=int, default=OCR_EVERY_N_FRAMES)
    parser.add_argument("--od-every", type=int, default=OD_EVERY_N_FRAMES)
    parser.add_argument("--ss-every", type=int, default=SS_EVERY_N_FRAMES)
    args = parser.parse_args()

    visual_mode = args.display or args.save_video or args.save_samples > 0

    os.makedirs("outputs", exist_ok=True)
    sample_dir = Path("outputs/samples")
    if args.save_samples > 0:
        sample_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("Cannot open video:", args.source)
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        src_fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save_video:
        writer = build_video_writer(args.output, src_fps, width, height)

    speaker = NullSpeaker() if args.benchmark else Speaker()

    print("[INFO] Loading models...")
    od_model = ODModel()
    ss_model = SegformerGuidance()
    ocr_model = OCRModel()
    fusion = NavigationFusion(speaker)

    last_od_res = None
    last_ss_res = None
    last_ocr_res = OCRResult(
        platforms=[],
        has_ticket_sign=False,
        has_gate=False,
        has_ben_thanh=False,
    )

    frame_idx = 0
    processed_frames = 0

    timing_rows = []

    print("[INFO] Starting video inference...")

    total_start = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_start = time.perf_counter()

        # ---------------- SegFormer ----------------
        ss_start = time.perf_counter()
        run_ss = (last_ss_res is None) or (args.ss_every <= 1) or (frame_idx % args.ss_every == 0)

        if run_ss:
            last_ss_res = ss_model.update(
                frame,
                enabled=fusion.ss_enabled,
                phase=fusion.phase,
                make_overlay=visual_mode,
            )

        ss_res = last_ss_res
        cuda_sync()
        ss_ms = (time.perf_counter() - ss_start) * 1000.0

        # ---------------- YOLO OD ----------------
        od_start = time.perf_counter()
        run_od = (last_od_res is None) or (args.od_every <= 1) or (frame_idx % args.od_every == 0)

        if run_od:
            last_od_res = od_model.infer(frame, root_xy=ss_res.root_xy)

        od_res = last_od_res
        cuda_sync()
        od_ms = (time.perf_counter() - od_start) * 1000.0

        # ---------------- OCR YOLO ----------------
        ocr_start = time.perf_counter()
        run_ocr = (args.ocr_every <= 1) or (frame_idx % args.ocr_every == 0)

        if run_ocr:
            last_ocr_res = ocr_model.infer(frame)

        ocr_res = last_ocr_res
        cuda_sync()
        ocr_ms = (time.perf_counter() - ocr_start) * 1000.0

        # ---------------- Fusion ----------------
        fusion_start = time.perf_counter()
        decision = fusion.update(od_res, ss_res, ocr_res)
        fusion_ms = (time.perf_counter() - fusion_start) * 1000.0

        # ---------------- Visualization / output ----------------
        vis_ms = 0.0
        if visual_mode:
            vis_start = time.perf_counter()
            overlay = draw_full_overlay(frame, ss_res, od_res, ocr_res, decision)

            if args.save_video and writer is not None:
                writer.write(overlay)

            if args.save_samples > 0 and frame_idx % args.save_samples == 0:
                sample_path = sample_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(sample_path), overlay)

            if args.display:
                cv2.imshow("Metro Guidance System", overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            vis_ms = (time.perf_counter() - vis_start) * 1000.0

        frame_ms = (time.perf_counter() - frame_start) * 1000.0

        timing_rows.append({
            "frame": frame_idx,
            "run_ss": int(run_ss),
            "run_od": int(run_od),
            "run_ocr": int(run_ocr),
            "phase": fusion.phase,
            "ss_enabled": int(fusion.ss_enabled),
            "ss_ms": ss_ms,
            "od_ms": od_ms,
            "ocr_ms": ocr_ms,
            "fusion_ms": fusion_ms,
            "visual_ms": vis_ms,
            "frame_ms": frame_ms,
            "ui_text": decision.ui_text,
        })

        processed_frames += 1
        frame_idx += 1

    total_time = time.perf_counter() - total_start
    avg_fps = processed_frames / max(total_time, 1e-9)
    avg_ms = 1000.0 / max(avg_fps, 1e-9)

    cap.release()
    if writer is not None:
        writer.release()
    if args.display:
        cv2.destroyAllWindows()
    speaker.stop()

    # Save runtime log
    if timing_rows:
        with open(args.log_csv, "w", newline="", encoding="utf-8") as f:
            writer_csv = csv.DictWriter(f, fieldnames=list(timing_rows[0].keys()))
            writer_csv.writeheader()
            writer_csv.writerows(timing_rows)

    print("\n========== Runtime Summary ==========")
    print(f"Frames processed: {processed_frames}")
    print(f"Total time:       {total_time:.3f} s")
    print(f"Average latency:  {avg_ms:.2f} ms/frame")
    print(f"Average FPS:      {avg_fps:.2f}")
    print(f"CSV log saved to: {args.log_csv}")

    if args.save_video:
        print(f"Output video:     {args.output}")


if __name__ == "__main__":
    main()
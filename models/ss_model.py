# MetroPaper/models/ss_model.py

from dataclasses import dataclass
from typing import Optional, Tuple, List
import contextlib
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from MetroPaper.config import (
    SS_BACKBONE,
    SS_WEIGHT_PATH,
    DEVICE,
    SS_IMG_SIZE,
    MORPH_KERNEL_SIZE,
    MIN_SAFE_AREA_PX,
    W_FORWARD,
    W_SIDE,
    LOOKAHEAD_MAX_AHEAD_RATIO,
    LOOKAHEAD_MIN_AHEAD_RATIO,
    LOOKAHEAD_ROI_HALF_RATIO,
    LOOKAHEAD_MIN_PIXELS,
    PHASE_ENTRY,
    PHASE_AFTER_TICKET,
    PHASE_AFTER_ESCALATOR_2,
)

from MetroReport.utils.clock_sync import anchor_line_geometry

ID2LABEL = {0: "background", 1: "blindway", 2: "curb_ramp"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

COLOR_MAP = {
    0: (128, 128, 128),
    1: (0, 255, 255),
    2: (0, 0, 255),
}
COLOR_LUT = np.array([COLOR_MAP[i] for i in range(len(COLOR_MAP))], dtype=np.uint8)


@dataclass
class SSResult:
    overlay: Optional[np.ndarray]

    # shared root/feet line geometry (for synchronized drawing + OD clock)
    line_p1: Tuple[int, int]
    line_p2: Tuple[int, int]
    root_xy: Tuple[int, int]
    line_thick_px: int

    mode: str  # "SEARCHING" or "FOLLOWING"

    # candidates (fusion chooses what to use / display)
    safe_best_pt: Optional[Tuple[int, int]]          # SEARCHING best safe region centroid
    lookahead_pt: Optional[Tuple[int, int]]          # FOLLOWING blindway lookahead (smoothed)
    curb_standalone_pt: Optional[Tuple[int, int]]    # stand-alone curb ramp (stage1 & stage3)
    curb_rightedge_pt: Optional[Tuple[int, int]]     # stage2 right-edge curb ramp lock
    curb_leftturn_pt: Optional[Tuple[int, int]]      # stage4 left-turn curb

    # closeness triggers
    rightedge_curb_close: bool                       # stage2 "Turn left" trigger


class SegformerGuidance:
    # Anchor line parameters (must match clock_sync default)
    ANCHOR_LINE_LEN_RATIO   = 0.23
    ANCHOR_LINE_THICK_PX    = 10
    ANCHOR_LINE_Y_OFFSET_PX = 6

    # Mode switching based on ANY-touch (blindway only)
    ON_FRAMES_THRESH  = 2
    OFF_FRAMES_THRESH = 4

    # SEARCHING morphology: close then open
    MORPH_CLOSE_KERNEL_SIZE = max(3, int(MORPH_KERNEL_SIZE))
    MORPH_OPEN_KERNEL_SIZE  = max(3, int(MORPH_KERNEL_SIZE) - 2)
    MORPH_CLOSE_ITERS = 1
    MORPH_OPEN_ITERS  = 1

    # FOLLOWING lookahead window logic (advanced)
    LOOKAHEAD_TARGET_RATIO = 0.38
    LOOKAHEAD_WINDOW_RATIO = 0.10
    LOOKAHEAD_EMA_ALPHA    = 0.73

    # close trigger for stage2 curb ramp
    RIGHTEDGE_CLOSE_DIST_RATIO = 0.12  # distance < 0.12*H OR line overlaps curb

    def __init__(self):
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            SS_BACKBONE,
            num_labels=len(ID2LABEL),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )
        processor = SegformerImageProcessor.from_pretrained(SS_BACKBONE)
        processor.do_reduce_labels = False

        ckpt = torch.load(SS_WEIGHT_PATH, map_location=DEVICE, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=True)

        self.model.to(DEVICE)
        self.model.eval()
        self.model = self.model.to(memory_format=torch.channels_last)

        self.mean = torch.tensor(processor.image_mean, device=DEVICE).view(1, 3, 1, 1)
        self.std  = torch.tensor(processor.image_std,  device=DEVICE).view(1, 3, 1, 1)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.use_cuda = (str(DEVICE).startswith("cuda") or str(DEVICE) == "cuda") and torch.cuda.is_available()

        self.mode = "SEARCHING"
        self.on_counter = 0
        self.off_counter = 0

        # following smoothing
        self.last_x_look: Optional[int] = None

        self.k_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.MORPH_CLOSE_KERNEL_SIZE, self.MORPH_CLOSE_KERNEL_SIZE)
        )
        self.k_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.MORPH_OPEN_KERNEL_SIZE, self.MORPH_OPEN_KERNEL_SIZE)
        )
        self.curb_kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.curb_kernel7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def _preprocess_frame(self, frame_bgr: np.ndarray) -> torch.Tensor:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (SS_IMG_SIZE, SS_IMG_SIZE), interpolation=cv2.INTER_LINEAR)

        tensor = torch.from_numpy(frame_resized).to(DEVICE, non_blocking=True)
        tensor = tensor.permute(2, 0, 1).contiguous()
        tensor = tensor.unsqueeze(0).to(memory_format=torch.channels_last)
        tensor = tensor.float() / 255.0
        tensor = (tensor - self.mean) / self.std
        return tensor

    def _segment_frame(
            self,
            pixel_values: torch.Tensor,
            out_h: int,
            out_w: int,
            make_overlay: bool = True,
    ):
        autocast_ctx = torch.cuda.amp.autocast(dtype=torch.float16) if self.use_cuda else contextlib.nullcontext()
        with torch.inference_mode(), autocast_ctx:
            logits = self.model(pixel_values).logits
            up = F.interpolate(logits, size=(out_h, out_w), mode="bilinear", align_corners=False)
            pred = up.argmax(dim=1).to(torch.int16)

        pred_np = pred.squeeze(0).cpu().numpy()

        if make_overlay:
            mask_bgr = COLOR_LUT[pred_np]
        else:
            mask_bgr = None

        return mask_bgr, pred_np

    def _anchor_touch_blindway(self, pred: np.ndarray, W: int, H: int):
        p1, p2, root_xy, thick = anchor_line_geometry(
            W, H,
            len_ratio=self.ANCHOR_LINE_LEN_RATIO,
            thick_px=self.ANCHOR_LINE_THICK_PX,
            y_offset_px=self.ANCHOR_LINE_Y_OFFSET_PX
        )

        x1, y = p1
        x2, _ = p2

        half_t = max(1, int(thick) // 2)
        y0 = max(0, y - half_t)
        y1 = min(H, y + half_t + 1)

        line_roi = pred[y0:y1, x1:x2 + 1]

        touch_blind = bool((line_roi == 1).any())
        touch_curb = bool((line_roi == 2).any())

        return p1, p2, root_xy, thick, touch_blind, touch_curb

    def _find_safe_regions(self, pred: np.ndarray, min_area_px: int = MIN_SAFE_AREA_PX):
        safe_mask = (((pred == 1) | (pred == 2)).astype(np.uint8) * 255)
        safe_mask = cv2.morphologyEx(safe_mask, cv2.MORPH_CLOSE, self.k_close, iterations=self.MORPH_CLOSE_ITERS)
        safe_mask = cv2.morphologyEx(safe_mask, cv2.MORPH_OPEN,  self.k_open,  iterations=self.MORPH_OPEN_ITERS)

        contours, _ = cv2.findContours(safe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area_px:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            regions.append({"contour": cnt, "area": area, "cx": cx, "cy": cy})
        return regions

    def _choose_best_region(self, regions, W, H, x_root, y_root, w_forward=W_FORWARD, w_side=W_SIDE):
        best = None
        best_score = -1e9
        for r in regions:
            cx, cy = r["cx"], r["cy"]
            X = cx - x_root
            Z = y_root - cy
            if Z <= 0:
                continue

            closeness = 1.0 - (Z / float(H))
            closeness = float(np.clip(closeness, 0.0, 1.0))
            lateral = min(abs(X) / (W / 2.0), 1.0)

            score = w_forward * closeness - w_side * lateral
            if score > best_score:
                best_score = score
                best = (cx, cy)
        return best

    def _find_lookahead_point(self, pred, x_root, y_root, W, H):
        y_min = max(0, int(y_root - H * LOOKAHEAD_MAX_AHEAD_RATIO))
        y_max = max(0, int(y_root - H * LOOKAHEAD_MIN_AHEAD_RATIO))
        if y_max <= y_min:
            return None

        x_min = max(0, int(x_root - W * LOOKAHEAD_ROI_HALF_RATIO))
        x_max = min(W - 1, int(x_root + W * LOOKAHEAD_ROI_HALF_RATIO))

        y_target = int(y_root - H * self.LOOKAHEAD_TARGET_RATIO)
        y_target = int(np.clip(y_target, y_min, y_max))

        win = int(H * self.LOOKAHEAD_WINDOW_RATIO)
        y0 = max(y_min, y_target - win)
        y1 = min(y_max, y_target + win)

        best_y = None
        best_count = -1
        best_mean_x = None
        best_dist = 1e9

        for y in range(y0, y1 + 1):
            row = pred[y, x_min:x_max + 1]
            mask = (row == 1)
            count = int(mask.sum())
            if count < LOOKAHEAD_MIN_PIXELS:
                continue

            dist = abs(y - y_target)
            if (count > best_count) or (count == best_count and dist < best_dist):
                xs = np.where(mask)[0] + x_min
                best_mean_x = int(xs.mean())
                best_y = y
                best_count = count
                best_dist = dist

        if best_y is None:
            return None

        # EMA smoothing on x only
        if self.last_x_look is None:
            x_look = best_mean_x
        else:
            x_look = int(self.LOOKAHEAD_EMA_ALPHA * self.last_x_look + (1.0 - self.LOOKAHEAD_EMA_ALPHA) * best_mean_x)
        self.last_x_look = x_look
        return (x_look, best_y)

    def _find_curb_ramp_focus(self, pred, x_root, y_root, W, H):
        # stand-alone curb ahead-left
        y_min = max(0, int(y_root - H * 0.5))
        y_max = max(0, int(y_root - H * 0.15))
        if y_max <= y_min:
            return None

        x_min = max(0, int(x_root - W * 0.5))
        x_max = max(0, x_root - 1)
        if x_max <= x_min:
            return None

        roi = pred[y_min:y_max, x_min:x_max]
        curb_mask = (roi == 2).astype(np.uint8) * 255
        if curb_mask.sum() == 0:
            return None

        curb_mask = cv2.morphologyEx(curb_mask, cv2.MORPH_OPEN, self.curb_kernel5, iterations=1)
        contours, _ = cv2.findContours(curb_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best_cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best_cnt)
        if area < 800:
            return None

        M = cv2.moments(best_cnt)
        if M["m00"] == 0:
            return None

        cx = x_min + int(M["m10"] / M["m00"])
        cy = y_min + int(M["m01"] / M["m00"])
        return (cx, cy)

    def _find_left_turn_curb_focus(self, pred, x_root, y_root, W, H):
        # platform left-turn curb near feet-left
        y_min = max(0, int(y_root - H * 0.25))
        y_max = min(H - 1, int(y_root + H * 0.05))
        if y_max <= y_min:
            return None

        x_min = max(0, int(x_root - W * 0.5))
        x_max = max(0, x_root - 1)
        if x_max <= x_min:
            return None

        roi = pred[y_min:y_max, x_min:x_max]
        curb_mask = (roi == 2).astype(np.uint8) * 255
        if curb_mask.sum() == 0:
            return None

        curb_mask = cv2.morphologyEx(curb_mask, cv2.MORPH_OPEN, self.curb_kernel7, iterations=1)
        contours, _ = cv2.findContours(curb_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best_cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best_cnt)
        if area < 200:
            return None

        M = cv2.moments(best_cnt)
        if M["m00"] == 0:
            return None

        cx = x_min + int(M["m10"] / M["m00"])
        cy = y_min + int(M["m01"] / M["m00"])
        return (cx, cy)

    def _find_right_edge_curb_focus(self, pred, x_root, y_root, W, H):
        """
        Stage-2 special: lock to the curb ramp that is on the right-most region.
        We choose the curb component with the maximum centroid-x in the ROI.
        """
        y_min = max(0, int(y_root - H * 0.6))
        y_max = max(0, int(y_root - H * 0.05))
        if y_max <= y_min:
            return None

        x_min = int(W * 0.55)
        x_max = W - 1
        if x_max <= x_min:
            return None

        roi = pred[y_min:y_max, x_min:x_max]
        curb_mask = (roi == 2).astype(np.uint8) * 255
        if curb_mask.sum() == 0:
            return None

        curb_mask = cv2.morphologyEx(curb_mask, cv2.MORPH_OPEN, self.curb_kernel5, iterations=1)
        contours, _ = cv2.findContours(curb_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best_pt = None
        best_x = -1
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 150:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx_roi = int(M["m10"] / M["m00"])
            cy_roi = int(M["m01"] / M["m00"])
            cx = x_min + cx_roi
            cy = y_min + cy_roi
            if cx > best_x:
                best_x = cx
                best_pt = (cx, cy)
        return best_pt

    def update(self, frame_bgr: np.ndarray, enabled: bool, phase: Optional[int] = None, make_overlay: bool = True,) -> SSResult:


        H, W = frame_bgr.shape[:2]

        # Always provide synchronized root/line geometry for OD and drawing
        line_p1, line_p2, root_xy, thick = anchor_line_geometry(
            W, H,
            len_ratio=self.ANCHOR_LINE_LEN_RATIO,
            thick_px=self.ANCHOR_LINE_THICK_PX,
            y_offset_px=self.ANCHOR_LINE_Y_OFFSET_PX
        )
        x_root, y_root = root_xy

        # If disabled: return raw frame (no mask)
        if not enabled:
            overlay = frame_bgr if make_overlay else None
            return SSResult(
                overlay=overlay,
                line_p1=line_p1, line_p2=line_p2, root_xy=root_xy, line_thick_px=thick,
                mode=self.mode,
                safe_best_pt=None,
                lookahead_pt=None,
                curb_standalone_pt=None,
                curb_rightedge_pt=None,
                curb_leftturn_pt=None,
                rightedge_curb_close=False,
            )

        pixel_values = self._preprocess_frame(frame_bgr)
        mask_bgr, pred = self._segment_frame(pixel_values, H, W, make_overlay=make_overlay)

        # anchor touch flags
        _, _, _, _, touch_blind, touch_curb = self._anchor_touch_blindway(pred, W, H)

        # mode switching (based on blindway touch)
        if self.mode == "SEARCHING":
            if touch_blind:
                self.on_counter += 1
                self.off_counter = 0
            else:
                self.on_counter = 0
                self.off_counter = 0

            if self.on_counter >= self.ON_FRAMES_THRESH:
                self.mode = "FOLLOWING"
                self.on_counter = 0
                self.off_counter = 0
                self.last_x_look = None

        else:
            if not touch_blind:
                self.off_counter += 1
                self.on_counter = 0
            else:
                self.off_counter = 0
                self.on_counter = 0

            if self.off_counter >= self.OFF_FRAMES_THRESH:
                self.mode = "SEARCHING"
                self.off_counter = 0
                self.on_counter = 0

        # candidates
        safe_best_pt = None
        lookahead_pt = None

        curb_standalone_pt = None
        curb_rightedge_pt = None
        curb_leftturn_pt = None

        if self.mode == "SEARCHING":
            regions = self._find_safe_regions(pred)
            safe_best_pt = self._choose_best_region(regions, W, H, x_root, y_root)
        else:
            lookahead_pt = self._find_lookahead_point(pred, x_root, y_root, W, H)

        # phase-based curbs (always computed; fusion chooses)
        if phase in (PHASE_ENTRY, PHASE_AFTER_TICKET):
            curb_standalone_pt = self._find_curb_ramp_focus(pred, x_root, y_root, W, H)

        # stage2 right-edge curb always available (fusion decides when to use)
        curb_rightedge_pt = self._find_right_edge_curb_focus(pred, x_root, y_root, W, H)

        if phase == PHASE_AFTER_ESCALATOR_2:
            curb_leftturn_pt = self._find_left_turn_curb_focus(pred, x_root, y_root, W, H)

        # stage2 close trigger: distance OR line overlap with curb
        rightedge_close = False
        if curb_rightedge_pt is not None:
            dx = (curb_rightedge_pt[0] - x_root)
            dy = (curb_rightedge_pt[1] - y_root)
            dist = float(np.hypot(dx, dy))
            if dist < (self.RIGHTEDGE_CLOSE_DIST_RATIO * H):
                rightedge_close = True
            if touch_curb:
                rightedge_close = True

        # mask overlay only (NO text, NO arrow, NO points)
        if make_overlay and mask_bgr is not None:
            overlay = cv2.addWeighted(frame_bgr, 0.6, mask_bgr, 0.4, 0)
        else:
            overlay = None

        return SSResult(
            overlay=overlay,
            line_p1=line_p1, line_p2=line_p2, root_xy=root_xy, line_thick_px=thick,
            mode=self.mode,
            safe_best_pt=safe_best_pt,
            lookahead_pt=lookahead_pt,
            curb_standalone_pt=curb_standalone_pt,
            curb_rightedge_pt=curb_rightedge_pt,
            curb_leftturn_pt=curb_leftturn_pt,
            rightedge_curb_close=rightedge_close,
        )

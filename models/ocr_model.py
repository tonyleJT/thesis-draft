from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
from ultralytics import YOLO

from MetroPaper.config import (
    DEVICE,
    OCR_MODEL_PATH,
    OCR_CONF_BASE,
    OCR_CONF_SHOW,
    OCR_IOU_THRESH,
    OCR_IMGSZ,
)

# NEW: logic threshold (keep independent from visualization threshold)
OCR_CONF_LOGIC = max(0.01, min(OCR_CONF_BASE, OCR_CONF_SHOW))  # safe default


@dataclass
class OCRComponent:
    class_name: str
    conf: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[float, float]


@dataclass
class OCRPair:
    type: str
    left_label: str
    right_label: str
    left_center: Tuple[float, float]
    right_center: Tuple[float, float]


@dataclass
class OCRPlatform:
    bbox: Tuple[int, int, int, int]
    components: List[OCRComponent]
    pairs: List[OCRPair]


@dataclass
class OCRResult:
    platforms: List[OCRPlatform]
    has_ticket_sign: bool
    has_gate: bool
    has_ben_thanh: bool


class OCRModel:
    """
    YOLO sign detector with platform grouping and left/right pair logic.
    """
    PLATFORM_CLASS_NAME_CANDIDATES = ["platform-sign", "platform_sign", "platform sign"]

    # Require station/gate/no-entry to be roughly on the same text line
    PAIR_MAX_DY_RATIO = 0.25  # fraction of platform height

    def __init__(self):
        self.model = YOLO(OCR_MODEL_PATH)
        self.names = self.model.names

        try:
            self.model.fuse()
        except Exception:
            pass

        self.platform_cls_id = None
        for cid, cname in self.names.items():
            if cname in self.PLATFORM_CLASS_NAME_CANDIDATES:
                self.platform_cls_id = cid
                break

        print("[OCR] Platform-sign class id:", self.platform_cls_id)

    def _group_by_platform(self, boxes, frame_shape) -> List[OCRPlatform]:
        H, W = frame_shape[:2]

        # --- collect platform boxes (filter by logic conf) ---
        platform_boxes: List[Tuple[float, float, float, float]] = []
        for box in boxes:
            cls_id = int(box.cls[0])
            if self.platform_cls_id is None or cls_id != self.platform_cls_id:
                continue
            conf = float(box.conf[0])
            if conf < OCR_CONF_LOGIC:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            platform_boxes.append((x1, y1, x2, y2))

        platforms: List[OCRPlatform] = [
            OCRPlatform(bbox=tuple(map(int, pb)), components=[], pairs=[])
            for pb in platform_boxes
        ]

        def contains(pb, cx, cy) -> bool:
            x1, y1, x2, y2 = pb
            return (x1 <= cx <= x2) and (y1 <= cy <= y2)

        def nearest_platform_idx(cx, cy) -> Optional[int]:
            if not platform_boxes:
                return None
            best_i = None
            best_d2 = None
            for i, (x1, y1, x2, y2) in enumerate(platform_boxes):
                px = 0.5 * (x1 + x2)
                py = 0.5 * (y1 + y2)
                d2 = (cx - px) ** 2 + (cy - py) ** 2
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best_i = i
            return best_i

        # --- assign components ---
        for box in boxes:
            conf = float(box.conf[0])
            if conf < OCR_CONF_LOGIC:
                continue

            cls_id = int(box.cls[0])
            if self.platform_cls_id is not None and cls_id == self.platform_cls_id:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            cname = self.names[cls_id]

            comp = OCRComponent(
                class_name=cname,
                conf=conf,
                bbox=(x1, y1, x2, y2),
                center=(cx, cy),
            )

            if platform_boxes:
                pid = None
                for i, pb in enumerate(platform_boxes):
                    if contains(pb, cx, cy):
                        pid = i
                        break
                if pid is None:
                    pid = nearest_platform_idx(cx, cy)  # fallback
                if pid is not None:
                    platforms[pid].components.append(comp)
            else:
                # no platform-sign detected -> single virtual platform
                if not platforms:
                    platforms.append(OCRPlatform(
                        bbox=(0, 0, W - 1, H - 1),
                        components=[comp],
                        pairs=[]
                    ))
                else:
                    platforms[0].components.append(comp)

        # --- build pairs per platform ---
        for p in platforms:
            p.pairs = self._build_pairs_for_platform(p)

        return platforms

    def _build_pairs_for_platform(self, platform: OCRPlatform) -> List[OCRPair]:
        comps = platform.components
        if not comps:
            return []

        x1, y1, x2, y2 = platform.bbox
        ph = max(1, (y2 - y1))
        max_dy = self.PAIR_MAX_DY_RATIO * ph

        best_by_class: Dict[str, OCRComponent] = {}
        for c in comps:
            prev = best_by_class.get(c.class_name)
            if (prev is None) or (c.conf > prev.conf):
                best_by_class[c.class_name] = c

        pairs: List[OCRPair] = []

        def add_pair(a: str, b: str, type_name: str):
            if a not in best_by_class or b not in best_by_class:
                return
            ca = best_by_class[a]
            cb = best_by_class[b]
            cx_a, cy_a = ca.center
            cx_b, cy_b = cb.center

            # NEW: require roughly same line
            if abs(cy_a - cy_b) > max_dy:
                return

            if cx_a < cx_b:
                left, right = ca, cb
            else:
                left, right = cb, ca

            pairs.append(OCRPair(
                type=type_name,
                left_label=left.class_name,
                right_label=right.class_name,
                left_center=left.center,
                right_center=right.center,
            ))

        add_pair("no-entry", "gate", "no-entry_gate")
        add_pair("ben-thanh-station", "suoi-tien-station", "ben-thanh_suoi-tien")
        add_pair("ben-thanh-station", "no-service", "ben-thanh_no-service")

        return pairs

    def infer(self, frame_bgr: np.ndarray) -> OCRResult:
        results = self.model.predict(
            frame_bgr,
            imgsz=OCR_IMGSZ,
            device=DEVICE,
            conf=OCR_CONF_BASE,
            iou=OCR_IOU_THRESH,
            verbose=False,
        )[0]

        platforms = self._group_by_platform(results.boxes, frame_bgr.shape)

        has_ticket_sign = False
        has_gate = False
        has_ben_thanh = False

        for p in platforms:
            for c in p.components:
                if c.class_name == "ticket-sign":
                    has_ticket_sign = True
                elif c.class_name == "gate":
                    has_gate = True
                elif c.class_name == "ben-thanh-station":
                    has_ben_thanh = True

        return OCRResult(
            platforms=platforms,
            has_ticket_sign=has_ticket_sign,
            has_gate=has_gate,
            has_ben_thanh=has_ben_thanh,
        )

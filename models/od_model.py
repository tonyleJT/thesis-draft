# MetroPaper/models/od_model.py

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from ultralytics import YOLO

from MetroPaper.config import (
    DEVICE,
    OD_MODEL_PATH,
    OD_HIGH_CONF_CLASSES,
    OD_HIGH_CONF_THRESH,
    OD_DEFAULT_CONF_THRESH,
    OD_IMGSZ,
)
from MetroPaper.utils.clock_sync import points_to_clock, anchor_line_geometry


@dataclass
class ODDetection:
    class_name: str
    conf: float
    bbox: Tuple[int, int, int, int]    # (x1, y1, x2, y2)
    center: Tuple[int, int]           # (cx, cy)
    clock_dir: Optional[str]          # "3 o'clock", ...
    angle_deg: float                  # signed


@dataclass
class ODResult:
    detections: List[ODDetection]
    escalator_detections: List[ODDetection]


class ODModel:
    def __init__(self):
        self.model = YOLO(OD_MODEL_PATH)
        self.class_names = self.model.names

        try:
            self.model.fuse()
        except Exception:
            pass

        self.min_conf = min(OD_DEFAULT_CONF_THRESH, OD_HIGH_CONF_THRESH)

    @torch.inference_mode()
    def infer(self, frame_bgr: np.ndarray, root_xy: Optional[Tuple[int, int]] = None) -> ODResult:
        H, W = frame_bgr.shape[:2]

        if root_xy is None:
            _, _, root_xy, _ = anchor_line_geometry(W, H)

        res = self.model.predict(
            frame_bgr,
            imgsz=OD_IMGSZ,
            device=DEVICE,
            conf=self.min_conf,
            verbose=False,
        )[0]

        detections: List[ODDetection] = []
        escalator_dets: List[ODDetection] = []

        boxes = res.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()

            for box, conf, cls in zip(xyxy, confs, clss):
                cls_id = int(cls)
                class_name = self.class_names[cls_id]
                conf = float(conf)

                if class_name in OD_HIGH_CONF_CLASSES:
                    if conf < OD_HIGH_CONF_THRESH:
                        continue
                elif conf < OD_DEFAULT_CONF_THRESH:
                    continue

                x1, y1, x2, y2 = map(int, box.tolist())
                cx = int(0.5 * (x1 + x2))
                cy = int(0.5 * (y1 + y2))

                clock_str, angle_deg = points_to_clock(root_xy, (cx, cy))

                det = ODDetection(
                    class_name=class_name,
                    conf=conf,
                    bbox=(x1, y1, x2, y2),
                    center=(cx, cy),
                    clock_dir=clock_str,
                    angle_deg=angle_deg,
                )

                detections.append(det)

                if class_name == "escalator entry node":
                    escalator_dets.append(det)

        if escalator_dets:
            return ODResult(
                detections=escalator_dets,
                escalator_detections=escalator_dets,
            )

        return ODResult(
            detections=detections,
            escalator_detections=[],
        )

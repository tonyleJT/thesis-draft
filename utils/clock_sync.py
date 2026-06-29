# MetroPaper/utils/clock_sync.py
import math
from typing import Tuple, Optional

# This mapping matches SS angle_to_clock style:
# angle = atan2(X, Z)  (X right+, Z forward+)
# angle_deg in [-180, 180], but we mainly use [-90..90] for guidance.
def angle_deg_to_clock(angle_deg: float) -> str:
    if -10 <= angle_deg <= 10:
        h = "12"
    elif 10 < angle_deg <= 30:
        h = "1"
    elif 30 < angle_deg <= 60:
        h = "2"
    elif angle_deg > 60:
        h = "3"
    elif -30 <= angle_deg < -10:
        h = "11"
    elif -60 <= angle_deg < -30:
        h = "10"
    else:
        h = "9"
    return f"{h} o'clock"


def points_to_clock(root_xy: Tuple[int, int], target_xy: Tuple[int, int]) -> Tuple[Optional[str], float]:
    xr, yr = root_xy
    xt, yt = target_xy

    X = float(xt - xr)          # right positive
    Z = float(yr - yt)          # forward positive (upwards in image)
    if Z <= 0:
        return None, 0.0

    angle_deg = math.degrees(math.atan2(X, Z))
    return angle_deg_to_clock(angle_deg), angle_deg


def anchor_line_geometry(
    width: int,
    height: int,
    len_ratio: float = 0.23,
    thick_px: int = 10,
    y_offset_px: int = 6
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], int]:
    """
    Returns: (p1, p2, root_xy, thick_px)
    root_xy is the center point of the feet line.
    """
    x_root = width // 2
    y_root = max(0, min(height - 1, (height - 1) - int(y_offset_px)))

    half_len = int((width * len_ratio) / 2.0)
    x1 = max(0, x_root - half_len)
    x2 = min(width - 1, x_root + half_len)

    p1 = (x1, y_root)
    p2 = (x2, y_root)
    return p1, p2, (x_root, y_root), int(thick_px)

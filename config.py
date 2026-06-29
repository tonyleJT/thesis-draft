import torch

# ------------------- DEVICE CONFIG -------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------- PATH CONFIG ---------------------
# TODO: change to your actual paths
OD_MODEL_PATH   = r"T:\2. Graduation Thesis\MetroPaper\model\yolo11m.pt"
SS_BACKBONE     = "nvidia/segformer-b0-finetuned-ade-512-512"
SS_WEIGHT_PATH  = r"T:\2. Graduation Thesis\MetroPaper\model\segformer.pt"
OCR_MODEL_PATH  = r"T:\2. Graduation Thesis\MetroPaper\model\yoloOCR.pt"

TEST_VIDEO_PATH = r"T:\2. Graduation Thesis\MetroPaper\reportVid\fps\full720.mp4"

# ------------------- OD CONFIG -----------------------
# These names must match your trained YOLO11m model
OD_HIGH_CONF_CLASSES = {"ticket booth", "stair node", "pillar node", "escalator entry node"}
OD_HIGH_CONF_THRESH  = 0.8
OD_DEFAULT_CONF_THRESH = 0.6
OD_IMGSZ = 640  # <<< TUNE if you want faster/slower OD

# ------------------- SS CONFIG -----------------------
SS_IMG_SIZE = 512
# Run segmentation every N-th frame. 0 = every frame.
SEG_EVERY_N_FRAMES = 0   # <<< TUNE: 1 or 2 to boost FPS

# Foot window / state machine (same as your SS code)
FOOT_W_RATIO        = 0.7
FOOT_H_RATIO        = 0.06
FOOT_ON_THRESHOLD   = 0.40
FOOT_OFF_THRESHOLD  = 0.20
ON_FRAMES_THRESH    = 3
OFF_FRAMES_THRESH   = 5

MORPH_KERNEL_SIZE = 5
MIN_SAFE_AREA_PX  = 2000
W_FORWARD         = 1.0
W_SIDE            = 0.5

LOOKAHEAD_MAX_AHEAD_RATIO = 0.50
LOOKAHEAD_MIN_AHEAD_RATIO = 0.15
LOOKAHEAD_ROI_HALF_RATIO  = 0.25
LOOKAHEAD_MIN_PIXELS      = 30
ARROW_LEN_PX              = 120

# ------------------- OCR CONFIG ----------------------
OCR_CONF_BASE  = 0.5
OCR_CONF_SHOW  = 0.7
OCR_IOU_THRESH = 0.45

# ------------------- SPEECH / ANNOUNCEMENT CONFIG -------------------
SS_SEARCH_COOLDOWN_SEC = 7.0   # <<< TUNE: min gap between SS "Move ..." messages
GLOBAL_SPEECH_COOLDOWN_SEC = 1.0  # gap between any two spoken messages

# ------------------- STAGE / FLOW CONSTANTS -------------------------
# We just define phase IDs for clarity
PHASE_ENTRY             = 1
PHASE_AFTER_ESCALATOR_1 = 2
PHASE_AFTER_TICKET      = 3
PHASE_AFTER_ESCALATOR_2 = 4

# ------------------- RUNTIME OPTIMIZATION CONFIG -------------------
OCR_EVERY_N_FRAMES = 8       # run OCR every 8 frames, reuse last OCR result
OD_EVERY_N_FRAMES = 1        # keep OD every frame for escalator/stair/gate responsiveness
SS_EVERY_N_FRAMES = 1        # keep SS every frame first; later test 2 if needed

OCR_IMGSZ = 640              # can reduce to 512 for speed if accuracy remains acceptable





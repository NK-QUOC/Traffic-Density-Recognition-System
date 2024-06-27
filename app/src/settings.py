from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parent
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
ROOT = ROOT.relative_to(Path.cwd())

# Images config
IMAGES_DIR = ROOT / 'assets/images'
DEFAULT_IMAGE = IMAGES_DIR / 'traffic.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'traffic_detected.jpg'

# Videos config
VIDEO_DIR = ROOT / 'assets/videos'
VIDEOS_DICT = {
    'video_1': VIDEO_DIR / 'traffic_4.mp4',
    'video_2': VIDEO_DIR / 'DuongNguyenHue-Sang.mp4',
    'video_3': VIDEO_DIR / 'video-test-NguyenHue-Sang-cut.mp4',
    'video_4': VIDEO_DIR / 'Camera_View_Công_Trình_bệnh_viện_Đà_Nẵng_18_05_2024_Trưa.mp4'
}

# output video
OUTPUT_VIDEO_DIR = ROOT / 'outputs'

# ML Model config
MODEL_DIR = ROOT / 'weights'
YOLOv8_MODEL = MODEL_DIR / 'yolov8-pretrained.pt'
YOLOv9_MODEL = MODEL_DIR / 'yolov9-pretrained.pt'
# YOLOv9_accident_MODEL = MODEL_DIR / 'yolov9-accident-50epoch.pt'
# YOLOV9_tensort_MODEL = MODEL_DIR / 'yolov9-tdrs-for-convert-int8-win.engine'
# YOLOV8_tensort_MODEL = MODEL_DIR / 'yolov8-tdrs-for-convert-win.engine'

# Tracker config
TRACKER_DIR = ROOT / 'cfg' / 'trackers'
bytetrack = TRACKER_DIR / 'bytetrack.yaml'
botsort = TRACKER_DIR / 'botsort.yaml'

# Webcam
WEBCAM_PATH = 0

DATA_CSV_PATH = ROOT / 'data'

# Database
POSTGRES_SERVER='localhost'
POSTGRES_PORT=5432
POSTGRES_USER='postgres'
POSTGRES_PASSWORD='quoc3010'
POSTGRES_DB='TDRS'

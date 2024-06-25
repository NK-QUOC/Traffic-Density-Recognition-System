from .dashboard import Dashboard
from .multimedia_detection import MultimediaDetection
from .multi_camera_monitor import MultiCameraMonitor
from ..utils import Page

from typing import Dict, Type


PAGE_MAP: Dict[str, Type[Page]] = {
    "Dashboard": Dashboard,
    "Multimedia Detection": MultimediaDetection,
    "Multi Camera Monitor": MultiCameraMonitor,
}

__all__ = ["PAGE_MAP"]

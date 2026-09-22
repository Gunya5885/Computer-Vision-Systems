import cv2
import numpy as np
from picamera2 import Picamera2

# Try to import autofocus controls for Pi Camera Module 3.
try:
    from libcamera import controls
    AUTOFOCUS_AVAILABLE = True
except ImportError:
    AUTOFOCUS_AVAILABLE = False


# -------------------------------------------------
# SETTINGS
# -------------------------------------------------
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MIN_AREA = 700

KERNEL = np.ones((5, 5), np.uint8)

CAMERA_WINDOW = "Pi Camera 3 - Multi Color Detection"
SELECTOR_WINDOW = "Color Selector"

COLOR_NAMES = [
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple"
]

# OpenCV color values use BGR channel order.
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (0, 255, 255)


# -------------------------------------------------
# HSV COLOR RANGES
# OpenCV hue values range from 0 to 179.
# -------------------------------------------------
color_ranges = {
    "red": [
        (np.array([0, 100, 80]), np.array([10, 255, 255])),
        (np.array([170, 100, 80]), np.array([179, 255, 255]))
    ],
    "orange": [
        (np.array([10, 100, 80]), np.array([24, 255, 255]))
    ],
    "yellow": [
        (np.array([25, 100, 80]), np.array([35, 255, 255]))
    ],
    "green": [
        (np.array([36, 70, 50]), np.array([85, 255, 255]))
    ],
    "blue": [
        (np.array([86, 70, 50]), np.array([130, 255, 255]))
    ],
    "purple": [
        (np.array([131, 60, 50]), np.array([169, 255, 255]))
    ]
}


# -------------------------------------------------
# SELECTOR WINDOW FUNCTIONS
# -------------------------------------------------
def nothing(value):
    pass


def make_selector_image(selected_color, selected_index):
    image = np.zeros((180, 560, 3), dtype=np.uint8)

    cv2.putText(
        image,
        "Choose target color with slider",
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        WHITE,
        2
    )

    cv2.putText(
        image,
        f"Target color: {selected_color.upper()}",
        (20, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        GREEN,
        2
    )

    cv2.putText(
        image,
        f"Slider number: {selected_index}",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        YELLOW,
        2
    )

    return image


# --------------------------------------------

import cv2
import csv
import numpy as np
from datetime import datetime
from pathlib import Path
from picamera2 import Picamera2


FRAME_WIDTH = 320
FRAME_HEIGHT = 240
MIN_AREA = 250

MIN_ASPECT_RATIO = 0.45
MAX_ASPECT_RATIO = 2.20
MIN_SOLIDITY = 0.78

KERNEL = np.ones((3, 3), np.uint8)

CAMERA_WINDOW = "Color Detector"
SELECTOR_WINDOW = "Choose Target Color"

DATA_FOLDER = Path("/home/lhsengr10a/Computer-Vision-Systems/data")
LOG_FILE = DATA_FOLDER / "color_detection_log.csv"

COLOR_NAMES = [
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple"
]

GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (0, 255, 255)


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
        (np.array([36, 60, 35]), np.array([85, 255, 255]))
    ],
    "blue": [
        (np.array([86, 70, 50]), np.array([130, 255, 255]))
    ],
    "purple": [
        (np.array([131, 40, 25]), np.array([169, 255, 255]))
    ]
}


def nothing(value):
    pass


def selector_image(selected_color):
    image = np.zeros((115, 420, 3), dtype=np.uint8)

    cv2.putText(
        image,
        "Move slider to choose target",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        WHITE,
        1
    )

    cv2.putText(
        image,
        f"TARGET: {selected_color.upper()}",
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        GREEN,
        2
    )

    return image


def create_log_file_if_needed():
    DATA_FOLDER.mkdir(parents=True, exist_ok=True)

    if not LOG_FILE.exists():
        with open(LOG_FILE, "w", newline="") as file:
            writer = csv.writer(file)

            writer.writerow([
                "timestamp",
                "selected_target",
                "result",
                "target_count",
                "wrong_colors"
            ])


def log_detection(selected_color, terminal_status,
                  target_objects, wrong_color_list):

    timestamp = datetime.now().isoformat(
        sep=" ",
        timespec="seconds"
    )

    wrong_colors_text = ", ".join(wrong_color_list)

    with open(LOG_FILE, "a", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            timestamp,
            selected_color,
            terminal_status,
            target_objects,
            wrong_colors_text
        ])


create_log_file_if_needed()

print(f"Detection log file: {LOG_FILE}")

cv2.namedWindow(SELECTOR_WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(SELECTOR_WINDOW, 420, 160)

cv2.createTrackbar(
    "Color: 0R 1O 2Y 3G 4B 5P",
    SELECTOR_WINDOW,
    0,
    len(COLOR_NAMES) - 1,
    nothing
)


picam2 = Picamera2()

camera_config = picam2.create_preview_configuration(
    main={
        "size": (FRAME_WIDTH, FRAME_HEIGHT),
        "format": "RGB888"
    }
)

picam2.configure(camera_config)
picam2.start()

print("Safe color detector started.")
print("Green box = selected correct color.")
print("Red box = other detected color.")
print("Press Q to quit.")

last_terminal_status = None
frame_number = 0
last_result_frame = None


try:
    while True:
        frame = picam2.capture_array()

        selected_index = cv2.getTrackbarPos(
            "Color: 0R 1O 2Y 3G 4B 5P",
            SELECTOR_WINDOW
        )

        selected_color = COLOR_NAMES[selected_index]

        cv2.imshow(
            SELECTOR_WINDOW,
            selector_image(selected_color)
        )

        frame_number += 1

        if frame_number % 2 == 0:
            display_frame = frame.copy()

            blurred = cv2.GaussianBlur(
                display_frame,
                (3, 3),
                0
            )

            hsv = cv2.cvtColor(
                blurred,
                cv2.COLOR_BGR2HSV
            )

            detected_counts = {
                color_name: 0
                for color_name in COLOR_NAMES
            }

            target_objects = 0
            other_objects = 0

            for color_name, ranges in color_ranges.items():
                mask = np.zeros(
                    hsv.shape[:2],
                    dtype=np.uint8
                )

                for lower, upper in ranges:
                    mask |= cv2.inRange(hsv, lower, upper)

                mask = cv2.morphologyEx(
                    mask,
                    cv2.MORPH_OPEN,
                    KERNEL
                )

                contours, _ = cv2.findContours(
                    mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                for contour in contours:
                    area = cv2.contourArea(contour)

                    if area < MIN_AREA:
                        continue

                    x, y, w, h = cv2.boundingRect(contour)

                    if h == 0:
                        continue

                    aspect_ratio = w / float(h)

                    if aspect_ratio < MIN_ASPECT_RATIO:
                        continue

                    if aspect_ratio > MAX_ASPECT_RATIO:
                        continue

                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)

                    if hull_area == 0:
                        continue

                    solidity = area / float(hull_area)

                    if solidity < MIN_SOLIDITY:
                        continue

                    detected_counts[color_name] += 1

                    is_target = (color_name == selected_color)

                    if is_target:
                        box_color = GREEN
                        object_status = "CORRECT"
                        target_objects += 1
                    else:
                        box_color = RED
                        object_status = "WRONG"
                        other_objects += 1

                    moments = cv2.moments(contour)

                    if moments["m00"] != 0:
                        center_x = int(
                            moments["m10"] / moments["m00"]
                        )
                        center_y = int(
                            moments["m01"] / moments["m00"]
                        )
                    else:
                        center_x = x + w // 2
                        center_y = y + h // 2

                    cv2.rectangle(
                        display_frame,
                        (x, y),
                        (x + w, y + h),
                        box_color,
                        2
                    )

                    cv2.circle(
                        display_frame,
                        (center_x, center_y),
                        3,
                        box_color,
                        -1
                    )

                    label = f"{color_name.upper()} {object_status}"

                    cv2.putText(
                        display_frame,
                        label,
                        (x, max(y - 7, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.40,
                        box_color,
                        1
                    )

            wrong_color_list = []

            for color_name in COLOR_NAMES:
                if color_name != selected_color:
                    count = detected_counts[color_name]

                    if count > 0:
                        wrong_color_list.append(
                            f"{color_name} ({count})"
                        )

            if target_objects > 0:
                terminal_status = (
                    f"CORRECT COLOR DETECTED: "
                    f"{selected_color.upper()} "
                    f"({target_objects})"
                )

                if wrong_color_list:
                    terminal_status += (
                        " | Wrong color(s): "
                        + ", ".join(wrong_color_list)
                    )

            elif wrong_color_list:
                terminal_status = (
                    "WRONG COLOR DETECTED: "
                    + ", ".join(wrong_color_list)
                    + f" | Waiting for {selected_color.upper()}"
                )

            else:
                terminal_status = None

            if terminal_status != last_terminal_status:
                if terminal_status is not None:
                    print(terminal_status)

                    log_detection(
                        selected_color,
                        terminal_status,
                        target_objects,
                        wrong_color_list
                    )

                last_terminal_status = terminal_status

            cv2.rectangle(
                display_frame,
                (0, 0),
                (FRAME_WIDTH, 45),
                BLACK,
                -1
            )

            cv2.putText(
                display_frame,
                f"Target: {selected_color.upper()}",
                (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                GREEN,
                1
            )

            if target_objects > 0:
                screen_status = "CORRECT DETECTED"
                screen_color = GREEN

            elif other_objects > 0:
                screen_status = "WRONG COLOR"
                screen_color = RED

            else:
                screen_status = "WAITING"
                screen_color = YELLOW

            cv2.putText(
                display_frame,
                screen_status,
                (8, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                screen_color,
                1
            )

            last_result_frame = display_frame

        if last_result_frame is not None:
            cv2.imshow(CAMERA_WINDOW, last_result_frame)
        else:
            cv2.imshow(CAMERA_WINDOW, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


except KeyboardInterrupt:
    print("\nStopped by user.")


finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("Camera stopped safely.")

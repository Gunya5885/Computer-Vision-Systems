import cv2
import numpy as np
from picamera2 import Picamera2

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={
        "size": (640, 480),
        "format": "BGR888"
    }
)

picam2.configure(config)
picam2.start()

# HSV ranges: OpenCV hue is 0-179
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

kernel = np.ones((5, 5), np.uint8)

try:
    while True:
        frame = picam2.capture_array()

        # Reduce small color noise
        blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_RBG2HSV)

        best_color = "unknown"
        best_mask = None
        best_area = 0
        best_box = None

        for color_name, ranges in color_ranges.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

            for lower, upper in ranges:
                mask |= cv2.inRange(hsv, lower, upper)

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(contour)

                if area > best_area:
                    best_area = area
                    best_color = color_name
                    best_mask = mask
                    best_box = cv2.boundingRect(contour)

        if best_area > 1200 and best_box is not None:
            x, y, w, h = best_box

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            label = f"{best_color}: {int(best_area)} px"
            cv2.putText(
                frame,
                label,
                (x, max(y - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            print(f"\rDetected: {best_color:8s} area={int(best_area):6d}",
                  end="", flush=True)
        else:
            cv2.putText(
                frame,
                "No target color",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            print("\rDetected: unknown ", end="", flush=True)

        cv2.imshow("Color detector", frame)

        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()

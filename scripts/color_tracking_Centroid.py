import cv2
import numpy as np
from picamera2 import Picamera2
from libcamera import controls   # Module 3 autofocus enums

# Replace these with YOUR tuned values from Part 2
# (read off the on-screen text overlay while tuning the trackbars)
LOWER = np.array([0, 156, 100])
UPPER = np.array([177, 255, 255])

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": 'XRGB8888', "size": (640, 480)}
))
picam2.start()

# Module 3 has motorised autofocus — keep the lens in focus while tracking.
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})

try:
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, LOWER, UPPER)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 300:  # ignore tiny noise blobs
                (x, y), radius = cv2.minEnclosingCircle(largest)
                M = cv2.moments(largest)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"({cx},{cy})", (cx + 10, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow('Color Tracking', frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    picam2.stop()
    cv2.destroyAllWindows()

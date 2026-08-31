import cv2
from picamera2 import Picamera2
from libcamera import controls

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": 'XRGB8888', "size": (640, 480)}
))
picam2.start()

# --- Camera Module 3 specific: enable continuous autofocus ---
# v2 has a fixed-focus lens, so this call would fail/do nothing on v2.
try:
    picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
except Exception as e:
    print(f"Autofocus not available (not a v3 module?): {e}")
# ---------------------------------------------------------------

def classify_shape(contour):
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    vertices = len(approx)
    if vertices == 3:
        return "Triangle"
    elif vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        return "Square" if 0.9 <= aspect_ratio <= 1.1 else "Rectangle"
    elif vertices == 5:
        return "Pentagon"
    elif vertices > 8:
        return "Circle"
    else:
        return f"{vertices}-gon"

try:
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            shape_name = classify_shape(contour)
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            cv2.putText(frame, shape_name, (cx - 40, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow('Shape Detection', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Interrupted by user")
finally:
    picam2.stop()
    cv2.destroyAllWindows()

import cv2
import numpy as np
from picamera2 import Picamera2

def nothing(x):
    pass

cv2.namedWindow('Trackbars')
cv2.createTrackbar('H Min', 'Trackbars', 0, 179, nothing)
cv2.createTrackbar('H Max', 'Trackbars', 179, 179, nothing)
cv2.createTrackbar('S Min', 'Trackbars', 100, 255, nothing)
cv2.createTrackbar('S Max', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('V Min', 'Trackbars', 100, 255, nothing)
cv2.createTrackbar('V Max', 'Trackbars', 255, 255, nothing)

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": 'XRGB8888', "size": (640, 480)}
))
picam2.start()

try:
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        h_min = cv2.getTrackbarPos('H Min', 'Trackbars')
        h_max = cv2.getTrackbarPos('H Max', 'Trackbars')
        s_min = cv2.getTrackbarPos('S Min', 'Trackbars')
        s_max = cv2.getTrackbarPos('S Max', 'Trackbars')
        v_min = cv2.getTrackbarPos('V Min', 'Trackbars')
        v_max = cv2.getTrackbarPos('V Max', 'Trackbars')
        print(f"H: {h_min}-{h_max}  S: {s_min}-{s_max}  V: {v_min}-{v_max}", end='\r')

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # NOTE: OpenCV trackbar sliders do NOT display their numeric value
        # anywhere in the window -- only the slider position is shown visually.
        # This overlay prints the live values directly on the video feed so
        # you can read off your tuned HSV range once the Mask looks clean.
        readout = f"H:{h_min}-{h_max} S:{s_min}-{s_max} V:{v_min}-{v_max}"
        cv2.putText(frame, readout, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Original', frame)
        cv2.imshow('Mask', mask)
        cv2.imshow('Result', result)

        # NOTE: waitKey delay increased from 1ms to 20ms.
        # With 4 windows open (Trackbars, Original, Mask, Result), the extra
        # render overhead per loop made a 1ms window too easy to miss,
        # causing 'q' presses to not register reliably. 20ms fixes this
        # without introducing any noticeable lag in the live feed.
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    picam2.stop()
    cv2.destroyAllWindows()

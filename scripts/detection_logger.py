import cv2
import csv
import numpy as np
from datetime import datetime
from picamera2 import Picamera2
from tflite_runtime.interpreter import Interpreter

MODEL_PATH = "models/model.tflite"
LABELS_PATH = "models/labels.txt"
LOG_PATH = "data/detections.csv"
CONFIDENCE_THRESHOLD = 0.5
TARGET_LABELS = None  # e.g., {"person", "bird"} to filter, or None for all

with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_height = input_details[0]['shape'][1]
input_width = input_details[0]['shape'][2]

with open(LOG_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "label", "confidence"])

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "XRGB8888", "size": (640, 480)}
))
picam2.start()

try:
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        h, w, _ = frame.shape

        resized = cv2.resize(frame, (input_width, input_height))
        input_data = np.expand_dims(resized, axis=0)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]

        for i in range(len(scores)):
            if scores[i] > CONFIDENCE_THRESHOLD:
                label = labels[int(classes[i])]
                if TARGET_LABELS and label not in TARGET_LABELS:
                    continue

                ymin, xmin, ymax, xmax = boxes[i]
                x1, y1, x2, y2 = int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {scores[i]:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                with open(LOG_PATH, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([datetime.now().isoformat(), label, round(float(scores[i]), 2)])

        cv2.imshow("Detection Logger", frame)
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    picam2.stop()
    cv2.destroyAllWindows()

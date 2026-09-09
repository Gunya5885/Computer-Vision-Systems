import cv2
import mediapipe as mp
from picamera2 import Picamera2

# Set up MediaPipe Pose and its drawing helper
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Set up the Raspberry Pi camera
picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"format": "XRGB8888", "size": (640, 480)}
    )
)
picam2.start()

try:
    while True:
        # Capture a frame from Picamera2
        frame = picam2.capture_array()

        # Picamera2 gives a 4-channel image; convert BGRA to normal BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # MediaPipe expects RGB, not OpenCV's BGR format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run pose landmark detection
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            # Draw the 33 landmarks and skeleton connections
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            # Store all detected body landmarks
            landmarks = results.pose_landmarks.landmark

            # Get the right wrist and right shoulder landmarks
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

            # Print normalized coordinates to the terminal
            print(
                f"Right wrist: x={right_wrist.x:.2f}, y={right_wrist.y:.2f} | "
                f"Right shoulder: x={right_shoulder.x:.2f}, y={right_shoulder.y:.2f}",
                end="\r"
            )

            # Convert normalized landmark coordinates into pixel coordinates
            frame_height, frame_width, _ = frame.shape

            wrist_x = int(right_wrist.x * frame_width)
            wrist_y = int(right_wrist.y * frame_height)

            shoulder_x = int(right_shoulder.x * frame_width)
            shoulder_y = int(right_shoulder.y * frame_height)

            # Mark the tracked points in red
            cv2.circle(frame, (wrist_x, wrist_y), 8, (0, 0, 255), -1)
            cv2.circle(frame, (shoulder_x, shoulder_y), 8, (0, 0, 255), -1)

            # Label the points on the camera feed
            cv2.putText(
                frame,
                "Right Wrist",
                (wrist_x + 10, wrist_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                "Right Shoulder",
                (shoulder_x + 10, shoulder_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )

            # Display the wrist y-coordinate on-screen
            cv2.putText(
                frame,
                f"Wrist y: {right_wrist.y:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        else:
            cv2.putText(
                frame,
                "No pose detected - step farther back",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        cv2.imshow("Day 9 - Pose Estimation", frame)

        # Click the video window and press q to quit
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    pose.close()
    picam2.stop()
    cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import math
from picamera2 import Picamera2


# ============================================================
# MediaPipe setup
# ============================================================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# Calculate angle between three landmarks
# ============================================================

def calculate_angle(a, b, c):
    """
    Calculates the angle at point b.

    For a bicep curl:
        a = shoulder
        b = elbow
        c = wrist
    """

    angle = math.degrees(
        math.atan2(c.y - b.y, c.x - b.x)
        - math.atan2(a.y - b.y, a.x - b.x)
    )

    angle = abs(angle)

    if angle > 180:
        angle = 360 - angle

    return angle


# ============================================================
# Camera setup
# ============================================================

picam2 = Picamera2()

camera_config = picam2.create_preview_configuration(
    main={
        "format": "XRGB8888",
        "size": (640, 480)
    }
)

picam2.configure(camera_config)
picam2.start()


# ============================================================
# Rep-counter settings
# ============================================================

rep_count = 0
stage = "Start"

# Adjust these if necessary
EXTENDED_ANGLE = 160
CURLED_ANGLE = 50

# Minimum visibility required for the landmarks
MIN_VISIBILITY = 0.5


# ============================================================
# Main camera loop
# ============================================================

try:
    while True:
        # Capture an image
        frame = picam2.capture_array()

        # Convert Picamera2 BGRA/XRGB frame to OpenCV BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Convert BGR to RGB because MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Improve performance by marking the image as not writeable
        rgb_frame.flags.writeable = False

        # Detect pose landmarks
        results = pose.process(rgb_frame)

        # Allow the frame to be modified again
        rgb_frame.flags.writeable = True

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get the left-arm landmarks
            left_shoulder = landmarks[
                mp_pose.PoseLandmark.LEFT_SHOULDER.value
            ]

            left_elbow = landmarks[
                mp_pose.PoseLandmark.LEFT_ELBOW.value
            ]

            left_wrist = landmarks[
                mp_pose.PoseLandmark.LEFT_WRIST.value
            ]

            # Check whether the required landmarks are visible
            landmarks_visible = (
                left_shoulder.visibility >= MIN_VISIBILITY
                and left_elbow.visibility >= MIN_VISIBILITY
                and left_wrist.visibility >= MIN_VISIBILITY
            )

            if landmarks_visible:
                # Calculate the angle at the left elbow
                elbow_angle = calculate_angle(
                    left_shoulder,
                    left_elbow,
                    left_wrist
                )

                # Rep-counting state machine
                if elbow_angle > EXTENDED_ANGLE:
                    stage = "Down"

                elif elbow_angle < CURLED_ANGLE and stage == "Down":
                    stage = "Up"
                    rep_count += 1
                    print(f"Rep counted! Total reps: {rep_count}")

                # Draw the body skeleton
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                # Convert normalized coordinates to pixel coordinates
                frame_height, frame_width, _ = frame.shape

                shoulder_x = int(left_shoulder.x * frame_width)
                shoulder_y = int(left_shoulder.y * frame_height)

                elbow_x = int(left_elbow.x * frame_width)
                elbow_y = int(left_elbow.y * frame_height)

                wrist_x = int(left_wrist.x * frame_width)
                wrist_y = int(left_wrist.y * frame_height)

                # Draw markers on the three landmarks
                cv2.circle(
                    frame,
                    (shoulder_x, shoulder_y),
                    8,
                    (255, 0, 0),
                    -1
                )

                cv2.circle(
                    frame,
                    (elbow_x, elbow_y),
                    8,
                    (0, 0, 255),
                    -1
                )

                cv2.circle(
                    frame,
                    (wrist_x, wrist_y),
                    8,
                    (0, 255, 0),
                    -1
                )

                # Display the angle beside the elbow
                cv2.putText(
                    frame,
                    f"{int(elbow_angle)} deg",
                    (elbow_x + 10, elbow_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2
                )

                # Display information panel
                cv2.rectangle(
                    frame,
                    (0, 0),
                    (260, 125),
                    (0, 0, 0),
                    -1
                )

                cv2.putText(
                    frame,
                    f"Angle: {int(elbow_angle)} deg",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Reps: {rep_count}",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Stage: {stage}",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2
                )

            else:
                cv2.putText(
                    frame,
                    "Move your left arm into view",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2
                )

        else:
            cv2.putText(
                frame,
                "No pose detected",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2
            )

        # Display the camera feed
        cv2.imshow("Day 10 - Bicep Curl Counter", frame)

        # Press q to quit
        key = cv2.waitKey(20) & 0xFF

        if key == ord("q"):
            break


except KeyboardInterrupt:
    print("\nProgram interrupted by user.")


finally:
    # Clean up camera and windows
    pose.close()
    picam2.stop()
    cv2.destroyAllWindows()

    print(f"\nFinal rep count: {rep_count}")

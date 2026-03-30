import argparse
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")


TIP_IDS = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}

PIP_IDS = {
    "thumb": 3,
    "index": 6,
    "middle": 10,
    "ring": 14,
    "pinky": 18,
}


@dataclass
class GestureResult:
    name: str
    confidence: float
    fingers_up: Dict[str, bool]


class HandGestureRecognizer:
    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.6,
    ) -> None:
        self.ensure_model_exists(MODEL_PATH)
        self.mp_draw = mp.tasks.vision.drawing_utils
        self.mp_styles = mp.tasks.vision.drawing_styles
        self.hand_connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.hands = mp.tasks.vision.HandLandmarker.create_from_options(options)

    @staticmethod
    def ensure_model_exists(model_path: str) -> None:
        if os.path.exists(model_path):
            return

        print("Downloading MediaPipe hand landmark model...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print(f"Model saved to: {model_path}")

    def process_frame(self, frame, timestamp_ms: int):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self.hands.detect_for_video(mp_image, timestamp_ms)

    def draw_landmarks(self, frame, hand_landmarks) -> None:
        self.mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            self.hand_connections,
            self.mp_styles.get_default_hand_landmarks_style(),
            self.mp_styles.get_default_hand_connections_style(),
        )

    def detect_fingers_up(self, hand_landmarks, hand_label: str, mirrored: bool = False) -> Dict[str, bool]:
        landmarks = hand_landmarks
        fingers_up = {}

        effective_hand_label = hand_label
        if mirrored:
            effective_hand_label = "Left" if hand_label == "Right" else "Right"

        if effective_hand_label == "Right":
            fingers_up["thumb"] = landmarks[TIP_IDS["thumb"]].x < landmarks[PIP_IDS["thumb"]].x
        else:
            fingers_up["thumb"] = landmarks[TIP_IDS["thumb"]].x > landmarks[PIP_IDS["thumb"]].x

        for finger in ("index", "middle", "ring", "pinky"):
            fingers_up[finger] = landmarks[TIP_IDS[finger]].y < landmarks[PIP_IDS[finger]].y

        return fingers_up

    def classify_gesture(self, fingers_up: Dict[str, bool]) -> GestureResult:
        total_up = sum(fingers_up.values())

        patterns: List[Tuple[str, Dict[str, bool], float]] = [
            ("Fist", {"thumb": False, "index": False, "middle": False, "ring": False, "pinky": False}, 0.98),
            ("Open Palm", {"thumb": True, "index": True, "middle": True, "ring": True, "pinky": True}, 0.98),
            ("Thumbs Up", {"thumb": True, "index": False, "middle": False, "ring": False, "pinky": False}, 0.95),
            ("Peace", {"thumb": False, "index": True, "middle": True, "ring": False, "pinky": False}, 0.96),
            ("Rock", {"thumb": False, "index": True, "middle": False, "ring": False, "pinky": True}, 0.92),
            ("Okay", {"thumb": False, "index": False, "middle": True, "ring": True, "pinky": True}, 0.86),
            ("Pointing Up", {"thumb": False, "index": True, "middle": False, "ring": False, "pinky": False}, 0.94),
        ]

        for name, pattern, confidence in patterns:
            if fingers_up == pattern:
                return GestureResult(name=name, confidence=confidence, fingers_up=fingers_up)

        fallback_name = f"{total_up} finger(s)"
        fallback_confidence = 0.55 if total_up else 0.5
        return GestureResult(name=fallback_name, confidence=fallback_confidence, fingers_up=fingers_up)

    def classify_from_landmarks(self, hand_landmarks, hand_label: str, mirrored: bool = False) -> GestureResult:
        fingers_up = self.detect_fingers_up(hand_landmarks, hand_label, mirrored=mirrored)

        thumb_tip = hand_landmarks[TIP_IDS["thumb"]]
        index_tip = hand_landmarks[TIP_IDS["index"]]
        wrist = hand_landmarks[0]
        middle_mcp = hand_landmarks[9]

        pinch_distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
        palm_scale = ((wrist.x - middle_mcp.x) ** 2 + (wrist.y - middle_mcp.y) ** 2) ** 0.5

        if palm_scale and pinch_distance / palm_scale < 0.35:
            if fingers_up["middle"] and fingers_up["ring"] and fingers_up["pinky"]:
                return GestureResult(name="Okay", confidence=0.93, fingers_up=fingers_up)

        return self.classify_gesture(fingers_up)

    def annotate_frame(
        self,
        frame,
        gesture: GestureResult,
        handedness_label: str,
        bbox: Tuple[int, int, int, int],
    ) -> None:
        x_min, y_min, x_max, y_max = bbox

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (40, 180, 90), 2)
        cv2.rectangle(frame, (x_min, max(0, y_min - 60)), (x_max, y_min), (40, 180, 90), -1)
        cv2.putText(
            frame,
            f"{gesture.name} ({gesture.confidence:.2f})",
            (x_min + 8, max(20, y_min - 34)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Hand: {handedness_label}",
            (x_min + 8, max(40, y_min - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def get_bounding_box(hand_landmarks, frame_shape) -> Tuple[int, int, int, int]:
        height, width, _ = frame_shape
        x_points = [int(lm.x * width) for lm in hand_landmarks]
        y_points = [int(lm.y * height) for lm in hand_landmarks]

        padding = 20
        x_min = max(0, min(x_points) - padding)
        y_min = max(0, min(y_points) - padding)
        x_max = min(width, max(x_points) + padding)
        y_max = min(height, max(y_points) + padding)
        return x_min, y_min, x_max, y_max

    def close(self) -> None:
        self.hands.close()


def open_capture(source: str):
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Custom hand gesture recognition using MediaPipe hand landmarks and OpenCV."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index or path to a video file. Default: 0",
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        default=1,
        help="Maximum number of hands to track. Default: 1",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror the output horizontally for a selfie-camera style preview.",
    )
    args = parser.parse_args()

    capture = open_capture(args.source)
    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video source '{args.source}'. Make sure your webcam is available."
        )

    recognizer = HandGestureRecognizer(max_num_hands=args.max_hands)

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            if args.mirror:
                frame = cv2.flip(frame, 1)

            timestamp_ms = int(time.time() * 1000)
            results = recognizer.process_frame(frame, timestamp_ms)

            if results.hand_landmarks and results.handedness:
                for hand_landmarks, handedness in zip(
                    results.hand_landmarks,
                    results.handedness,
                ):
                    hand_label = handedness[0].category_name
                    gesture = recognizer.classify_from_landmarks(
                        hand_landmarks,
                        hand_label,
                        mirrored=args.mirror,
                    )
                    bbox = recognizer.get_bounding_box(hand_landmarks, frame.shape)

                    recognizer.draw_landmarks(frame, hand_landmarks)
                    recognizer.annotate_frame(frame, gesture, hand_label, bbox)

            cv2.putText(
                frame,
                "Press 'q' to quit",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (20, 20, 20),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Custom Hand Gesture Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        recognizer.close()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

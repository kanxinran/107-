"""
MediaPipe pose extraction for fall prediction system.

Extracts 33 keypoints (x, y, z) from each video frame.
"""

import cv2
import numpy as np
import mediapipe as mp


class PoseExtractor:
    def __init__(self, static_mode=False, model_complexity=1,
                 min_detection_confidence=0.3, min_tracking_confidence=0.3):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def extract_raw(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Extract 33 keypoints (x, y, z) from a BGR frame.

        Returns: (99,) flattened float32 array, zero-filled if no detection.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)

        if result.pose_landmarks:
            return np.array(
                [[lm.x, lm.y, lm.z] for lm in result.pose_landmarks.landmark],
                dtype=np.float32
            ).flatten()

        return np.zeros(99, dtype=np.float32)

    def extract_structured(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Extract 33 keypoints as (33, 3) array.
        Returns: (33, 3) float32, zero-filled if no detection.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)

        if result.pose_landmarks:
            return np.array(
                [[lm.x, lm.y, lm.z] for lm in result.pose_landmarks.landmark],
                dtype=np.float32
            )

        return np.zeros((33, 3), dtype=np.float32)

    def close(self):
        self.pose.close()

"""
2-class fall detection visualizer with anatomical keypoints.
"""

import cv2
import numpy as np

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
    (24, 26), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32),
    (0, 11), (0, 12),
]

COLORS = {
    0: (0, 255, 0),
    1: (0, 0, 255),
}

STATUS_BG = {
    0: (0, 100, 0),
    1: (0, 0, 150),
}

def draw_pose_skeleton(frame_bgr: np.ndarray, keypoints_33x3: np.ndarray,
                       anat_points: dict = None, thickness: int = 2) -> np.ndarray:
    """Draw MediaPipe pose skeleton + anatomical keypoints on frame."""
    h, w = frame_bgr.shape[:2]

    for conn in POSE_CONNECTIONS:
        p1 = keypoints_33x3[conn[0]]
        p2 = keypoints_33x3[conn[1]]
        if np.all(p1 == 0) or np.all(p2 == 0):
            continue
        pt1 = (int(p1[0] * w), int(p1[1] * h))
        pt2 = (int(p2[0] * w), int(p2[1] * h))
        cv2.line(frame_bgr, pt1, pt2, (0, 255, 128), thickness)

    for i in range(33):
        kp = keypoints_33x3[i]
        if np.all(kp == 0):
            continue
        pt = (int(kp[0] * w), int(kp[1] * h))
        cv2.circle(frame_bgr, pt, 3, (0, 128, 255), -1)

    # Draw anatomical keypoints — same style as default dots, just different color
    if anat_points:
        for name in ["shoulder_center", "body_center"]:
            if name in anat_points:
                pt = anat_points[name]
                cv2.circle(frame_bgr, pt, 3, (0, 200, 255), -1)

    return frame_bgr


def draw_status_overlay(frame_bgr: np.ndarray, prediction: dict,
                        history: list = None) -> np.ndarray:
    """Draw 2-class status bar (Normal green / Fall red)."""
    h, w = frame_bgr.shape[:2]
    class_id = prediction["class_id"]

    bar_height = 45
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_height), STATUS_BG[class_id], -1)
    cv2.addWeighted(overlay, 0.5, frame_bgr, 0.5, 0, frame_bgr)

    status_text = f"[{prediction['class_name']}]  conf={prediction['confidence']:.2f}"
    cv2.putText(frame_bgr, status_text, (15, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    probs = prediction["probabilities"]
    names = ["Normal", "Fall"]
    bar_x = 15
    bar_w = (w - 40) // 2
    for i, (name, prob) in enumerate(zip(names, probs)):
        bx = bar_x + i * (bar_w + 5)
        cv2.rectangle(frame_bgr, (bx, 22), (bx + bar_w, 38), (60, 60, 60), -1)
        fill_w = int(bar_w * prob)
        bar_color = COLORS[1] if class_id == 1 else COLORS[i]
        cv2.rectangle(frame_bgr, (bx, 22), (bx + fill_w, 38), bar_color, -1)
        cv2.putText(frame_bgr, f"{name} {prob:.2f}", (bx + 3, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    if class_id == 1:
        alert_text = "ALERT: FALL DETECTED!"
        (tw, th), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cx = (w - tw) // 2
        cy = h // 2
        cv2.rectangle(frame_bgr, (cx - 20, cy - 40),
                      (cx + tw + 20, cy + 20), (0, 0, 0), -1)
        cv2.rectangle(frame_bgr, (cx - 20, cy - 40),
                      (cx + tw + 20, cy + 20), (0, 0, 255), 1)
        cv2.putText(frame_bgr, alert_text, (cx, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    if history:
        timeline_h = 30
        timeline_y = h - timeline_h
        n_show = min(len(history), 30)
        hist_show = history[-n_show:]
        seg_w = w / n_show
        for i, h_cls in enumerate(hist_show):
            sx = int(i * seg_w)
            cv2.rectangle(frame_bgr, (sx, timeline_y),
                          (int(sx + seg_w), h),
                          COLORS.get(h_cls, (100, 100, 100)), -1)

    return frame_bgr

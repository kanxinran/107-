"""
Handcrafted feature computation for fall prediction.

21 features organized in 6 groups (A-F) computed per frame.
Adapted from new_features.py — validated via ablation study.
"""

import numpy as np

# MediaPipe keypoint indices
KP = {
    'nose': 0, 'left_shoulder': 11, 'right_shoulder': 12,
    'left_hip': 23, 'right_hip': 24,
    'left_knee': 25, 'right_knee': 26,
    'left_ankle': 27, 'right_ankle': 28,
}

FEATURE_NAMES = [
    # A: Body Geometry (5)
    "A1:躯干长度", "A2:包围盒宽高比", "A3:有效高度", "A4:平均腿长", "A5:躯干占比",
    # B: Posture Angles (4)
    "B1:躯干倾斜角", "B2:肩线倾斜角", "B3:左膝角", "B4:右膝角",
    # C: Velocity (4)
    "C1:身体速率", "C2:头部垂直速率", "C3:重心垂直速率", "C4:髋部下降速率",
    # D: Acceleration (2)
    "D1:身体加速度", "D2:垂直加速度",
    # E: Stability (4)
    "E1:髋高不对称", "E2:肩高不对称", "E3:站姿宽度", "E4:躯干晃动",
    # F: Morphological Change (2)
    "F1:宽高比变化", "F2:关键点离散度",
]


def _safe_div(a, b, eps=1e-8):
    return a / (np.abs(b) + eps)


def _angle_between(v1, v2):
    dot = np.sum(v1 * v2, axis=-1)
    norm = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-8
    cos = np.clip(dot / norm, -1.0, 1.0)
    return np.arccos(cos)


def compute_features(kps_current: np.ndarray, kps_previous: np.ndarray = None) -> np.ndarray:
    """
    Compute 21 handcrafted features from a single frame's (33, 3) keypoints.

    Args:
        kps_current:  (33, 3) keypoints for current frame
        kps_previous: (33, 3) keypoints for previous frame, or None

    Returns: (21,) float32 feature vector
    """
    feats = []

    # ── Common keypoints ──
    nose = kps_current[KP['nose']]
    l_sho = kps_current[KP['left_shoulder']]
    r_sho = kps_current[KP['right_shoulder']]
    l_hip = kps_current[KP['left_hip']]
    r_hip = kps_current[KP['right_hip']]
    l_knee = kps_current[KP['left_knee']]
    r_knee = kps_current[KP['right_knee']]
    l_ankle = kps_current[KP['left_ankle']]
    r_ankle = kps_current[KP['right_ankle']]

    hip_center = (l_hip + r_hip) * 0.5
    shoulder_center = (l_sho + r_sho) * 0.5
    body_center = (hip_center + shoulder_center) * 0.5
    ankle_mid = (l_ankle + r_ankle) * 0.5

    # ── A: Body Geometry (5) ──
    torso_vec = shoulder_center - hip_center
    A1_torso_len = np.linalg.norm(torso_vec)

    all_x = kps_current[:, 0]
    all_y = kps_current[:, 1]
    bbox_w = all_x.max() - all_x.min() + 1e-8
    bbox_h = all_y.max() - all_y.min()
    A2_aspect = bbox_h / bbox_w

    A3_eff_h = np.linalg.norm(nose - ankle_mid)

    left_leg = np.linalg.norm(l_hip - l_ankle)
    right_leg = np.linalg.norm(r_hip - r_ankle)
    A4_leg_len = (left_leg + right_leg) * 0.5

    A5_trunk_ratio = _safe_div(A1_torso_len, A3_eff_h)

    feats.extend([A1_torso_len, A2_aspect, A3_eff_h, A4_leg_len, A5_trunk_ratio])

    # ── B: Posture Angles (4) ──
    dx_torso = np.abs(shoulder_center[0] - hip_center[0])
    dy_torso = np.abs(shoulder_center[1] - hip_center[1])
    B1_torso_tilt = np.arctan2(dx_torso, dy_torso + 1e-8)

    dx_shoulder = r_sho[0] - l_sho[0]
    dy_shoulder = r_sho[1] - l_sho[1]
    B2_shoulder_obl = np.abs(np.arctan2(dy_shoulder, dx_shoulder + 1e-8))

    B3_l_knee_angle = _angle_between(l_hip - l_knee, l_ankle - l_knee)
    B4_r_knee_angle = _angle_between(r_hip - r_knee, r_ankle - r_knee)

    feats.extend([B1_torso_tilt, B2_shoulder_obl, B3_l_knee_angle, B4_r_knee_angle])

    # ── C: Velocity (4) ──
    if kps_previous is not None:
        prev_l_hip = kps_previous[KP['left_hip']]
        prev_r_hip = kps_previous[KP['right_hip']]
        prev_hip_ctr = (prev_l_hip + prev_r_hip) * 0.5
        prev_nose = kps_previous[KP['nose']]
        prev_l_sho = kps_previous[KP['left_shoulder']]
        prev_r_sho = kps_previous[KP['right_shoulder']]
        prev_sho_ctr = (prev_l_sho + prev_r_sho) * 0.5
        prev_body_ctr = (prev_hip_ctr + prev_sho_ctr) * 0.5

        C1_body_speed = np.linalg.norm(body_center - prev_body_ctr)
        C2_head_vy = np.abs(nose[1] - prev_nose[1])
        C3_body_vy = np.abs(body_center[1] - prev_body_ctr[1])
        C4_hip_down = np.maximum(0.0, prev_hip_ctr[1] - hip_center[1])
    else:
        C1_body_speed = C2_head_vy = C3_body_vy = C4_hip_down = 0.0

    feats.extend([C1_body_speed, C2_head_vy, C3_body_vy, C4_hip_down])

    # ── D: Acceleration (2) ──
    feats.extend([C1_body_speed * 0.2, C3_body_vy * 0.2])  # proxy

    # ── E: Stability (4) ──
    E1_hip_asym = np.abs(l_hip[1] - r_hip[1])
    E2_sho_asym = np.abs(l_sho[1] - r_sho[1])
    E3_stance = np.linalg.norm(l_ankle - r_ankle)
    E4_wobble = np.abs(hip_center[0] - body_center[0])

    feats.extend([E1_hip_asym, E2_sho_asym, E3_stance, E4_wobble])

    # ── F: Morphological Change (2) ──
    F1_ar_change = 0.0
    if kps_previous is not None:
        prev_x = kps_previous[:, 0]
        prev_y = kps_previous[:, 1]
        prev_ar = (prev_y.max() - prev_y.min()) / (prev_x.max() - prev_x.min() + 1e-8)
        F1_ar_change = np.abs(A2_aspect - prev_ar)

    kp_xy = kps_current[:, :2].flatten()
    F2_dispersion = np.std(kp_xy)

    feats.extend([F1_ar_change, F2_dispersion])

    return np.array(feats, dtype=np.float32)


def compute_window_features(kps_window: np.ndarray) -> np.ndarray:
    """
    Compute 21 features for each frame in a (T, 33, 3) window.

    Returns: (T, 120) = 99 raw keypoints + 21 features
    """
    T = kps_window.shape[0]
    raw_flat = kps_window.reshape(T, 99).astype(np.float32)
    feat_frames = []

    for t in range(T):
        prev_kp = kps_window[t - 1] if t > 0 else None
        feats = compute_features(kps_window[t], prev_kp)
        feat_frames.append(feats)

    feat_window = np.stack(feat_frames, axis=0)
    return np.concatenate([raw_flat, feat_window], axis=1)

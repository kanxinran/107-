"""
10 new anatomical keypoints/features for fall detection.
Each returns per-frame values appended to input vector.

Points (x,y = 2 dims each):
  P1: hip_center        (左胯23 + 右胯24)/2
  P2: shoulder_center   (左肩11 + 右肩12)/2
  P3: body_center       (shoulder_center + hip_center)/2
  P4: X_intersection    左胯→右肩 × 右胯→左肩 2D交点
  P5: ankle_center      (左踝27 + 右踝28)/2
  P6: knee_center       (左膝25 + 右膝26)/2

Vectors (x,y = 2 dims each):
  V1: head_hip_vec      nose(0) → hip_center

Scalars (1 dim each):
  S1: diagonal_ratio    ‖左肩-右胯‖ / ‖右肩-左胯‖
  S2: head_hip_angle    头-髋向量 vs 垂直轴 夹角(rad)
  S3: trunk_aspect      ‖肩中-髋中‖ / ‖左肩-右肩‖

Total: 6*2 + 1*2 + 3 = 17 dims per frame
"""

import numpy as np

KP = {
    'nose': 0, 'left_shoulder': 11, 'right_shoulder': 12,
    'left_hip': 23, 'right_hip': 24,
    'left_knee': 25, 'right_knee': 26,
    'left_ankle': 27, 'right_ankle': 28,
}

# Feature names for ablation labeling
ANATOMICAL_NAMES = [
    "P1:hip_center",
    "P2:shoulder_center",
    "P3:body_center",
    "P4:X_intersection",
    "P5:ankle_center",
    "P6:knee_center",
    "V1:head_hip_vec",
    "S1:diagonal_ratio",
    "S2:head_hip_angle",
    "S3:trunk_aspect",
]

# Which dimension indices each feature occupies in the 17-dim vector
FEATURE_SLICES = {
    "P1:hip_center":        slice(0, 2),
    "P2:shoulder_center":   slice(2, 4),
    "P3:body_center":       slice(4, 6),
    "P4:X_intersection":    slice(6, 8),
    "P5:ankle_center":      slice(8, 10),
    "P6:knee_center":       slice(10, 12),
    "V1:head_hip_vec":      slice(12, 14),
    "S1:diagonal_ratio":    slice(14, 15),
    "S2:head_hip_angle":    slice(15, 16),
    "S3:trunk_aspect":      slice(16, 17),
}


def _line_intersection_2d(p1, p2, p3, p4):
    """Intersection of line(p1→p2) and line(p3→p4) in 2D (x,y only). Returns (x,y) or (0,0)."""
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    x3, y3 = p3[0], p3[1]
    x4, y4 = p4[0], p4[1]
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-8:
        return np.array([0.0, 0.0], dtype=np.float32)
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    ix = x1 + t * (x2 - x1)
    iy = y1 + t * (y2 - y1)
    return np.array([ix, iy], dtype=np.float32)


def compute_anatomical_features(kps_current: np.ndarray) -> np.ndarray:
    """
    Compute 17 new anatomical features from a single frame's (33, 3) keypoints.
    Returns (17,) float32 array, zero-filled if detection missing.
    """
    if np.all(kps_current == 0):
        return np.zeros(17, dtype=np.float32)

    feats = []

    # Extract keypoints
    nose = kps_current[KP['nose']]
    l_sho = kps_current[KP['left_shoulder']]
    r_sho = kps_current[KP['right_shoulder']]
    l_hip = kps_current[KP['left_hip']]
    r_hip = kps_current[KP['right_hip']]
    l_knee = kps_current[KP['left_knee']]
    r_knee = kps_current[KP['right_knee']]
    l_ankle = kps_current[KP['left_ankle']]
    r_ankle = kps_current[KP['right_ankle']]

    # P1: hip_center (2)
    hip_ctr = (l_hip + r_hip) * 0.5
    feats.extend(hip_ctr[:2])

    # P2: shoulder_center (2)
    sho_ctr = (l_sho + r_sho) * 0.5
    feats.extend(sho_ctr[:2])

    # P3: body_center (2)
    body_ctr = (hip_ctr + sho_ctr) * 0.5
    feats.extend(body_ctr[:2])

    # P4: X_intersection (2)
    x_int = _line_intersection_2d(l_hip, r_sho, r_hip, l_sho)
    feats.extend(x_int)

    # P5: ankle_center (2)
    ankle_ctr = (l_ankle + r_ankle) * 0.5
    feats.extend(ankle_ctr[:2])

    # P6: knee_center (2)
    knee_ctr = (l_knee + r_knee) * 0.5
    feats.extend(knee_ctr[:2])

    # V1: head_hip_vec (2)
    head_hip_vec = nose[:2] - hip_ctr[:2]
    feats.extend(head_hip_vec)

    # S1: diagonal_ratio (1)
    diag1 = np.linalg.norm(l_sho[:2] - r_hip[:2])
    diag2 = np.linalg.norm(r_sho[:2] - l_hip[:2])
    diag_ratio = diag1 / (diag2 + 1e-8)
    feats.append(diag_ratio)

    # S2: head_hip_angle vs vertical (1)
    vec_hh = head_hip_vec
    vertical = np.array([0.0, -1.0], dtype=np.float32)  # y-up in image coords
    dot = np.dot(vec_hh, vertical)
    norm_v = np.linalg.norm(vec_hh) + 1e-8
    angle = np.arccos(np.clip(dot / norm_v, -1.0, 1.0))
    feats.append(angle)

    # S3: trunk_aspect (1)
    torso_len = np.linalg.norm(sho_ctr[:2] - hip_ctr[:2])
    shoulder_w = np.linalg.norm(l_sho[:2] - r_sho[:2])
    trunk_aspect = torso_len / (shoulder_w + 1e-8)
    feats.append(trunk_aspect)

    return np.array(feats, dtype=np.float32)


def compute_window_anatomical(kps_window: np.ndarray) -> np.ndarray:
    """
    Compute 17 anatomical features for each frame in a (T, 33, 3) window.
    Returns (T, 17) array.
    """
    T = kps_window.shape[0]
    feats = [compute_anatomical_features(kps_window[t]) for t in range(T)]
    return np.stack(feats, axis=0)

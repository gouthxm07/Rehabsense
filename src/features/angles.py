"""
Joint-angle geometry: angle(A, B, C) = angle at vertex B, in degrees.
This is the arccos((A-B)·(C-B) / (|A-B||C-B|)) formula from the blueprint.
"""

import numpy as np


def calculate_angle(a, b, c):
    """
    a, b, c: (x, y) tuples (visibility ignored if present).
    Returns the angle at point b, in degrees, range [0, 180].
    """
    a = np.array(a[:2])
    b = np.array(b[:2])
    c = np.array(c[:2])

    ba = a - b
    bc = c - b

    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom == 0:
        return None

    cosine = np.dot(ba, bc) / denom
    cosine = np.clip(cosine, -1.0, 1.0)  # guard against float rounding > |1|
    angle = np.degrees(np.arccos(cosine))
    return float(angle)


def angle_for_exercise(landmarks, joint_triplet):
    """
    landmarks: dict from PoseDetector.get_landmark_dict()
    joint_triplet: e.g. ("left_hip", "left_knee", "left_ankle")
    """
    a_name, b_name, c_name = joint_triplet
    if not all(name in landmarks for name in joint_triplet):
        return None
    return calculate_angle(landmarks[a_name], landmarks[b_name], landmarks[c_name])

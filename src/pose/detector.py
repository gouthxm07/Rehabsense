"""
Pose detection wrapper around MediaPipe Pose.

Gives you two things any downstream code needs:
1. get_landmark_dict(results) -> {"left_knee": (x, y, visibility), ...}
2. draw_skeleton(frame, results) -> annotated frame for the live demo
"""

import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Map MediaPipe's 33 landmark indices to friendly names we use everywhere else.
LANDMARK_MAP = {
    "left_shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
    "left_elbow": mp_pose.PoseLandmark.LEFT_ELBOW,
    "right_elbow": mp_pose.PoseLandmark.RIGHT_ELBOW,
    "left_wrist": mp_pose.PoseLandmark.LEFT_WRIST,
    "right_wrist": mp_pose.PoseLandmark.RIGHT_WRIST,
    "left_hip": mp_pose.PoseLandmark.LEFT_HIP,
    "right_hip": mp_pose.PoseLandmark.RIGHT_HIP,
    "left_knee": mp_pose.PoseLandmark.LEFT_KNEE,
    "right_knee": mp_pose.PoseLandmark.RIGHT_KNEE,
    "left_ankle": mp_pose.PoseLandmark.LEFT_ANKLE,
    "right_ankle": mp_pose.PoseLandmark.RIGHT_ANKLE,
}


class PoseDetector:
    def __init__(self, min_detection_confidence=0.6, min_tracking_confidence=0.6):
        self.pose = mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr):
        """Run pose estimation on a single BGR frame (as read by cv2)."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.pose.process(frame_rgb)
        return results

    def get_landmark_dict(self, results, min_visibility=0.5):
        """Return {name: (x, y, visibility)} in normalized 0-1 coords, or None."""
        if not results.pose_landmarks:
            return None
        lm = results.pose_landmarks.landmark
        out = {}
        for name, idx in LANDMARK_MAP.items():
            point = lm[idx.value]
            out[name] = (point.x, point.y, point.visibility)
        # basic quality gate: bail if key joints are not confidently visible
        key_joints = ["left_hip", "left_knee", "left_shoulder"]
        if any(out[j][2] < min_visibility for j in key_joints):
            return None
        return out

    def draw_skeleton(self, frame_bgr, results):
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame_bgr,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 120, 255), thickness=2),
            )
        return frame_bgr

    def close(self):
        self.pose.close()

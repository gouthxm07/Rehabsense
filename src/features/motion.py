"""
Frame-to-frame motion features: angular velocity, running ROM (min/max),
and torso displacement (a simple stability proxy).
"""

import time
import numpy as np


class AngleTracker:
    """Tracks one joint angle over time to derive velocity + ROM for a session."""

    def __init__(self):
        self.last_angle = None
        self.last_time = None
        self.min_angle = None
        self.max_angle = None
        self.history = []  # (timestamp, angle)

    def update(self, angle):
        if angle is None:
            return {"velocity": 0.0, "rom": self.rom()}

        now = time.time()
        velocity = 0.0
        if self.last_angle is not None and self.last_time is not None:
            dt = max(now - self.last_time, 1e-3)
            velocity = abs(angle - self.last_angle) / dt  # deg/sec

        self.last_angle = angle
        self.last_time = now
        self.min_angle = angle if self.min_angle is None else min(self.min_angle, angle)
        self.max_angle = angle if self.max_angle is None else max(self.max_angle, angle)
        self.history.append((now, angle))

        return {"velocity": velocity, "rom": self.rom()}

    def rom(self):
        if self.min_angle is None or self.max_angle is None:
            return 0.0
        return round(self.max_angle - self.min_angle, 1)

    def reset(self):
        self.__init__()


class TorsoStability:
    """Approximates trunk sway using shoulder-hip midpoint displacement."""

    def __init__(self, window=15):
        self.window = window
        self.positions = []

    def update(self, landmarks):
        if not all(k in landmarks for k in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")):
            return 0.0
        sx = (landmarks["left_shoulder"][0] + landmarks["right_shoulder"][0]) / 2
        sy = (landmarks["left_shoulder"][1] + landmarks["right_shoulder"][1]) / 2
        hx = (landmarks["left_hip"][0] + landmarks["right_hip"][0]) / 2
        hy = (landmarks["left_hip"][1] + landmarks["right_hip"][1]) / 2
        midpoint = np.array([(sx + hx) / 2, (sy + hy) / 2])

        self.positions.append(midpoint)
        if len(self.positions) > self.window:
            self.positions.pop(0)

        if len(self.positions) < 2:
            return 0.0
        arr = np.array(self.positions)
        displacement = np.linalg.norm(arr.std(axis=0))  # normalized 0-1 coords
        return float(displacement)

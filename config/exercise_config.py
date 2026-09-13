"""
Central config for every supported exercise.

Everything a physiotherapist would want to tune lives here — angle
joints to track, the "rest" vs "flexed" thresholds that drive the
repetition state machine, and a target ROM used only as a configurable
demo baseline (NOT a medical target).
"""

# Which 3 landmarks define the tracked angle for each exercise.
# Names must match the keys returned by src/pose/detector.py -> get_landmark_dict()
EXERCISES = {
    "knee_flexion": {
        "label": "Knee Flexion / Extension",
        "joint_triplet": ("left_hip", "left_knee", "left_ankle"),
        "rest_angle_min": 150,      # standing / leg straight
        "flexed_angle_max": 110,    # knee bent enough to count as a rep
        "target_rom": 80,           # configurable demo target, not clinical
    },
    "straight_leg_raise": {
        "label": "Straight Leg Raise",
        "joint_triplet": ("left_shoulder", "left_hip", "left_knee"),
        "rest_angle_min": 165,      # leg flat on ground/relaxed
        "flexed_angle_max": 130,    # leg raised
        "target_rom": 40,
    },
    "shoulder_raise": {
        "label": "Shoulder Raise",
        "joint_triplet": ("left_hip", "left_shoulder", "left_elbow"),
        "rest_angle_min": 20,       # arm down at side
        "flexed_angle_max": 90,     # arm raised to/above shoulder height
        "target_rom": 70,
        "invert": True,             # angle increases during the rep (see repetition_counter)
    },
    "elbow_flexion": {
        "label": "Elbow Flexion",
        "joint_triplet": ("left_shoulder", "left_elbow", "left_wrist"),
        "rest_angle_min": 160,      # arm straight
        "flexed_angle_max": 60,     # arm curled
        "target_rom": 100,
    },
}

# Movement-quality thresholds used by src/rehab/form_analyzer.py
FORM_RULES = {
    "max_angular_velocity": 220,     # deg/sec - above this = "too fast"
    "max_torso_displacement": 0.06,  # normalized (0-1 frame width/height) - above = "unstable torso"
    "min_rom_ratio": 0.6,            # actual_rom / target_rom below this = "limited ROM"
}

# Rehabilitation score weights (must sum to 1.0) - see src/rehab/scoring.py
SCORE_WEIGHTS = {
    "movement_quality": 0.30,
    "range_of_motion": 0.25,
    "correct_reps": 0.20,
    "consistency": 0.15,
    "completion": 0.10,
}

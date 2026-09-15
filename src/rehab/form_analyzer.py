"""
Rule-based movement-quality layer (Feature: Correct vs Incorrect Movement
Detection + Explainable AI from the blueprint).

Produces both a short feedback message AND the reasons behind it, so the
UI can show "Movement Quality: 72% — because ROM was below target and
speed was inconsistent" instead of a bare label.

Also includes compensation/cheating detection: a secondary joint check
that flags when the patient uses other muscle groups to complete a rep
(e.g. hip hike during shoulder raise, shoulder shrug during elbow curl).
This is a clinically relevant pattern in real rehabilitation.

None of this is a medical judgement — it only reports whether measured
motion is inside the configured, adjustable parameters for the exercise.
"""

import numpy as np
from config.exercise_config import FORM_RULES


# ---------------------------------------------------------------------------
# Compensation detection
# ---------------------------------------------------------------------------

def check_compensation(exercise_name: str, landmarks: dict, baseline: dict | None) -> dict:
    """
    Detect secondary-joint compensation patterns for each exercise.

    Args:
        exercise_name: key from EXERCISES config (e.g. "shoulder_raise")
        landmarks:     dict from PoseDetector.get_landmark_dict()
        baseline:      landmark dict captured at session start (resting posture)

    Returns:
        {"detected": bool, "type": str, "description": str}
    """
    result = {"detected": False, "type": None, "description": None}

    def _get(lm, name):
        return np.array(lm[name][:2]) if name in lm else None

    if exercise_name == "shoulder_raise":
        # Hip hike: during arm raise, hip should stay level.
        # Flag if hip midpoint rises > 6% of estimated body height vs baseline.
        if baseline and all(k in landmarks for k in ("left_hip", "right_hip")) \
                and all(k in baseline  for k in ("left_hip", "right_hip")):
            cur_hip_y  = (landmarks["left_hip"][1] + landmarks["right_hip"][1]) / 2
            base_hip_y = (baseline["left_hip"][1]  + baseline["right_hip"][1])  / 2
            # In normalised coords (0-1), a rise of >0.06 is clinically significant
            if (base_hip_y - cur_hip_y) > 0.06:
                result.update(detected=True, type="hip_hike",
                               description="Hip hike detected — keep hips level during shoulder raise.")

    elif exercise_name == "elbow_flexion":
        # Shoulder shrug: shoulder should not rise during elbow curl.
        if baseline and "left_shoulder" in landmarks and "left_shoulder" in baseline:
            cur_sh_y  = landmarks["left_shoulder"][1]
            base_sh_y = baseline["left_shoulder"][1]
            if (base_sh_y - cur_sh_y) > 0.05:
                result.update(detected=True, type="shoulder_shrug",
                               description="Shoulder shrug detected — keep shoulder relaxed during elbow curl.")

    elif exercise_name == "knee_flexion":
        # Forward trunk lean: torso angle from vertical should stay < 20°
        if all(k in landmarks for k in ("left_shoulder", "left_hip")):
            sh = _get(landmarks, "left_shoulder")
            hp = _get(landmarks, "left_hip")
            dx = sh[0] - hp[0]
            dy = hp[1] - sh[1]
            trunk_angle = abs(np.degrees(np.arctan2(dx, dy))) if dy != 0 else 0
            if trunk_angle > 20:
                result.update(detected=True, type="trunk_lean",
                               description=f"Forward lean {trunk_angle:.0f}° — keep back straight during knee flexion.")

    elif exercise_name == "straight_leg_raise":
        # Contralateral hip drop: pelvis should stay level.
        if all(k in landmarks for k in ("left_hip", "right_hip")):
            lh = landmarks["left_hip"][1]
            rh = landmarks["right_hip"][1]
            if abs(lh - rh) > 0.07:
                result.update(detected=True, type="hip_drop",
                               description="Hip drop detected — keep pelvis level during straight leg raise.")

    return result


# ---------------------------------------------------------------------------
# Form analysis (existing, extended)
# ---------------------------------------------------------------------------

def analyze_form(exercise_cfg, current_rom, velocity, torso_displacement,
                 exercise_name: str = None, landmarks: dict = None, baseline: dict = None):
    """
    Analyse movement quality for a single frame / rep.

    Optional args exercise_name, landmarks, baseline enable compensation
    detection when the Python CLI pipeline provides landmark data.
    """
    reasons = []
    penalty = 0

    target_rom = exercise_cfg.get("target_rom", 1) or 1
    rom_ratio = current_rom / target_rom if target_rom else 0

    if rom_ratio < FORM_RULES["min_rom_ratio"]:
        reasons.append("Range of motion was below the configured target.")
        penalty += 25

    if velocity > FORM_RULES["max_angular_velocity"]:
        reasons.append("Movement speed was faster than the configured comfortable pace.")
        penalty += 15

    if torso_displacement > FORM_RULES["max_torso_displacement"]:
        reasons.append("Upper-body/torso position was not stable during the movement.")
        penalty += 15

    # Compensation detection (when landmark data is provided)
    compensation = {"detected": False, "type": None, "description": None}
    if exercise_name and landmarks:
        compensation = check_compensation(exercise_name, landmarks, baseline)
        if compensation["detected"]:
            reasons.append(compensation["description"])
            penalty += 20

    quality_score = max(0, 100 - penalty)

    feedback_map = {
        "Range of motion was below the configured target.": "Try to increase your range of motion.",
        "Movement speed was faster than the configured comfortable pace.": "Try moving more slowly and with control.",
        "Upper-body/torso position was not stable during the movement.": "Keep your torso stable during the movement.",
    }

    if not reasons:
        feedback = "Good movement."
    else:
        parts = []
        for r in reasons:
            if r in feedback_map:
                parts.append(feedback_map[r])
            elif compensation["detected"] and r == compensation["description"]:
                parts.append(r)  # compensation message is already human-readable
        feedback = " / ".join(parts) if parts else "Check your form."

    return {
        "quality_score": quality_score,
        "feedback": feedback,
        "reasons": reasons,          # the "Why?" explainability list
        "compensation": compensation,
    }

"""
Rule-based movement-quality layer (Feature: Correct vs Incorrect Movement
Detection + Explainable AI from the blueprint).

Produces both a short feedback message AND the reasons behind it, so the
UI can show "Movement Quality: 72% — because ROM was below target and
speed was inconsistent" instead of a bare label.

None of this is a medical judgement - it only reports whether measured
motion is inside the configured, adjustable parameters for the exercise.
"""

from config.exercise_config import FORM_RULES


def analyze_form(exercise_cfg, current_rom, velocity, torso_displacement):
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

    quality_score = max(0, 100 - penalty)

    if not reasons:
        feedback = "Good movement."
    else:
        feedback = " / ".join(
            {
                "Range of motion was below the configured target.": "Try to increase your range of motion.",
                "Movement speed was faster than the configured comfortable pace.": "Try moving more slowly and with control.",
                "Upper-body/torso position was not stable during the movement.": "Keep your torso stable during the movement.",
            }[r]
            for r in reasons
        )

    return {
        "quality_score": quality_score,
        "feedback": feedback,
        "reasons": reasons,  # the "Why?" explainability list
    }

"""
Weighted rehabilitation score, per blueprint section 15:

Movement Quality       30%
Range of Motion        25%
Correct Repetitions    20%
Consistency            15%
Exercise Completion    10%

This is a software-defined performance indicator, NOT a medical
severity score.
"""

from config.exercise_config import SCORE_WEIGHTS


def calculate_rehab_score(
    movement_quality,      # 0-100, avg of per-rep form_analyzer quality_score
    rom_achieved,          # degrees
    target_rom,            # degrees (from exercise config)
    correct_reps,
    total_reps,
    target_reps,
    rep_quality_scores,    # list of per-rep quality scores, for consistency
):
    rom_pct = min(100, (rom_achieved / target_rom) * 100) if target_rom else 0
    correct_pct = (correct_reps / total_reps) * 100 if total_reps else 0
    completion_pct = min(100, (total_reps / target_reps) * 100) if target_reps else 0

    if len(rep_quality_scores) >= 2:
        spread = max(rep_quality_scores) - min(rep_quality_scores)
        consistency_pct = max(0, 100 - spread)
    else:
        consistency_pct = 100  # not enough data to penalize yet

    weighted = (
        movement_quality * SCORE_WEIGHTS["movement_quality"]
        + rom_pct * SCORE_WEIGHTS["range_of_motion"]
        + correct_pct * SCORE_WEIGHTS["correct_reps"]
        + consistency_pct * SCORE_WEIGHTS["consistency"]
        + completion_pct * SCORE_WEIGHTS["completion"]
    )

    return {
        "rehab_score": round(weighted, 1),
        "components": {
            "movement_quality": round(movement_quality, 1),
            "range_of_motion_pct": round(rom_pct, 1),
            "correct_reps_pct": round(correct_pct, 1),
            "consistency_pct": round(consistency_pct, 1),
            "completion_pct": round(completion_pct, 1),
        },
    }

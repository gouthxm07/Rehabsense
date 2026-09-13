"""
RehabSense AI — live camera demo (blueprint sections 29 + 39).

Run:  python main.py --patient P001 --exercise knee_flexion --target-reps 10

Controls:
  q  - end session, save to database, quit
  r  - reset current session counters

Note: this is a hackathon prototype for assistive monitoring, not a
diagnostic or medical device.
"""

import argparse
import time
import cv2

from config.exercise_config import EXERCISES
from src.pose.detector import PoseDetector
from src.features.angles import angle_for_exercise
from src.features.motion import AngleTracker, TorsoStability
from src.rehab.repetition_counter import RepetitionCounter
from src.rehab.form_analyzer import analyze_form
from src.rehab.scoring import calculate_rehab_score
from src.database import db


def draw_hud(frame, exercise_label, angle, rep_snapshot, rom, feedback, target_reps):
    h, w = frame.shape[:2]
    overlay_lines = [
        f"Exercise: {exercise_label}",
        f"Angle: {angle:.0f} deg" if angle is not None else "Angle: --",
        f"Reps: {rep_snapshot['correct_reps']}/{target_reps}  (incorrect: {rep_snapshot['incorrect_reps']})",
        f"ROM: {rom:.0f} deg",
        f"Feedback: {feedback}",
    ]
    cv2.rectangle(frame, (0, 0), (w, 25 * len(overlay_lines) + 15), (0, 0, 0), -1)
    for i, line in enumerate(overlay_lines):
        cv2.putText(frame, line, (10, 25 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", default="P001")
    parser.add_argument("--exercise", default="knee_flexion", choices=EXERCISES.keys())
    parser.add_argument("--target-reps", type=int, default=10)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    db.init_db()
    db.ensure_patient(args.patient)

    exercise_cfg = EXERCISES[args.exercise]
    detector = PoseDetector()
    rep_counter = RepetitionCounter(exercise_cfg)
    angle_tracker = AngleTracker()
    torso_tracker = TorsoStability()

    rep_quality_scores = []
    latest_feedback = "Waiting for pose..."
    session_start = time.time()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Check --camera index or permissions.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        results = detector.process(frame)
        landmarks = detector.get_landmark_dict(results)

        angle = None
        rom = angle_tracker.rom()
        rep_snapshot = rep_counter.snapshot()

        if landmarks:
            angle = angle_for_exercise(landmarks, exercise_cfg["joint_triplet"])
            motion = angle_tracker.update(angle)
            torso_disp = torso_tracker.update(landmarks)

            prev_state = rep_counter.state
            rep_snapshot = rep_counter.update(angle)

            form_result = analyze_form(exercise_cfg, motion["rom"], motion["velocity"], torso_disp)
            latest_feedback = form_result["feedback"]

            # A rep just completed -> log its quality for the consistency score
            if prev_state == "FLEXED" and rep_snapshot["state"] == "REST":
                rep_quality_scores.append(form_result["quality_score"])

            rom = motion["rom"]

        frame = detector.draw_skeleton(frame, results)
        frame = draw_hud(frame, exercise_cfg["label"], angle, rep_snapshot, rom,
                          latest_feedback, args.target_reps)

        cv2.imshow("RehabSense AI - press q to end session", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            rep_counter.reset()
            angle_tracker.reset()
            rep_quality_scores.clear()
            session_start = time.time()

        if rep_snapshot["total_reps"] >= args.target_reps:
            # auto-end once target reps reached
            time.sleep(1)
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

    # ---- Save session summary ----
    duration = round(time.time() - session_start, 1)
    avg_quality = sum(rep_quality_scores) / len(rep_quality_scores) if rep_quality_scores else 0
    final_snapshot = rep_counter.snapshot()
    final_rom = angle_tracker.rom()

    score_result = calculate_rehab_score(
        movement_quality=avg_quality,
        rom_achieved=final_rom,
        target_rom=exercise_cfg["target_rom"],
        correct_reps=final_snapshot["correct_reps"],
        total_reps=final_snapshot["total_reps"] or 1,
        target_reps=args.target_reps,
        rep_quality_scores=rep_quality_scores,
    )

    session_id = db.save_session(
        patient_id=args.patient,
        exercise_name=args.exercise,
        duration=duration,
        total_reps=final_snapshot["total_reps"],
        correct_reps=final_snapshot["correct_reps"],
        incorrect_reps=final_snapshot["incorrect_reps"],
        rom=final_rom,
        movement_score=round(avg_quality, 1),
        rehab_score=score_result["rehab_score"],
    )
    db.check_for_alerts(args.patient, session_id, final_rom, score_result["rehab_score"])

    print("\n--- SESSION SUMMARY ---")
    print(f"Patient: {args.patient}  Exercise: {exercise_cfg['label']}")
    print(f"Total reps: {final_snapshot['total_reps']}  Correct: {final_snapshot['correct_reps']}  "
          f"Incorrect: {final_snapshot['incorrect_reps']}")
    print(f"ROM: {final_rom}°   Movement quality: {avg_quality:.1f}%")
    print(f"Rehabilitation score: {score_result['rehab_score']}/100")
    print("Saved to database. Open the dashboard with: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()

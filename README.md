# RehabSense AI — Implementation Guide

This repo is a **working starter implementation** of the RehabSense AI blueprint:
pose detection → joint angles/ROM → repetition counting → rule-based form
analysis → ML exercise recognition → rehabilitation score → SQLite storage →
Streamlit patient/therapist dashboards.

All core modules have already been written and tested (angle math, rep
state machine, scoring, database, and a Random Forest exercise classifier
trained on synthetic data). What's left is mostly **calibration with your
own webcam/body** and **swapping synthetic training data for real recordings**.

> Prototype only — not a medical device, not a diagnostic tool, not a
> substitute for a physiotherapist.

---

## 0. One-time setup

```bash
cd rehabsense-ai
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Train the exercise classifier (synthetic data, ready to go)

```bash
python -m src.ml.train
```

This generates a synthetic dataset (`data/processed/synthetic_training_data.csv`)
and saves `models/exercise_classifier.pkl` + `models/scaler.pkl`. It's a
placeholder so the rest of the pipeline works immediately — **replace it with
real recorded sessions before your demo** (see step 4).

## 2. Run a live session

```bash
python main.py --patient P001 --exercise knee_flexion --target-reps 10
```

- A window opens with your webcam feed, skeleton overlay, and a live HUD
  showing angle, rep count, ROM, and feedback.
- Press **q** to end early and save; press **r** to reset counters mid-session.
- The session auto-ends and saves once `--target-reps` is reached.
- Supported `--exercise` values: `knee_flexion`, `straight_leg_raise`,
  `shoulder_raise`, `elbow_flexion` (see `config/exercise_config.py`).

## 3. View the dashboard

```bash
streamlit run dashboard/app.py
```

- **Patient View** — pick a patient ID, see latest score/ROM/reps and a
  trend chart across sessions.
- **Physiotherapist View** — all patients, ROM/score trend per patient, and
  any auto-generated review flags (e.g. a sudden ROM drop).

Run a few sessions with different simulated performance (vary how far you
bend, how fast you move) so the dashboard has a trend to show.

## 4. Replace synthetic data with real recordings (important before demo day)

The synthetic classifier is only there so you have an end-to-end pipeline
on day one. To make exercise recognition real:

1. Add a small CSV logger inside `main.py`'s loop (append `angle`,
   `velocity`, `rom`, and the known `exercise` label to
   `data/raw/features_log.csv` each frame) while you and teammates perform
   each of the 4 exercises correctly and a few times incorrectly.
2. Aim for ~150–300 labeled rows per exercise (a couple minutes of
   recording each).
3. Point `src/ml/train.py`'s `train()` function at your real CSV instead of
   `generate_synthetic_dataset()`.
4. Retrain: `python -m src.ml.train` — check the printed classification
   report before trusting it on stage.

## 5. Calibrate thresholds for your camera setup

Everything a physiotherapist would tune lives in **`config/exercise_config.py`**:
- `rest_angle_min` / `flexed_angle_max` per exercise — adjust based on your
  own test reps (print the live angle from the HUD and watch its range).
- `target_rom` — used only for the demo score, not a clinical claim.
- `FORM_RULES` — velocity/torso-stability/ROM-ratio thresholds for the
  "good form" vs "check your form" feedback.

Test in your actual demo lighting/room before relying on these numbers.

---

## Suggested build order (maps to the blueprint's 7-day plan)

| Day | Focus | What's already done here | What you still do |
|---|---|---|---|
| 1 | Planning | Repo structure, config file | Confirm your 3–4 exercises, assign owners |
| 2 | Pose detection | `src/pose/detector.py`, skeleton overlay in `main.py` | Test detection in your actual room/lighting |
| 3 | Movement analysis | `src/features/angles.py`, `src/features/motion.py` | Calibrate thresholds in `exercise_config.py` |
| 4 | Machine learning | `src/ml/train.py` + `predict.py`, synthetic data working | Swap in real recorded data, retrain, check report |
| 5 | Rehab engine | `repetition_counter.py`, `form_analyzer.py`, `scoring.py`, `db.py` | Tune scoring weights/feedback wording if needed |
| 6 | Dashboard | `dashboard/app.py` (patient + therapist tabs) | Style pass, add your team's branding |
| 7 | Demo + presentation | Auto-session-save, alert system (`check_for_alerts`) | Rehearse the live demo script below, prep backup video |

## Demo script (matches blueprint section 39)

1. Open a terminal, run `python main.py --patient P001 --exercise knee_flexion --target-reps 10`.
2. Perform reps live in front of the judges — narrate what the skeleton
   overlay and live HUD are showing.
3. Deliberately do one shallow rep to show "limited range of motion"
   feedback firing.
4. Let it auto-end at 10 reps — read the printed session summary aloud.
5. Switch to `streamlit run dashboard/app.py`, show the Patient View trend
   chart (ideally after 2–3 pre-recorded practice sessions so there's a
   visible trend), then the Physiotherapist View with any review flags.
6. Close with the positioning line: *"This gives physiotherapists objective
   data between appointments — it doesn't replace their judgment."*

## Known limitations to state proactively to judges

- Angle thresholds are heuristic and camera-angle dependent — not clinically
  validated.
- The classifier ships trained on synthetic data by default; swap in real
  data before claiming real accuracy numbers.
- Single-camera 2D pose estimation can't capture true 3D joint rotation —
  frame the camera perpendicular to the movement plane for best results.
- This is explicitly an assistive monitoring prototype, not a diagnostic or
  emergency-detection system.

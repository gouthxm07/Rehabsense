"""
RehabSense AI dashboard.

Run:  streamlit run dashboard/app.py

Two tabs:
  - Patient view: pick a patient, see session history + progress trend.
  - Physiotherapist view: all patients, ROM/accuracy trends, review flags.

This dashboard reads from the SQLite DB that main.py writes to — run a
session or two with main.py first so there's data to show.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # allow `src.` imports

import streamlit as st
import pandas as pd
import plotly.express as px

from src.database import db
from config.exercise_config import EXERCISES

st.set_page_config(page_title="RehabSense AI", layout="wide")
db.init_db()

st.title("🦾 RehabSense AI")
st.caption(
    "AI-assisted rehabilitation monitoring prototype. "
    "This is not a medical device and does not provide diagnoses."
)

tab_patient, tab_therapist = st.tabs(["Patient View", "Physiotherapist View"])

# ---------------------------------------------------------------- PATIENT TAB
with tab_patient:
    patients = db.get_all_patients()
    patient_ids = [p["patient_id"] for p in patients] or ["P001"]
    selected_patient = st.selectbox("Select patient", patient_ids, key="patient_select")

    sessions = db.get_sessions_for_patient(selected_patient)

    if not sessions:
        st.info("No sessions recorded yet. Run `python main.py --patient "
                 f"{selected_patient}` to record a live session.")
    else:
        df = pd.DataFrame(sessions)
        latest = df.iloc[-1]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Latest Rehabilitation Score", f"{latest['rehab_score']}/100")
        col2.metric("Latest ROM", f"{latest['rom']}°")
        col3.metric("Correct Reps", f"{latest['correct_reps']}/{latest['total_reps']}")
        col4.metric("Movement Quality", f"{latest['movement_score']}%")

        st.subheader("Progress Over Sessions")
        df["session_number"] = range(1, len(df) + 1)
        fig = px.line(
            df, x="session_number", y=["rehab_score", "rom"],
            markers=True, labels={"value": "Score / Degrees", "session_number": "Session"},
            title="Rehabilitation Score & ROM Trend",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Session History")
        st.dataframe(
            df[["date", "exercise_name", "total_reps", "correct_reps", "incorrect_reps",
                "rom", "movement_score", "rehab_score"]],
            use_container_width=True,
        )

# ------------------------------------------------------------- THERAPIST TAB
with tab_therapist:
    patients = db.get_all_patients()
    if not patients:
        st.info("No patients yet.")
    else:
        for p in patients:
            with st.expander(f"Patient: {p['patient_id']}", expanded=False):
                sessions = db.get_sessions_for_patient(p["patient_id"])
                if not sessions:
                    st.write("No sessions yet.")
                    continue

                df = pd.DataFrame(sessions)
                df["session_number"] = range(1, len(df) + 1)

                trend = "IMPROVING" if len(df) >= 2 and df["rehab_score"].iloc[-1] > df["rehab_score"].iloc[0] else "STABLE / NEEDS REVIEW"
                st.markdown(f"**Progress: {trend}**")

                fig = px.line(
                    df, x="session_number", y=["rom", "rehab_score"],
                    markers=True, title=f"{p['patient_id']} — ROM & Score Trend",
                )
                st.plotly_chart(fig, use_container_width=True)

                alerts = db.get_alerts_for_patient(p["patient_id"])
                if alerts:
                    st.markdown("**Review Flags:**")
                    for a in alerts:
                        st.warning(f"[{a['date'][:10]}] {a['description']}")
                else:
                    st.success("No review flags.")

st.divider()
st.caption(
    "RehabSense AI is a hackathon prototype for assistive rehabilitation "
    "monitoring — not a diagnostic system or a substitute for professional "
    "clinical judgment."
)

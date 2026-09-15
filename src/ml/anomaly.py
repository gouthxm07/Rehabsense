"""
Isolation Forest anomaly detector for RehabSense AI.

Detects sessions that deviate significantly from a patient's
established movement patterns. More principled than fixed if/else
thresholds — the model learns what "normal" looks like for each patient.

Usage:
    detector = AnomalyDetector()
    detector.fit(sessions_df)            # train on historical sessions
    score = detector.score(session_dict) # -1 = anomaly, 1 = normal
    is_anomaly = detector.is_anomaly(session_dict)

Not a diagnostic tool — flags sessions for physiotherapist review only.
"""

import numpy as np
from sklearn.ensemble import IsolationForest

# Minimum sessions required before anomaly detection is meaningful
MIN_SESSIONS_FOR_MODEL = 5

FEATURE_COLS = ["rom", "movement_score", "rehab_score", "correct_reps", "total_reps"]


class AnomalyDetector:
    """
    Per-patient Isolation Forest trained on session feature vectors.
    Falls back gracefully when insufficient history exists.
    """

    def __init__(self, contamination: float = 0.15, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self._model: IsolationForest | None = None
        self._fitted = False
        self._n_sessions = 0

    # ------------------------------------------------------------------
    def fit(self, sessions):
        """
        sessions: list[dict] or pd.DataFrame — each row is one session.
        Silently skips training if there are fewer than MIN_SESSIONS_FOR_MODEL rows.
        """
        import pandas as pd

        if isinstance(sessions, list):
            df = pd.DataFrame(sessions)
        else:
            df = sessions.copy()

        available_cols = [c for c in FEATURE_COLS if c in df.columns]
        if not available_cols:
            return self

        df_clean = df[available_cols].fillna(0)
        self._n_sessions = len(df_clean)

        if self._n_sessions < MIN_SESSIONS_FOR_MODEL:
            self._fitted = False
            return self

        self._model = IsolationForest(
            n_estimators=100,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self._model.fit(df_clean.values)
        self._fitted = True
        self._feature_cols = available_cols
        return self

    # ------------------------------------------------------------------
    def score(self, session: dict) -> float:
        """
        Returns the raw anomaly score (lower = more anomalous).
        Returns 0.0 if the model is not fitted yet.
        """
        if not self._fitted or self._model is None:
            return 0.0
        vec = np.array([[session.get(c, 0) for c in self._feature_cols]], dtype=float)
        return float(self._model.score_samples(vec)[0])

    # ------------------------------------------------------------------
    def is_anomaly(self, session: dict) -> bool:
        """
        Returns True if this session is an outlier relative to the
        patient's historical pattern.
        """
        if not self._fitted or self._model is None:
            return False
        vec = np.array([[session.get(c, 0) for c in self._feature_cols]], dtype=float)
        prediction = self._model.predict(vec)[0]
        return prediction == -1  # -1 = outlier in sklearn

    # ------------------------------------------------------------------
    def status(self) -> dict:
        return {
            "fitted": self._fitted,
            "n_sessions": self._n_sessions,
            "min_required": MIN_SESSIONS_FOR_MODEL,
            "ready": self._fitted,
        }

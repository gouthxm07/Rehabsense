"""
ROM recovery trajectory predictor for RehabSense AI.

Uses polynomial regression (degree 1 or 2, whichever fits better) to
project a patient's Range of Motion into the future, and flags when
actual ROM is trailing behind the expected recovery curve.

Usage:
    result = predict_trajectory(sessions_df, weeks_ahead=2)
    # result = {
    #     "projected_rom": 78.4,
    #     "projected_rom_4w": 91.2,
    #     "on_track": True,
    #     "trend_slope": 3.2,   # degrees ROM gained per session
    #     "r_squared": 0.87,
    #     "degree": 1,
    #     "sessions_used": 7,
    #     "message": "On track — steady 3.2°/session improvement"
    # }

Not a medical prognosis — a software-defined trend indicator only.
"""

import numpy as np

# Assume ~3 sessions per week (configurable)
SESSIONS_PER_WEEK = 3
MIN_SESSIONS_FOR_PROJECTION = 3


def _r_squared(y_actual: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_actual - y_pred) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


def predict_trajectory(sessions, weeks_ahead: int = 2) -> dict:
    """
    sessions: list[dict] or pd.DataFrame, must contain 'rom' column,
              ordered by date ascending.

    Returns a dict with projection values and trend metadata.
    Returns a 'not_enough_data' result if < MIN_SESSIONS_FOR_PROJECTION sessions.
    """
    import pandas as pd

    if isinstance(sessions, list):
        df = pd.DataFrame(sessions)
    else:
        df = sessions.copy()

    if "rom" not in df.columns or len(df) < MIN_SESSIONS_FOR_PROJECTION:
        return {
            "projected_rom": None,
            "projected_rom_4w": None,
            "on_track": None,
            "trend_slope": None,
            "r_squared": None,
            "degree": None,
            "sessions_used": len(df) if "rom" in df.columns else 0,
            "message": f"Need at least {MIN_SESSIONS_FOR_PROJECTION} sessions for trajectory prediction.",
            "not_enough_data": True,
        }

    rom_values = df["rom"].astype(float).values
    n = len(rom_values)
    x = np.arange(n)

    # Fit degree-1 and degree-2 polynomials; pick the one with higher R²
    best_poly = None
    best_r2 = -np.inf
    best_degree = 1

    for degree in [1, 2]:
        if n < degree + 2:
            continue
        coeffs = np.polyfit(x, rom_values, deg=degree)
        poly = np.poly1d(coeffs)
        y_pred = poly(x)
        r2 = _r_squared(rom_values, y_pred)
        if r2 > best_r2:
            best_r2 = r2
            best_poly = poly
            best_degree = degree

    if best_poly is None:
        best_poly = np.poly1d(np.polyfit(x, rom_values, deg=1))
        best_r2 = 0.0
        best_degree = 1

    # Project forward
    sessions_2w = weeks_ahead * SESSIONS_PER_WEEK
    sessions_4w = 4 * SESSIONS_PER_WEEK
    x_2w = n - 1 + sessions_2w
    x_4w = n - 1 + sessions_4w

    projected_2w = float(max(0, best_poly(x_2w)))
    projected_4w = float(max(0, best_poly(x_4w)))

    # Trend slope: average improvement per session (linear component)
    if best_degree == 1:
        slope = float(best_poly.coeffs[0])
    else:
        # For quadratic, report the slope at the last data point
        deriv = best_poly.deriv()
        slope = float(deriv(n - 1))

    # "On track": actual last ROM vs expected at same point
    expected_last = float(best_poly(n - 1))
    actual_last = float(rom_values[-1])
    on_track = actual_last >= expected_last * 0.85  # within 15% of expected

    # Human-readable message
    if slope > 0:
        trend_desc = f"Gaining {slope:.1f}°/session on average"
    elif slope < -0.5:
        trend_desc = f"Declining {abs(slope):.1f}°/session — review recommended"
    else:
        trend_desc = "ROM is plateauing — consider exercise progression"

    status = "On track — " if on_track else "Behind target — "
    message = status + trend_desc

    return {
        "projected_rom": round(projected_2w, 1),
        "projected_rom_4w": round(projected_4w, 1),
        "on_track": on_track,
        "trend_slope": round(slope, 2),
        "r_squared": round(best_r2, 3),
        "degree": best_degree,
        "sessions_used": n,
        "message": message,
        "not_enough_data": False,
        # For plotting: x positions and fitted values
        "_x": x.tolist(),
        "_y_fit": [round(float(best_poly(xi)), 1) for xi in x],
        "_x_future": [n - 1 + i for i in range(1, sessions_4w + 1)],
        "_y_future": [round(float(max(0, best_poly(n - 1 + i))), 1) for i in range(1, sessions_4w + 1)],
    }

"""
PREDICTOR — maps raw mission inputs and engineered features to mission projections.

MVP "physics-inspired" + evidence-informed simulator:
- Cortisol Load (0–100)
- Cycle Variability (absolute +/- days)
- Bone Loss Rate (% BMD per month)

Designed to be replaced by a trained Azure ML model later.

Recommended run (module mode):
  cd iunoia-core
  python -m model.predictor

Also supports (script mode):
  python model/predictor.py
"""

"""
...docstring...
"""

from dataclasses import dataclass
from typing import Dict, Any
import math

try:
    from .features import build_features
    from .risk_scores import attach_risk_breakdown
except ImportError:
    from features import build_features
    from risk_scores import attach_risk_breakdown


# --- imports that work in BOTH module and script mode ---
try:
    # Module mode: python -m model.predictor
    from .features import build_features
    from .risk_scores import attach_risk_breakdown
except ImportError:
    # Script mode: python model/predictor.py
    from features import build_features
    from risk_scores import attach_risk_breakdown


# -------------------------
# Types
# -------------------------

@dataclass(frozen=True)
class Prediction:
    mission_day: int
    cortisol_load: float               # 0–100 (higher = worse)
    cycle_variability_days: float      # abs deviation from baseline cycle length (days)
    bone_loss_pct_per_month: float     # % BMD loss per month
    drivers: Dict[str, str]            # short explanations of top drivers


# -------------------------
# Helpers
# -------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# -------------------------
# Core simulator
# -------------------------

def predict_from_features(features: Dict[str, float]) -> Prediction:
    """
    Engineered features -> projections.
    Feature keys come from build_features() in features.py
    """

    mission_day = int(features.get("mission_day", 1))

    g = float(features.get("gravity_factor", 0.0))
    r = float(features.get("radiation_factor", 0.6))
    iso = float(features.get("isolation_factor", 0.6))

    stress = float(features.get("stress_score", 0.5))
    sleep_def = float(features.get("sleep_deficit", 0.0))

    irr_hist = float(features.get("cycle_irregularity_history", 0.0))
    bmd_conc = float(features.get("bone_density_concerns", 0.0))
    sleep_dis = float(features.get("sleep_disorders", 0.0))
    repro_flag = float(features.get("pcos_or_endometriosis_flag", 0.0))

    stress_x_sleep = float(features.get("stress_x_sleep", stress * sleep_def))
    rad_x_grav = float(features.get("rad_x_grav", r * g))
    iso_x_stress = float(features.get("iso_x_stress", iso * stress))

    # 1) Cortisol load (0–100)
    t = _clamp(mission_day / 180.0, 0.0, 1.5)
    time_gain = 0.25 + 0.55 * _sigmoid(3.0 * (t - 0.35))  # ~0.25..0.80

    base_cort = (
        0.40 * stress +
        0.35 * sleep_def +
        0.15 * iso +
        0.10 * stress_x_sleep
    )

    sensitivity = 1.0 + 0.12 * repro_flag + 0.08 * sleep_dis
    cortisol_score_0_to_1 = _clamp(base_cort * time_gain * sensitivity, 0.0, 1.0)
    cortisol_load = round(100.0 * cortisol_score_0_to_1, 1)

    # 2) Cycle variability (days)
    cycle_instability = (
        0.30 * stress +
        0.25 * sleep_def +
        0.15 * iso +
        0.10 * r +
        0.10 * irr_hist +
        0.10 * repro_flag
    )
    cycle_instability += 0.08 * stress_x_sleep + 0.05 * iso_x_stress
    cycle_instability = _clamp(cycle_instability, 0.0, 1.0)

    cycle_variability_days = round(0.5 + 9.5 * cycle_instability, 2)

    # 3) Bone loss rate (% per month)
    bone_risk_internal = (
        0.55 * g +
        0.20 * rad_x_grav +
        0.10 * bmd_conc +
        0.08 * sleep_def +
        0.07 * stress
    )
    bone_risk_internal = _clamp(bone_risk_internal, 0.0, 1.0)
    bone_loss_pct_per_month = round(0.2 + 2.0 * bone_risk_internal, 2)

    # Drivers (explainability)
    drivers: Dict[str, str] = {}

    if sleep_def >= stress and sleep_def >= iso:
        drivers["cortisol"] = "Primary driver: cumulative sleep deficit."
    elif stress >= sleep_def and stress >= iso:
        drivers["cortisol"] = "Primary driver: elevated workload/stress."
    else:
        drivers["cortisol"] = "Primary driver: isolation load."

    if stress_x_sleep > 0.25:
        drivers["cycle"] = "Primary driver: stress × sleep interaction."
    elif irr_hist > 0.4 or repro_flag > 0.0:
        drivers["cycle"] = "Primary driver: pre-flight cycle sensitivity."
    else:
        drivers["cycle"] = "Primary driver: circadian disruption proxy."

    if g > 0.5:
        drivers["bone"] = "Primary driver: microgravity exposure duration."
    else:
        drivers["bone"] = "Primary driver: baseline bone density concerns."

    return Prediction(
        mission_day=mission_day,
        cortisol_load=cortisol_load,
        cycle_variability_days=cycle_variability_days,
        bone_loss_pct_per_month=bone_loss_pct_per_month,
        drivers=drivers,
    )


# -------------------------
# Endpoint-style wrappers
# -------------------------

def predict_features(features: Dict[str, float]) -> Dict[str, Any]:
    """
    Engineered features -> outputs (mirrors an Azure ML inference script).
    Risk is attached consistently via risk_scores.attach_risk_breakdown().
    """
    pred = predict_from_features(features)
    outputs = {
        "mission_day": pred.mission_day,
        "cortisol_load": pred.cortisol_load,
        "cycle_variability_days": pred.cycle_variability_days,
        "bone_loss_pct_per_month": pred.bone_loss_pct_per_month,
        "drivers": pred.drivers,
        # risk_overall computed in risk_scores for consistency
    }
    return attach_risk_breakdown(outputs)


def predict_raw(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Raw mission inputs -> engineered features -> outputs.
    This is your end-to-end 'online endpoint' function.
    """
    feats = build_features(payload)
    outputs = predict_features(feats)
    return {"inputs": payload, "features": feats, "outputs": outputs}


# -------------------------
# Local demo
# -------------------------

if __name__ == "__main__":
    payload = {
        "mission": {
            "mission_day": 87,
            "gravity": "microgravity",
            "radiation_level": "high",
            "isolation_level": "high",
            "stress_level": 0.72,
            "sleep_hours_last_72h": 16.5
        },
        "history": {
            "cycle_irregularity_history": "moderate",
            "bone_density_concerns": "some",
            "sleep_disorders": "circadian",
            "prior_pcos_or_endometriosis": False
        }
    }

    res = predict_raw(payload)

    print("\n--- FEATURES ---")
    for k, v in res["features"].items():
        print(f"{k:28s} {v}")

    print("\n--- OUTPUTS ---")
    for k, v in res["outputs"].items():
        print(f"{k}: {v}")

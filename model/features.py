"""
FEATURES
Converts mission conditions + user context into normalized numerical features
for downstream prediction. Designed to mirror an Azure ML preprocessing step.

Inputs are validated and mapped into a consistent numeric feature vector.
"""

from __future__ import annotations
from typing import Dict, Any



# -- helpers --

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _as_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _as_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def _as_bool(x: Any, default: bool = False) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        v = x.strip().lower()
        if v in {"true", "yes", "y", "1"}:
            return True
        if v in {"false", "no", "n", "0"}:
            return False
    if isinstance(x, (int, float)):
        return bool(x)
    return default


# -- encoders -- 


def encode_gravity(gravity: str) -> float:
    """
    gravity: "earth" or "microgravity" (defaults to earth)
    """
    g = (gravity or "earth").strip().lower()
    return 1.0 if g in {"microgravity", "micro", "mg"} else 0.0


def encode_radiation(level: str) -> float:
    """
    radiation_level: "low" | "medium" | "high"
    Values are scaled 0..1.
    """
    m = {
        "low": 0.3,
        "medium": 0.6,
        "high": 1.0,       #scaled proxy for MVP, calibrated later using analog datasets
    }
    key = (level or "medium").strip().lower()
    return float(m.get(key, 0.6))


def encode_isolation(level: str) -> float:
    """
    isolation_level: "low" | "medium" | "high"
    """
    m = {
        "low": 0.2,
        "medium": 0.6,
        "high": 1.0,
    }
    key = (level or "medium").strip().lower()
    return float(m.get(key, 0.6))


def sleep_deficit_72h(sleep_hours_last_72h: float) -> float:
    """
    Normalized sleep deficit (0..1) vs. an 8h/night baseline over 72h.
    """
    optimal = 24.0  # 8h/night * 3 nights
    slept = _clamp(_as_float(sleep_hours_last_72h, 24.0), 0.0, 72.0)
    deficit = _clamp((optimal - slept) / optimal, 0.0, 1.0)
    return round(deficit, 3)


def encode_cycle_irregularity_history(history: str) -> float:
    """
    cycle_irregularity_history:
      "none" | "mild" | "moderate" | "severe"
    """
    m = {
        "none": 0.0,
        "mild": 0.33,
        "moderate": 0.66,
        "severe": 1.0,
    }
    key = (history or "none").strip().lower()
    return float(m.get(key, 0.0))


def encode_bone_density_concerns(concerns: str) -> float:
    """
    bone_density_concerns:
      "none" | "some" | "high"
    """
    m = {
        "none": 0.0,
        "some": 0.5,
        "high": 1.0,
    }
    key = (concerns or "none").strip().lower()
    return float(m.get(key, 0.0))


def encode_sleep_disorders(disorders: str) -> float:
    """
    sleep_disorders:
      "none" | "insomnia" | "circadian" | "other"
    """
    m = {
        "none": 0.0,
        "insomnia": 0.8,
        "circadian": 0.7,
        "other": 0.5,
    }
    key = (disorders or "none").strip().lower()
    return float(m.get(key, 0.0))


# -- main feature builder --

def build_features(payload: Dict[str, Any]) -> Dict[str, float]:
    """
    Build a model-ready feature vector.

    Expected payload structure (flexible, but recommended):

    {
      "mission": {
        "mission_day": 42,
        "gravity": "microgravity",
        "radiation_level": "medium",
        "stress_level": 0.65,             # 0..1
        "isolation_level": "high",
        "sleep_hours_last_72h": 18.0
      },
      "history": {
        "cycle_irregularity_history": "moderate",
        "bone_density_concerns": "some",
        "sleep_disorders": "circadian",
        "prior_pcos_or_endometriosis": false
      }
    }

    Returns: dict[str, float] numeric features.
    """
    mission = payload.get("mission", {}) if isinstance(payload, dict) else {}
    history = payload.get("history", {}) if isinstance(payload, dict) else {}

    # Mission features
    mission_day = _clamp(_as_int(mission.get("mission_day"), 0), 0, 10_000)
    gravity_factor = encode_gravity(str(mission.get("gravity", "earth")))
    radiation_factor = encode_radiation(str(mission.get("radiation_level", "medium")))
    isolation_factor = encode_isolation(str(mission.get("isolation_level", "medium")))

    stress = _clamp(_as_float(mission.get("stress_level"), 0.5), 0.0, 1.0)
    sleep_deficit = sleep_deficit_72h(mission.get("sleep_hours_last_72h", 24))

    # Health history features
    cycle_hist = encode_cycle_irregularity_history(
        str(history.get("cycle_irregularity_history", "none"))
    )
    bone_hist = encode_bone_density_concerns(
        str(history.get("bone_density_concerns", "none"))
    )
    sleep_hist = encode_sleep_disorders(
        str(history.get("sleep_disorders", "none"))
    )
    pcos_endo_flag = 1.0 if _as_bool(history.get("prior_pcos_or_endometriosis"), False) else 0.0

    # Simple interaction terms (helps a basic model capture combined effects)
    stress_x_sleep = round(stress * sleep_deficit, 3)
    rad_x_grav = round(radiation_factor * gravity_factor, 3)
    iso_x_stress = round(isolation_factor * stress, 3)

    features: Dict[str, float] = {
        # mission
        "mission_day": float(mission_day),
        "gravity_factor": gravity_factor,
        "radiation_factor": radiation_factor,
        "isolation_factor": isolation_factor,
        "stress_score": round(stress, 3),
        "sleep_deficit": sleep_deficit,

        # history
        "cycle_irregularity_history": cycle_hist,
        "bone_density_concerns": bone_hist,
        "sleep_disorders": sleep_hist,
        "pcos_or_endometriosis_flag": pcos_endo_flag,

        # interactions
        "stress_x_sleep": stress_x_sleep,
        "rad_x_grav": rad_x_grav,
        "iso_x_stress": iso_x_stress,
    }

    return features


# -- quick local test --

if __name__ == "__main__":
    sample = {
        "mission": {
            "mission_day": 60,
            "gravity": "microgravity",
            "radiation_level": "high",
            "stress_level": 0.72,
            "isolation_level": "high",
            "sleep_hours_last_72h": 16.5
        },
        "history": {
            "cycle_irregularity_history": "moderate",
            "bone_density_concerns": "some",
            "sleep_disorders": "circadian",
            "prior_pcos_or_endometriosis": False
        }
    }

    feats = build_features(sample)
    for k, v in feats.items():
        print(f"{k:28s} {v}")

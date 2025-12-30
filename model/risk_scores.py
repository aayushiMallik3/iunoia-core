"""
RISK SCORES
Turns model outputs into consistent risk scores + buckets.
This keeps risk logic centralized and easy to modify.

Inputs: outputs dict from predictor (cortisol_load, cycle_variability_days, bone_loss_pct_per_month)
Outputs: same dict + risk_score (0..1), risk_overall (Low/Moderate/High), risk_breakdown.
"""

from __future__ import annotations
from typing import Dict, Any


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _bucket(score: float) -> str:
    if score < 0.35:
        return "Low"
    if score < 0.70:
        return "Moderate"
    return "High"


def cortisol_risk(cortisol_load: float) -> float:
    """
    cortisol_load is 0..100.
    Map to 0..1 (simple linear for MVP).
    """
    return _clamp(float(cortisol_load) / 100.0, 0.0, 1.0)


def cycle_risk(cycle_variability_days: float) -> float:
    """
    cycle_variability_days ~ 0.5..10 in your simulator.
    Map 0..10 days variability to 0..1.
    """
    return _clamp(float(cycle_variability_days) / 10.0, 0.0, 1.0)


def bone_risk(bone_loss_pct_per_month: float) -> float:
    """
    bone loss MVP range: ~0.2..2.2 %/month
    We normalize within that window.
    """
    x = float(bone_loss_pct_per_month)
    return _clamp((x - 0.2) / 2.0, 0.0, 1.0)


def overall_risk_from_outputs(outputs: Dict[str, Any]) -> float:
    """
    Weighted composite risk score (0..1).
    Tune weights later when you have real validation data.
    """
    c = cortisol_risk(outputs.get("cortisol_load", 0.0))
    y = cycle_risk(outputs.get("cycle_variability_days", 0.0))
    b = bone_risk(outputs.get("bone_loss_pct_per_month", 0.2))

    # Weights: cortisol 0.45, cycle 0.30, bone 0.25
    score = 0.45 * c + 0.30 * y + 0.25 * b
    return _clamp(score, 0.0, 1.0)


def attach_risk_breakdown(outputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mutates a copy of outputs by adding:
      - risk_score (0..1)
      - risk_overall (Low/Moderate/High)
      - risk_breakdown {cortisol, cycle, bone} each with score + bucket
    """
    out = dict(outputs)

    c = cortisol_risk(out.get("cortisol_load", 0.0))
    y = cycle_risk(out.get("cycle_variability_days", 0.0))
    b = bone_risk(out.get("bone_loss_pct_per_month", 0.2))

    risk_score = overall_risk_from_outputs(out)

    out["risk_score"] = round(risk_score, 2)
    out["risk_overall"] = _bucket(risk_score)
    out["risk_breakdown"] = {
        "cortisol": {"score": round(c, 3), "bucket": _bucket(c)},
        "cycle": {"score": round(y, 3), "bucket": _bucket(y)},
        "bone": {"score": round(b, 3), "bucket": _bucket(b)},
    }

    return out


if __name__ == "__main__":
    # quick sanity test
    sample = {
        "cortisol_load": 34.8,
        "cycle_variability_days": 6.81,
        "bone_loss_pct_per_month": 1.95,
    }
    print(attach_risk_breakdown(sample))

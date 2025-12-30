"""
EXPLANATION
Turns model outputs into human-friendly insight text + UI-ready cards.

Input: the *outputs* dict returned by predictor.predict_raw(...)[\"outputs\"]
(or predictor.predict_features(...)).

Output: a structured explanation object you can render in Figma as:
- headline insight
- key metric chips
- risk breakdown cards
- suggested next steps
"""

from __future__ import annotations

from typing import Dict, Any, List
import json
from pathlib import Path


def _load_constants() -> Dict[str, Any]:
    """
    Loads data/constants.json regardless of where this file is executed from.
    """
    root = Path(__file__).resolve().parents[1]  # iunoia-core/
    constants_path = root / "data" / "constants.json"
    with open(constants_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pct(x_0_to_1: float) -> int:
    return int(round(100.0 * float(x_0_to_1)))


def _fmt(x: float, digits: int = 2) -> str:
    return f"{float(x):.{digits}f}"


def build_explanation(outputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert outputs into an explainable, UI-ready bundle.
    """
    C = _load_constants()

    mission_day = outputs.get("mission_day", None)
    cortisol_load = float(outputs.get("cortisol_load", 0.0))
    cycle_var = float(outputs.get("cycle_variability_days", 0.0))
    bone_loss = float(outputs.get("bone_loss_pct_per_month", 0.0))

    risk_overall = outputs.get("risk_overall", "Unknown")
    risk_score = float(outputs.get("risk_score", 0.0))

    drivers = outputs.get("drivers", {}) or {}
    risk_breakdown = outputs.get("risk_breakdown", {}) or {}

    # --- Pick "current insight" focus ---
    # Simple rule: highlight whichever subsystem has the highest risk score in breakdown.
    focus = "cortisol"
    if isinstance(risk_breakdown, dict) and risk_breakdown:
        focus = max(
            risk_breakdown.keys(),
            key=lambda k: float(risk_breakdown.get(k, {}).get("score", 0.0))
        )

    templates = C.get("insight_templates", {})
    mito = C.get("mitigation_suggestions", {})

    # --- Build the main insight paragraph ---
    if focus == "cortisol":
        pct = _pct(cortisol_load / C["baselines"]["cortisol_load_scale_max"])
        primary = templates["cortisol"]["primary"].format(pct=pct)
        driver = drivers.get("cortisol", templates["cortisol"]["driver_sleep"])
        actions = mito.get("sleep_recovery", [])
        chip = {"label": "Cortisol Load", "value": f"{int(round(cortisol_load))}/100"}
    elif focus == "cycle":
        primary = templates["cycle"]["primary"].format(days=_fmt(cycle_var, 2))
        driver = drivers.get("cycle", templates["cycle"]["driver_circadian"])
        actions = mito.get("stress_downshift", [])
        chip = {"label": "Cycle Variability", "value": f"±{_fmt(cycle_var, 2)} days"}
    else:
        primary = templates["bone"]["primary"].format(pct=_fmt(bone_loss, 2))
        driver = drivers.get("bone", templates["bone"]["driver_microgravity"])
        actions = mito.get("bone_countermeasures", [])
        chip = {"label": "Bone Loss Rate", "value": f"{_fmt(bone_loss, 2)}% / month"}

    current_insight = f"{primary} {driver}"

    # --- Risk cards for UI ---
    cards: List[Dict[str, Any]] = []
    for k in ["cortisol", "cycle", "bone"]:
        sub = risk_breakdown.get(k, {})
        cards.append(
            {
                "key": k,
                "title": k.capitalize(),
                "score": float(sub.get("score", 0.0)),
                "bucket": sub.get("bucket", "Unknown"),
                "driver": drivers.get(k, ""),
            }
        )

    return {
        "mission_day": mission_day,
        "headline": "Analysis",
        "subheadline": f"Mission Day {mission_day}" if mission_day is not None else "Mission Day",
        "current_insight": current_insight,
        "chips": [
            chip,
            {"label": "Overall Risk", "value": str(risk_overall)},
            {"label": "Risk Score", "value": _fmt(risk_score, 2)}
        ],
        "risk_cards": cards,
        "next_steps": actions[:3],  # keep it tight for MVP UI
        "debug": {
            "raw_outputs": outputs
        }
    }


if __name__ == "__main__":
    # Quick local test (paste outputs from predictor here if you want)
    sample_outputs = {
        "mission_day": 87,
        "cortisol_load": 34.8,
        "cycle_variability_days": 6.81,
        "bone_loss_pct_per_month": 1.95,
        "drivers": {
            "cortisol": "Primary driver: isolation load.",
            "cycle": "Primary driver: pre-flight cycle sensitivity.",
            "bone": "Primary driver: microgravity exposure duration."
        },
        "risk_score": 0.58,
        "risk_overall": "Moderate",
        "risk_breakdown": {
            "cortisol": {"score": 0.348, "bucket": "Low"},
            "cycle": {"score": 0.681, "bucket": "Moderate"},
            "bone": {"score": 0.875, "bucket": "High"}
        }
    }

    explanation = build_explanation(sample_outputs)
    print(json.dumps(explanation, indent=2))

# Iunoia Core

Iunoia is a modular AI health modeling system that predicts and explains menstrual and hormonal health risks in long-duration space missions by modeling the combined effects of microgravity, radiation, isolation, stress, and sleep disruption.

It is designed as an explainable, end-to-end inference pipeline that converts raw mission conditions into interpretable health projections and risk scores. The current version is a deterministic, physics-inspired simulator built for rapid prototyping, judging clarity, and future integration with Azure ML.

---

## What Iunoia Does

Iunoia takes mission conditions and individual health history and produces:

- **Cortisol Load** (0–100 scale)
- **Cycle Variability** (absolute deviation in days)
- **Bone Loss Rate** (% BMD loss per month)
- **Overall Risk Score** (0–1) with bucket (*Low / Moderate / High*)
- **Risk Breakdown** by subsystem (cortisol, cycle, bone)
- **Human-readable explanations** describing why each risk is elevated

The system is designed to be:

- Evidence-informed
- Numerically stable and testable
- Explainable for judges and non-technical stakeholders
- Azure-ready (the simulator can be swapped for a trained ML model later)

---

## System Architecture

sample_input.json
↓
build_features (features.py)
↓
predictor (physiology simulator)
↓
risk_scores (normalization + buckets)
↓
explanation (UI-ready insights)


Each stage is modular and independently replaceable.

---

## Project Structure

iunoia-core/
├── model/
│ ├── init.py # Package exports
│ ├── features.py # Feature engineering
│ ├── predictor.py # Physiology simulator
│ └── risk_scores.py # Risk normalization + buckets
│
├── explain/
│ └── explanation.py # Human-readable explanations
│
├── data/
│ └── constants.json # Tunable model constants
│
├── demo/
│ └── sample_input.json # Example mission input
│
├── README.md
└── requirements.txt



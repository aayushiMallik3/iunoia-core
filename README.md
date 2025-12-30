Iunoia Core

Iunoia is a modular AI health modeling system that predicts and explains menstrual and hormonal health risks in space missions by modeling the combined effects of microgravity, radiation, isolation, stress, and sleep disruption.

It is designed as an explainable, end-to-end inference pipeline that converts raw mission conditions into interpretable health projections and risk scores. The current version is a deterministic, physics-inspired simulator built for rapid prototyping, judging clarity, and future integration with Azure ML.

What Iunoia Does

Iunoia takes mission conditions and individual health history and produces:

Cortisol Load (0–100 scale)

Cycle Variability (absolute deviation in days)

Bone Loss Rate (% BMD loss per month)

Overall Risk Score (0–1) with bucket (Low / Moderate / High)

Risk Breakdown by subsystem (cortisol, cycle, bone)

Human-readable explanations describing why each risk is elevated

The system is designed to be:

Evidence-informed

Numerically stable and testable

Explainable for judges and non-technical stakeholders

Azure-ready (simulator can be swapped for a trained ML model later)

Architecture (End-to-End)
sample_input.json
        ↓
build_features (features.py)
        ↓
predictor (physiology simulator)
        ↓
risk_scores (normalization + buckets)
        ↓
explanation (UI-ready insights)


This mirrors how an Azure ML online endpoint would be structured:
preprocessing → inference → post-processing → explanation.

How to Run (Local)
1. Install dependencies
pip install -r requirements.txt

2. Run the model on sample input
python -c "from model import predict_raw; import json; \
print(json.dumps(predict_raw(json.load(open('demo/sample_input.json'))), indent=2))"

Expected Output

You should see a structured JSON object containing:

engineered features

physiological projections

risk score and risk buckets

subsystem-level drivers

Example (truncated):

{
  "outputs": {
    "cortisol_load": 34.8,
    "cycle_variability_days": 6.81,
    "bone_loss_pct_per_month": 1.95,
    "risk_score": 0.58,
    "risk_overall": "Moderate",
    "risk_breakdown": {
      "cortisol": { "score": 0.348, "bucket": "Low" },
      "cycle": { "score": 0.681, "bucket": "Moderate" },
      "bone": { "score": 0.875, "bucket": "High" }
    }
  }
}

3. Generate explanations (optional)
python explain/explanation.py


This converts model outputs into UI-ready insight text and cards.

Who This Is For

Space health and human performance research

Early-stage mission design and risk exploration

Hackathons, demos, and technical judging

Educators and students working at the intersection of:

Aerospace

Women’s health

AI / ML

Human-centered system design

Roadmap

Replace simulator with trained Azure ML model

Calibrate weights using analog astronaut and ISS datasets

Add temporal modeling across mission duration

Integrate directly with UI dashboards

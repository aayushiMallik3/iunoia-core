Iunoia is a modular AI health modeling system that predicts and explains menstrual and hormonal health risks in space missions by modeling the combined effects of microgravity, radiation, isolation, stress, and sleep disruption.

It is designed as an explainable, end-to-end inference pipeline that converts raw mission conditions into interpretable health projections and risk scores.
The current version is a deterministic, physics-inspired simulator built for rapid prototyping, judging clarity, and future integration with Azure ML.

What Iunoia Does
Iunoia takes mission conditions + individual health history and produces:

- Cortisol Load (0–100 scale)
- Cycle Variability (absolute deviation in days)
- Bone Loss Rate (% BMD loss per month)
- Overall Risk Score (0–1) and bucket (Low / Moderate / High)
- Risk Breakdown by subsystem (cortisol, cycle, bone)
- Human-readable explanations of why each risk is elevated

The system is built to be:
- Evidence-informed
- Numerically stable and testable
- Explainable (judges + non-technical stakeholders)
- Azure-ready (swap simulator → trained ML model later)

Who This Is For
- Space health & human performance research
- Early-stage mission design and risk exploration
- Hackathons, demos, and technical judging
- Educators and students working at the intersection of:
    - Aerospace
    - Women’s health
    - AI / ML
    - Human-centered system design


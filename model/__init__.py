"""
IUNOIA Model Package

Core modeling components for the Iunoia AI system.

Exposes clean imports for:
- feature engineering (features.py)
- physiological simulation (predictor.py)
- risk interpretation/scoring (risk_scores.py)
"""

from .features import build_features
from .predictor import predict_raw, predict_features, predict_from_features
from .risk_scores import attach_risk_breakdown

__all__ = [
    "build_features",
    "predict_raw",
    "predict_features",
    "predict_from_features",
    "attach_risk_breakdown",
]

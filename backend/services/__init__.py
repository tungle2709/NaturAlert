"""
Backend Services Module

This module contains core services for the Disaster Early Warning System:
- PredictionEngine: ML-based disaster prediction
- GeminiService: AI-powered explanations (to be implemented)
- AlertManager: Alert triggering and notifications (to be implemented)
"""

from .prediction_engine import PredictionEngine

__all__ = ['PredictionEngine']

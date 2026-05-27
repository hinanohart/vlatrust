"""vlatrust core — simulator-independent data model and math.

Depends only on numpy + the standard library. Importing this package never
imports torch, a simulator, or any policy weights.
"""

from .types import (
    MODALITIES,
    CalibrationResult,
    CollapseCurve,
    CollapsePoint,
    ConfidenceSource,
    ConformalResult,
    FragilityReport,
    Perturbation,
    ReliabilityGap,
    Scorecard,
    Step,
    Trace,
    TraceSet,
)

__all__ = [
    "MODALITIES",
    "CalibrationResult",
    "CollapseCurve",
    "CollapsePoint",
    "ConfidenceSource",
    "ConformalResult",
    "FragilityReport",
    "Perturbation",
    "ReliabilityGap",
    "Scorecard",
    "Step",
    "Trace",
    "TraceSet",
]

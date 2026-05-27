"""Core data model for vlatrust — the simulator-independent vocabulary.

Everything downstream (perturbation, conformal, reliability, collapse, score)
operates on these frozen value types and nothing else. They depend only on
numpy + the standard library, so the entire core is CPU-testable with no
simulator, GPU, or policy weights present.

Design rules:
  * All inputs and all result containers live here, so the math modules import
    *from* this module and never the reverse (no import cycle).
  * Dataclasses are ``frozen`` (immutable). ``Step.action`` is normalised to a
    read-only ``numpy`` array so a trace cannot be mutated after construction.
  * Confidence is explicitly typed by source. A policy family that exposes no
    usable confidence yields :data:`ConfidenceSource.NONE`, and the abstention
    axis then returns ``N/A`` rather than fabricating a number (fail-closed).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Any

import numpy as np

__all__ = [
    "ConfidenceSource",
    "MODALITIES",
    "Step",
    "Perturbation",
    "Trace",
    "TraceSet",
    "ConformalResult",
    "ReliabilityGap",
    "CollapsePoint",
    "CollapseCurve",
    "FragilityReport",
    "CalibrationResult",
    "Scorecard",
]


class ConfidenceSource(StrEnum):
    """Where a per-step confidence value comes from, per policy family.

    The Trust-Shift CLAIM is made only for :data:`TOKEN_ENTROPY` (Tier-A).
    :data:`SAMPLING_VARIANCE` (Tier-B, flow-matching) is opt-in and non-claim.
    :data:`NONE` means no usable confidence exists; the abstention axis returns
    ``N/A`` instead of a fabricated value (fail-closed).
    """

    TOKEN_ENTROPY = "token_entropy"
    SAMPLING_VARIANCE = "sampling_variance"
    NONE = "none"


#: Canonical perturbation modality tags. Open set, but these are the names the
#: built-in injector and Mondrian grouping use.
MODALITIES: tuple[str, ...] = (
    "language",
    "vision",
    "init_state",
    "sensor_noise",
    "dynamics",
    "camera",
    "actuation",
)


@dataclass(frozen=True, slots=True, eq=False)
class Step:
    """One timestep of a recorded trajectory.

    ``action`` is required; everything else is optional and may be ``None`` when
    the recording did not capture it. Nonconformity scoring degrades gracefully:
    if neither ``neg_log_prob`` nor ``action_residual`` is present the step is
    treated as having an undefined nonconformity (which the OOD gate maps to a
    fail-closed ABSTAIN).
    """

    action: np.ndarray
    confidence: float | None = None  # native policy confidence in [0, 1]
    neg_log_prob: float | None = None  # -log p_theta(a_t | o_t) if available
    action_residual: float | None = None  # ||a_t - a_ref|| if a reference exists
    physically_valid: bool = True  # False => joint-limit / dynamics violation

    def __post_init__(self) -> None:
        a = np.asarray(self.action, dtype=float).reshape(-1)
        a.setflags(write=False)
        object.__setattr__(self, "action", a)
        if self.confidence is not None and not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence!r}")

    @property
    def action_dim(self) -> int:
        return int(self.action.shape[0])


@dataclass(frozen=True, slots=True)
class Perturbation:
    """A named, seeded, intensity-parameterised shift applied to a TraceSet.

    ``intensity`` (tau) is in ``[0, 1]``; ``0.0`` denotes the identity (clean).
    ``params`` is a frozen key/value tuple so the perturbation stays hashable
    and reproducible. Composition and seed-derivation live in
    :mod:`vlatrust.core.perturb.algebra`.
    """

    name: str
    modality: str
    intensity: float = 0.0
    seed: int = 0
    reversible: bool = True
    params: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if not (0.0 <= float(self.intensity) <= 1.0):
            raise ValueError(f"intensity must be in [0, 1], got {self.intensity!r}")
        # normalise params to a sorted tuple for deterministic identity
        object.__setattr__(self, "params", tuple(sorted(self.params)))

    @property
    def is_identity(self) -> bool:
        return self.intensity == 0.0


@dataclass(frozen=True, slots=True)
class Trace:
    """One rollout: an ordered sequence of steps plus task-level metadata.

    ``perturbation is None`` (or an identity perturbation) marks a clean/nominal
    rollout. ``success`` is the task outcome label; ``None`` means unknown, which
    forces reliability metrics into an explicitly "uncalibrated" mode rather than
    assuming success.
    """

    steps: tuple[Step, ...]
    confidence_source: ConfidenceSource = ConfidenceSource.NONE
    success: bool | None = None
    perturbation: Perturbation | None = None
    task: str | None = None
    trace_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", tuple(self.steps))

    @property
    def horizon(self) -> int:
        return len(self.steps)

    @property
    def is_clean(self) -> bool:
        return self.perturbation is None or self.perturbation.is_identity

    @property
    def modality(self) -> str | None:
        return None if self.perturbation is None else self.perturbation.modality

    @property
    def intensity(self) -> float:
        return 0.0 if self.perturbation is None else self.perturbation.intensity


@dataclass(frozen=True, slots=True)
class TraceSet:
    """An immutable collection of traces with light, allocation-cheap views.

    Filtering returns new ``TraceSet`` views (the underlying ``Trace`` objects
    are shared, never copied). All heavier I/O, per-trajectory splitting, and
    exchangeability helpers live in :mod:`vlatrust.core.trace`.
    """

    traces: tuple[Trace, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "traces", tuple(self.traces))

    def __len__(self) -> int:
        return len(self.traces)

    def __iter__(self):
        return iter(self.traces)

    def __getitem__(self, i):
        return self.traces[i]

    def filter(self, predicate) -> TraceSet:
        return TraceSet(tuple(t for t in self.traces if predicate(t)))

    def clean(self) -> TraceSet:
        return self.filter(lambda t: t.is_clean)

    def perturbed(self) -> TraceSet:
        return self.filter(lambda t: not t.is_clean)

    def by_modality(self, modality: str) -> TraceSet:
        return self.filter(lambda t: t.modality == modality)

    def by_perturbation(self, name: str) -> TraceSet:
        return self.filter(lambda t: t.perturbation is not None and t.perturbation.name == name)

    def at_intensity(self, intensity: float, *, atol: float = 1e-9) -> TraceSet:
        return self.filter(lambda t: abs(t.intensity - intensity) <= atol)

    @property
    def modalities(self) -> tuple[str, ...]:
        seen: list[str] = []
        for t in self.traces:
            m = t.modality
            if m is not None and m not in seen:
                seen.append(m)
        return tuple(seen)

    @property
    def confidence_sources(self) -> frozenset[ConfidenceSource]:
        return frozenset(t.confidence_source for t in self.traces)

    @property
    def intensities(self) -> tuple[float, ...]:
        return tuple(sorted({t.intensity for t in self.traces}))

    def success_rate(self) -> float | None:
        """Fraction of traces with ``success is True``.

        Returns ``None`` if no trace carries a success label (so callers must
        surface "uncalibrated" rather than silently treating unknown as failure).
        """
        labelled = [t.success for t in self.traces if t.success is not None]
        if not labelled:
            return None
        return float(np.mean([1.0 if s else 0.0 for s in labelled]))


# --------------------------------------------------------------------------- #
# Result containers (produced by the math modules, consumed by report/).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ConformalResult:
    """Split-conformal calibration outcome at level ``alpha``.

    ``coverage`` is the empirical fraction of test traces whose sequence-level
    nonconformity falls at or below ``q_hat``. ``per_group`` holds Mondrian
    (per-modality) coverages when grouping was requested.
    """

    alpha: float
    q_hat: float
    coverage: float
    n_calib: int
    n_test: int
    abstention_rate: float
    beta: float  # spike-preserving aggregation quantile
    weighted: bool = False
    per_group: tuple[tuple[str, float], ...] = ()  # (modality, coverage)


@dataclass(frozen=True, slots=True)
class ReliabilityGap:
    """Sim-vs-real reliability gaps with bootstrap confidence intervals.

    ``delta_succ`` = SR_sim - SR_real. ``delta_cov`` = coverage of a sim-fitted
    conformal threshold when applied to real traces, minus the nominal ``1-alpha``
    (a negative value means "safe in sim, dangerous in real"). A claim is only
    made when the CI excludes zero.
    """

    delta_succ: float | None
    delta_succ_ci: tuple[float, float] | None
    delta_cov: float | None
    delta_cov_ci: tuple[float, float] | None

    @property
    def succ_claimable(self) -> bool:
        ci = self.delta_succ_ci
        return ci is not None and (ci[0] > 0.0 or ci[1] < 0.0)

    @property
    def cov_claimable(self) -> bool:
        ci = self.delta_cov_ci
        return ci is not None and (ci[0] > 0.0 or ci[1] < 0.0)


@dataclass(frozen=True, slots=True)
class CollapsePoint:
    intensity: float
    success_rate: float
    ci_low: float
    ci_high: float
    n: int


@dataclass(frozen=True, slots=True)
class CollapseCurve:
    """Success rate vs. perturbation intensity for one modality.

    ``fragility`` F_m in ``[0, 1]`` is ``1 - AUC_tau(SR(tau)/SR(0))``: 0 = robust,
    1 = total collapse. ``mechanism`` is a coarse tag in
    {``"cliff"``, ``"gradual"``, ``"brittle-at-zero"``, ``"robust"``}.
    """

    modality: str
    points: tuple[CollapsePoint, ...]
    fragility: float
    mechanism: str


@dataclass(frozen=True, slots=True)
class FragilityReport:
    curves: tuple[CollapseCurve, ...]
    most_fragile_modality: str | None
    mean_fragility: float


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Calibration quality of confidence vs. correctness.

    ``ece`` is expected calibration error; ``inverse_brier`` = ``1 - Brier`` is
    the weight used when aggregating multiple sources. ``bins`` holds
    (confidence, accuracy, count) triples after isotonic (PAVA) fitting.
    """

    ece: float
    inverse_brier: float
    n: int
    bins: tuple[tuple[float, float, int], ...] = ()


@dataclass(frozen=True, slots=True)
class Scorecard:
    """Top-level result. Either wing (collapse / reliability) may be ``None``.

    ``trust_shift`` is the single headline number in ``[0, 1]`` and is ``None``
    when no token-confidence (Tier-A) signal is available — never fabricated.
    ``hard_valid`` is the multiplicative physics gate: if ``False`` the trust
    score is forced to ``0.0`` regardless of the other axes.
    """

    confidence_source: ConfidenceSource
    trust_shift: float | None = None
    hard_valid: bool = True
    fragility: FragilityReport | None = None
    reliability: ReliabilityGap | None = None
    conformal: ConformalResult | None = None
    calibration: CalibrationResult | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        """JSON-ready nested dict (no numpy types, no pickling)."""
        result = _to_jsonable(self)
        assert isinstance(result, dict)  # a Scorecard is always a dataclass
        return result


def _to_jsonable(obj: object) -> Any:
    """Recursively convert dataclasses / enums / numpy scalars to JSON types."""
    from dataclasses import fields, is_dataclass

    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_jsonable(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (float, np.floating)):
        # Map non-finite to null: RFC-8259 / draft-07 JSON has no Infinity/NaN
        # token. The fail-closed sentinel (q_hat=+inf) and undefined values
        # (nan fragility) survive as ``null``; their meaning is carried by
        # companion fields (abstention_rate=1.0, mechanism="insufficient", the
        # *_claimable flags), so no information is lost.
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return [_to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    return obj

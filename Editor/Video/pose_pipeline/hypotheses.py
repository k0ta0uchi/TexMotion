"""Data structures and ranking helpers for ambiguous pose hypotheses.

The temporal fitter deliberately keeps the two explanations for a self
occlusion alive until all of the available evidence has been evaluated.  This
module contains the small, serialisable value objects used for that purpose;
the actual pose-specific candidate construction lives in :mod:`sequence_fit`.

Scores are *higher is better*.  A score component may therefore be negative
when it represents a penalty.  Keeping the components and provenance beside
the score makes an ambiguous result useful to the editor instead of reducing
it to an unexplained depth edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .observations import UncertaintyInterval


def _json_value(value: Any) -> Any:
    """Convert numpy/dataclass-ish values to values safe for JSON encoding."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True)
class HypothesisScore:
    """A ranked score and its explainable evidence components."""

    name: str
    score: float
    components: Mapping[str, float] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def total_score(self) -> float:
        """Compatibility/readability alias for callers using ``total_score``."""
        return float(self.score)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": str(self.name),
            "score": float(self.score),
            "totalScore": float(self.score),
            "components": _json_value(dict(self.components)),
            "provenance": _json_value(dict(self.provenance)),
        }


@dataclass
class PoseHypothesis:
    """A candidate pose/sequence and the evidence used to produce it."""

    name: str
    pose: Optional[np.ndarray] = None
    score: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_score(self) -> float:
        return float(self.score)

    def to_dict(self, include_pose: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": str(self.name),
            "score": float(self.score),
            "totalScore": float(self.score),
            "components": _json_value(dict(self.components)),
            "provenance": _json_value(dict(self.provenance)),
        }
        if self.pose is not None:
            payload["poseShape"] = list(np.asarray(self.pose).shape)
            if include_pose:
                payload["pose"] = _json_value(np.asarray(self.pose))
        return payload


@dataclass
class HypothesisResult:
    """The selected candidate together with all alternatives and diagnostics."""

    selected: PoseHypothesis
    alternatives: List[PoseHypothesis] = field(default_factory=list)
    scores: List[HypothesisScore] = field(default_factory=list)
    score_gap: float = 0.0
    ambiguity_margin: float = 0.0
    ambiguous: bool = False
    confidence: float = 0.0
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def primary_hypothesis(self) -> str:
        return self.selected.name

    @property
    def alternative_hypotheses(self) -> List[str]:
        return [candidate.name for candidate in self.alternatives]

    @property
    def best_score(self) -> float:
        return float(self.selected.score)

    def to_dict(self, include_pose: bool = False) -> Dict[str, Any]:
        return {
            "selected": self.selected.to_dict(include_pose=include_pose),
            "primaryHypothesis": self.primary_hypothesis,
            "alternatives": [candidate.to_dict(include_pose=include_pose) for candidate in self.alternatives],
            "alternativeHypotheses": self.alternative_hypotheses,
            "scores": [score.to_dict() for score in self.scores],
            "scoreGap": float(self.score_gap),
            "ambiguityMargin": float(self.ambiguity_margin),
            "ambiguous": bool(self.ambiguous),
            "confidence": float(self.confidence),
            "provenance": _json_value(dict(self.provenance)),
        }

    def to_uncertainty_interval(
        self,
        start_frame: int,
        end_frame: int,
        reason: str,
        recommended_action: str = "review_depth_hypothesis",
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> "HypothesisUncertaintyInterval":
        """Build an uncertainty interval retaining this result's alternatives."""
        merged_provenance = dict(self.provenance)
        if provenance:
            merged_provenance.update(dict(provenance))
        return HypothesisUncertaintyInterval(
            start_frame=int(start_frame),
            end_frame=int(end_frame),
            reason=str(reason),
            confidence=float(self.confidence),
            recommended_action=str(recommended_action),
            primary_hypothesis=self.primary_hypothesis,
            alternative_hypotheses=tuple(self.alternative_hypotheses),
            hypothesis_scores=tuple(self.scores),
            provenance=merged_provenance,
            score_gap=float(self.score_gap),
        )


@dataclass(frozen=True)
class HypothesisUncertaintyInterval(UncertaintyInterval):
    """Backward-compatible uncertainty interval enriched with alternatives.

    The class subclasses the legacy ``UncertaintyInterval`` so old consumers
    can continue to use the original fields and ``isinstance`` checks.  New
    consumers can inspect ranked hypotheses and provenance without guessing
    which depth edit was applied.
    """

    primary_hypothesis: str = ""
    alternative_hypotheses: Tuple[str, ...] = ()
    hypothesis_scores: Tuple[HypothesisScore, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    score_gap: float = 0.0

    @property
    def alternatives(self) -> Tuple[str, ...]:
        """Short alias useful to UI adapters."""
        return self.alternative_hypotheses

    def to_dict(self) -> Dict[str, Any]:
        payload = super().to_dict()
        payload.update(
            {
                "primaryHypothesis": str(self.primary_hypothesis),
                "alternativeHypotheses": [str(name) for name in self.alternative_hypotheses],
                "hypothesisScores": [score.to_dict() for score in self.hypothesis_scores],
                "alternatives": [str(name) for name in self.alternative_hypotheses],
                "scoreGap": float(self.score_gap),
                "provenance": _json_value(dict(self.provenance)),
            }
        )
        return payload


# Clear aliases make the value object discoverable for callers that use the
# shorter names while retaining one implementation and one wire format.
ScoredUncertaintyInterval = HypothesisUncertaintyInterval
HypothesisInterval = HypothesisUncertaintyInterval
ScoredHypothesis = PoseHypothesis


def _candidate_from_value(value: Any, index: int) -> PoseHypothesis:
    """Normalise common candidate representations for ``score_hypotheses``."""
    if isinstance(value, PoseHypothesis):
        return PoseHypothesis(
            name=value.name,
            pose=None if value.pose is None else np.asarray(value.pose).copy(),
            score=float(value.score),
            components=dict(value.components),
            provenance=dict(value.provenance),
        )

    if isinstance(value, Mapping):
        name = str(value.get("name", value.get("hypothesis", f"hypothesis_{index}")))
        pose_value = value.get("pose", value.get("candidate_pose"))
        pose = None if pose_value is None else np.asarray(pose_value).copy()
        raw_components = value.get("components", value.get("evidence", {}))
        components = {
            str(key): float(item)
            for key, item in dict(raw_components or {}).items()
            if isinstance(item, (int, float, np.integer, np.floating))
        }
        provenance = dict(value.get("provenance", {}))
        if "evidence" in value and "evidence" not in provenance:
            provenance["evidence"] = _json_value(value["evidence"])
        explicit_score = value.get("score", value.get("total_score"))
        if explicit_score is None:
            score = float(sum(components.values()))
        else:
            score = float(explicit_score)
        return PoseHypothesis(name=name, pose=pose, score=score, components=components, provenance=provenance)

    # A tuple/list is accepted as ``(name, pose, evidence)`` for lightweight
    # integrations and makes the helper pleasant to use outside the fitter.
    if isinstance(value, (tuple, list)) and value:
        name = str(value[0])
        pose = None if len(value) < 2 or value[1] is None else np.asarray(value[1]).copy()
        evidence = dict(value[2]) if len(value) >= 3 and isinstance(value[2], Mapping) else {}
        score = float(sum(float(item) for item in evidence.values() if isinstance(item, (int, float))))
        return PoseHypothesis(
            name=name,
            pose=pose,
            score=score,
            components={str(key): float(item) for key, item in evidence.items()},
            provenance={"evidence": _json_value(evidence)},
        )

    raise TypeError(f"Unsupported hypothesis candidate at index {index}: {type(value)!r}")


def score_hypotheses(
    candidates: Iterable[Any],
    ambiguity_margin: float = 0.05,
    lower_is_better: bool = False,
    provenance: Optional[Mapping[str, Any]] = None,
) -> HypothesisResult:
    """Rank candidates and report whether the best explanation is ambiguous.

    ``candidates`` may contain :class:`PoseHypothesis` instances or mappings
    with ``name``, ``pose``, ``evidence``/``components`` and optional
    ``provenance`` fields.  The first candidate wins exact ties, providing a
    deterministic provisional output while ``ambiguous`` and ``alternatives``
    make the unresolved depth explicit.
    """
    normalised = [_candidate_from_value(candidate, index) for index, candidate in enumerate(candidates)]
    if not normalised:
        raise ValueError("At least one hypothesis candidate is required")

    # A candidate's provenance always carries its components, so a consumer
    # can audit a result even when a caller supplied only an explicit score.
    for candidate in normalised:
        candidate.provenance.setdefault("components", dict(candidate.components))

    if lower_is_better:
        ordered = sorted(enumerate(normalised), key=lambda item: (item[1].score, item[0]))
    else:
        ordered = sorted(enumerate(normalised), key=lambda item: (-item[1].score, item[0]))

    best = ordered[0][1]
    second = ordered[1][1] if len(ordered) > 1 else None
    score_gap = 0.0 if second is None else abs(float(best.score) - float(second.score))
    margin = max(0.0, float(ambiguity_margin))
    ambiguous = second is not None and score_gap <= margin

    # Confidence is intentionally conservative for ties.  It is a confidence
    # in the selected *explanation*, not a detector confidence.
    scale = max(1.0, abs(float(best.score)), abs(float(second.score)) if second is not None else 0.0)
    relative_gap = score_gap / scale
    confidence = float(np.clip(0.35 + 0.65 * relative_gap, 0.35, 0.99))
    if second is None:
        confidence = 0.99

    selected = best
    alternatives = [item[1] for item in ordered[1:]]
    score_objects = [
        HypothesisScore(
            name=item[1].name,
            score=float(item[1].score),
            components=dict(item[1].components),
            provenance=dict(item[1].provenance),
        )
        for item in ordered
    ]
    result_provenance = dict(provenance or {})
    result_provenance.setdefault("candidateCount", len(normalised))
    result_provenance.setdefault("ranking", "lower_is_better" if lower_is_better else "higher_is_better")

    return HypothesisResult(
        selected=selected,
        alternatives=alternatives,
        scores=score_objects,
        score_gap=float(score_gap),
        ambiguity_margin=margin,
        ambiguous=bool(ambiguous),
        confidence=confidence,
        provenance=result_provenance,
    )


# Semantic aliases for integrations that describe the operation as ranking or
# selecting rather than scoring.
rank_hypotheses = score_hypotheses
select_hypothesis = score_hypotheses


__all__ = [
    "HypothesisScore",
    "PoseHypothesis",
    "ScoredHypothesis",
    "HypothesisResult",
    "HypothesisUncertaintyInterval",
    "ScoredUncertaintyInterval",
    "HypothesisInterval",
    "score_hypotheses",
    "rank_hypotheses",
    "select_hypothesis",
]

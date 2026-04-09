"""
Periodic backbone plus dual-signal burst cloud scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..base import Action, CloudScheduler


@dataclass(frozen=True)
class DualSignalPeriodicConfig:
    """Configuration for the periodic dual-signal scheduler."""

    cloud_latency: int
    base_interval: int = 4
    burst_threshold: float = 0.7
    tau: float = 1.0
    age_weight: float = 0.2
    deviation_weight: float = 0.6
    momentum_weight: float = 0.2
    deviation_cap: float = 0.10
    age_horizon_multiplier: int = 2
    force_age_multiplier: int = 3
    min_extra_gap: int = 1
    burst_queue_cap: int = 1
    eps: float = 1e-8
    seed: int | None = None


class DualSignalPeriodicScheduler(CloudScheduler):  # pylint: disable=too-many-instance-attributes
    """Use a periodic refresh cadence with deterministic dual-signal bursts."""

    _DEFAULT_WEIGHTS: tuple[float, float, float] = (0.20, 0.60, 0.20)
    _CALL_REASONS: tuple[str, ...] = ("bootstrap", "backbone", "burst", "force_age", "none")

    def __init__(self, config: DualSignalPeriodicConfig) -> None:
        self.config = config
        self.beta = 1.0 - 1.0 / max(2, int(config.cloud_latency))
        self._prev_edge_features: np.ndarray | None = None
        self._prev_edge_action: np.ndarray | None = None
        self._sigma_e: np.ndarray | None = None
        self._sigma_u: np.ndarray | None = None
        self._momentum = 0.0
        self._last_call_step: int | None = None
        self._weights = self._normalized_weights(
            config.age_weight,
            config.deviation_weight,
            config.momentum_weight,
        )
        self.last_call_reason = "none"
        self.last_risk_score = 0.0
        self.last_terms = {
            "age_term": 0.0,
            "deviation_term": 0.0,
            "momentum_term": 0.0,
        }
        self.call_reason_counts = {reason: 0 for reason in self._CALL_REASONS}

    @classmethod
    def _normalized_weights(
        cls,
        age_weight: float,
        deviation_weight: float,
        momentum_weight: float,
    ) -> tuple[float, float, float]:
        weights = np.asarray(
            [float(age_weight), float(deviation_weight), float(momentum_weight)],
            dtype=float,
        )
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            return cls._DEFAULT_WEIGHTS
        total = float(weights.sum())
        if total <= 0.0:
            return cls._DEFAULT_WEIGHTS
        normalized = weights / total
        return (float(normalized[0]), float(normalized[1]), float(normalized[2]))

    @staticmethod
    def _vector_from_context(
        context: dict[str, Any] | None,
        key: str,
        expected_size: int,
    ) -> np.ndarray | None:
        if context is None or key not in context:
            return None
        value = context.get(key)
        if value is None:
            return None
        vector = np.asarray(value, dtype=float).reshape(-1)
        if vector.size != expected_size:
            return None
        return vector

    @staticmethod
    def _cloud_age(context: dict[str, Any] | None) -> int:
        if context is None:
            return 999
        value = context.get("cloud_age", 999)
        try:
            return int(value)
        except (TypeError, ValueError):
            return 999

    @staticmethod
    def _cloud_queue_depth(context: dict[str, Any] | None) -> int:
        if context is None:
            return 0
        value = context.get("cloud_queue_depth")
        try:
            if value is not None:
                return max(0, int(value))
        except (TypeError, ValueError):
            pass
        return 1 if bool(context.get("cloud_in_flight", False)) else 0

    @staticmethod
    def _deviation(context: dict[str, Any] | None) -> float:
        if context is None:
            return 0.0
        value = context.get("deviation", 0.0)
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return 0.0

    def reset(self) -> None:
        """Reset the scheduler's running state."""
        self._prev_edge_features = None
        self._prev_edge_action = None
        self._sigma_e = None
        self._sigma_u = None
        self._momentum = 0.0
        self._last_call_step = None
        self.last_call_reason = "none"
        self.last_risk_score = 0.0
        self.last_terms = {
            "age_term": 0.0,
            "deviation_term": 0.0,
            "momentum_term": 0.0,
        }
        self.call_reason_counts = {reason: 0 for reason in self._CALL_REASONS}

    def debug_state(self) -> dict[str, Any]:
        """Return scheduler diagnostics for analysis and tests."""
        total_actual_calls = sum(
            count
            for reason, count in self.call_reason_counts.items()
            if reason != "none"
        )
        return {
            "last_call_reason": self.last_call_reason,
            "call_reason_counts": dict(self.call_reason_counts),
            "last_risk_score": float(self.last_risk_score),
            "last_terms": dict(self.last_terms),
            "last_call_step": self._last_call_step,
            "total_actual_calls": total_actual_calls,
        }

    def _initialize_state(self, edge_features: np.ndarray, edge_action: np.ndarray) -> None:
        self._prev_edge_features = edge_features.copy()
        self._prev_edge_action = edge_action.copy()
        self._sigma_e = np.zeros_like(edge_features, dtype=float)
        self._sigma_u = np.zeros_like(edge_action, dtype=float)

    def _record_reason(self, step: int, reason: str) -> bool:
        self.last_call_reason = reason
        self.call_reason_counts[reason] = self.call_reason_counts.get(reason, 0) + 1
        if reason != "none":
            self._last_call_step = step
            return True
        return False

    def _compute_momentum_term(
        self,
        edge_features: np.ndarray,
        edge_action: np.ndarray,
    ) -> float:
        if self._prev_edge_features is None or self._prev_edge_action is None:
            self._initialize_state(edge_features, edge_action)
            return 0.0

        if self._sigma_e is None:
            self._sigma_e = np.zeros_like(edge_features, dtype=float)
        if self._sigma_u is None:
            self._sigma_u = np.zeros_like(edge_action, dtype=float)

        delta_edge = edge_features - self._prev_edge_features
        delta_action = edge_action - self._prev_edge_action

        self._sigma_e = self.beta * self._sigma_e + (1.0 - self.beta) * np.abs(delta_edge)
        self._sigma_u = self.beta * self._sigma_u + (1.0 - self.beta) * np.abs(delta_action)

        instability = float(np.linalg.norm(delta_edge / (self._sigma_e + self.config.eps)))
        effort = float(np.linalg.norm(delta_action / (self._sigma_u + self.config.eps)))
        score = instability * effort
        self._momentum = self.beta * self._momentum + (1.0 - self.beta) * score

        self._prev_edge_features = edge_features.copy()
        self._prev_edge_action = edge_action.copy()

        return float(
            np.clip(
                score / (float(self.config.tau) * (self._momentum + float(self.config.eps))) - 1.0,
                0.0,
                1.0,
            )
        )

    def should_call_cloud(
        self,
        step: int,
        obs: dict[str, Any],
        latest_cloud_action: Action | None,
        context: dict[str, Any] | None = None,
    ) -> bool:
        del obs
        edge_features = self._vector_from_context(context, "edge_features", expected_size=3)
        edge_action = self._vector_from_context(context, "edge_action", expected_size=2)
        cloud_age = self._cloud_age(context)
        queue_depth = self._cloud_queue_depth(context)
        backbone_step = (step % max(1, int(self.config.base_interval))) == 0

        momentum_term = 0.0
        if edge_features is not None and edge_action is not None:
            momentum_term = self._compute_momentum_term(edge_features, edge_action)
        elif latest_cloud_action is None and self._prev_edge_features is None:
            self._momentum = 0.0

        age_denom = max(
            1.0,
            float(int(self.config.age_horizon_multiplier) * int(self.config.cloud_latency)),
        )
        force_age = max(
            1,
            int(self.config.force_age_multiplier) * int(self.config.cloud_latency),
        )
        age_term = float(np.clip(float(cloud_age) / age_denom, 0.0, 1.0))
        deviation_term = float(
            np.clip(
                self._deviation(context) / max(float(self.config.deviation_cap), float(self.config.eps)),
                0.0,
                1.0,
            )
        )
        age_weight, deviation_weight, momentum_weight = self._weights
        risk = (
            age_weight * age_term
            + deviation_weight * deviation_term
            + momentum_weight * momentum_term
        )
        self.last_terms = {
            "age_term": age_term,
            "deviation_term": deviation_term,
            "momentum_term": momentum_term,
        }
        self.last_risk_score = float(risk)

        if latest_cloud_action is None:
            if backbone_step:
                return self._record_reason(step, "bootstrap")
            return self._record_reason(step, "none")

        if backbone_step:
            return self._record_reason(step, "backbone")

        if cloud_age >= force_age and queue_depth == 0:
            return self._record_reason(step, "force_age")

        gap_ok = self._last_call_step is None or (
            step - self._last_call_step >= max(1, int(self.config.min_extra_gap))
        )
        if (
            risk >= float(self.config.burst_threshold)
            and gap_ok
            and queue_depth < max(0, int(self.config.burst_queue_cap))
        ):
            return self._record_reason(step, "burst")

        return self._record_reason(step, "none")

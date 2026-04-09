"""
Shift-Response Policy (SRP) cloud scheduler.

Compatibility-critical momentum-named exports remain available from this
module, but the implemented algorithm is the SRP scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..base import Action, CloudScheduler


@dataclass(frozen=True, init=False)
class ShiftResponsePolicyConfig:
    """Configuration for the Shift-Response Policy (SRP) scheduler."""

    cloud_latency: int
    tau: float
    nmax: int
    eps: float
    seed: int | None
    causal_feature_scales: bool
    causal_baseline: bool

    def __init__(
        self,
        cloud_latency: int,
        tau: float = 1.0,
        nmax: int = 3,
        eps: float = 1e-8,
        seed: int | None = None,
        causal_feature_scales: bool = False,
        causal_baseline: bool = False,
        staleness_multiplier: int | None = None,
    ) -> None:
        resolved_nmax = nmax if staleness_multiplier is None else staleness_multiplier
        object.__setattr__(self, "cloud_latency", int(cloud_latency))
        object.__setattr__(self, "tau", float(tau))
        object.__setattr__(self, "nmax", max(1, int(resolved_nmax)))
        object.__setattr__(self, "eps", max(float(eps), np.finfo(float).eps))
        object.__setattr__(self, "seed", seed if seed is None else int(seed))
        object.__setattr__(self, "causal_feature_scales", bool(causal_feature_scales))
        object.__setattr__(self, "causal_baseline", bool(causal_baseline))

    @property
    def staleness_multiplier(self) -> int:
        """Compatibility alias for older benchmark payloads."""
        return int(self.nmax)


class SelfNormalizingMomentumConfig(ShiftResponsePolicyConfig):
    """Legacy compatibility alias for :class:`ShiftResponsePolicyConfig`."""


class ShiftResponsePolicyScheduler(CloudScheduler):  # pylint: disable=too-many-instance-attributes
    """SRP scheduler driven by self-normalized edge shift and control response."""

    _CALL_REASONS: tuple[str, ...] = ("bootstrap", "pressure", "force_age", "none")

    def __init__(self, config: ShiftResponsePolicyConfig) -> None:
        self.config = config
        self.beta = 1.0 - 1.0 / max(2, int(config.cloud_latency))
        self._seed = config.seed
        self.reset()

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
    def _cloud_received(context: dict[str, Any] | None) -> bool:
        if context is None:
            return False
        return bool(context.get("cloud_received", False))

    @staticmethod
    def _cloud_in_flight(context: dict[str, Any] | None) -> bool:
        if context is None:
            return False
        return bool(context.get("cloud_in_flight", False))

    @staticmethod
    def _rms(vector: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(vector))))

    def reset(self) -> None:
        """Reset SRP state, diagnostics, and the seeded RNG."""
        self._rng = np.random.default_rng(self._seed)
        self._sigma_e: np.ndarray | None = None
        self._sigma_u: np.ndarray | None = None
        self._baseline = 0.0
        self._last_cloud_receipt_step: int | None = None
        self._has_received_cloud = False
        self._bootstrap_issued = False
        self.last_probability = 1.0
        self.last_ratio = 0.0
        self.last_score = 0.0
        self.last_instability = 0.0
        self.last_effort = 0.0
        self.last_stale_override = False
        self.last_call_reason = "none"
        self.call_reason_counts = {reason: 0 for reason in self._CALL_REASONS}

    def debug_state(self) -> dict[str, Any]:
        """Return scheduler diagnostics for tests and benchmark reports."""
        total_actual_calls = sum(
            count
            for reason, count in self.call_reason_counts.items()
            if reason != "none"
        )
        return {
            "last_probability": float(self.last_probability),
            "last_ratio": float(self.last_ratio),
            "last_score": float(self.last_score),
            "last_instability": float(self.last_instability),
            "last_effort": float(self.last_effort),
            "last_stale_override": bool(self.last_stale_override),
            "last_call_reason": self.last_call_reason,
            "call_reason_counts": dict(self.call_reason_counts),
            "last_cloud_receipt_step": self._last_cloud_receipt_step,
            "has_received_cloud": bool(self._has_received_cloud),
            "bootstrap_issued": bool(self._bootstrap_issued),
            "pressure_baseline": float(self._baseline),
            "total_actual_calls": total_actual_calls,
            "causal_feature_scales": bool(self.config.causal_feature_scales),
            "causal_baseline": bool(self.config.causal_baseline),
        }

    def _record_reason(self, reason: str) -> bool:
        self.last_call_reason = reason
        self.call_reason_counts[reason] = self.call_reason_counts.get(reason, 0) + 1
        return reason != "none"

    def _set_pre_receipt_diagnostics(self) -> None:
        self.last_probability = 1.0
        self.last_ratio = 0.0
        self.last_score = 0.0
        self.last_instability = 0.0
        self.last_effort = 0.0
        self.last_stale_override = False

    def _set_missing_input_diagnostics(self) -> None:
        self.last_probability = 0.0
        self.last_ratio = 0.0
        self.last_score = 0.0
        self.last_instability = 0.0
        self.last_effort = 0.0
        self.last_stale_override = False

    def should_call_cloud(
        self,
        step: int,
        obs: dict[str, Any],
        latest_cloud_action: Action | None,
        context: dict[str, Any] | None = None,
    ) -> bool:
        del obs, latest_cloud_action

        if self._cloud_received(context):
            self._has_received_cloud = True
            self._last_cloud_receipt_step = int(step)

        cloud_in_flight = self._cloud_in_flight(context)

        if not self._has_received_cloud:
            self._set_pre_receipt_diagnostics()
            if not self._bootstrap_issued and not cloud_in_flight:
                self._bootstrap_issued = True
                return self._record_reason("bootstrap")
            return self._record_reason("none")

        edge_features = self._vector_from_context(context, "edge_features", expected_size=3)
        prev_edge_features = self._vector_from_context(
            context,
            "prev_edge_features",
            expected_size=3,
        )
        current_command = self._vector_from_context(context, "current_command", expected_size=2)
        prev_command = self._vector_from_context(context, "prev_command", expected_size=2)
        if (
            edge_features is None
            or prev_edge_features is None
            or current_command is None
            or prev_command is None
        ):
            self._set_missing_input_diagnostics()
            return self._record_reason("none")

        if self._sigma_e is None:
            self._sigma_e = np.zeros_like(edge_features, dtype=float)
        if self._sigma_u is None:
            self._sigma_u = np.zeros_like(current_command, dtype=float)

        delta_edge = edge_features - prev_edge_features
        delta_command = current_command - prev_command

        sigma_e_prev = self._sigma_e.copy()
        sigma_u_prev = self._sigma_u.copy()
        baseline_prev = float(self._baseline)

        if self.config.causal_feature_scales:
            sigma_e_for_score = sigma_e_prev
            sigma_u_for_score = sigma_u_prev
        else:
            self._sigma_e = self.beta * self._sigma_e + (1.0 - self.beta) * np.abs(delta_edge)
            self._sigma_u = self.beta * self._sigma_u + (1.0 - self.beta) * np.abs(delta_command)
            sigma_e_for_score = self._sigma_e
            sigma_u_for_score = self._sigma_u

        instability = self._rms(delta_edge / (sigma_e_for_score + self.config.eps))
        effort = self._rms(delta_command / (sigma_u_for_score + self.config.eps))
        score = instability * effort

        if self.config.causal_baseline:
            ratio_denominator = baseline_prev
        else:
            self._baseline = self.beta * self._baseline + (1.0 - self.beta) * score
            ratio_denominator = self._baseline
        ratio = score / (ratio_denominator + self.config.eps)
        probability = float(
            np.clip(
                (ratio - 1.0) / max(float(self.config.tau), float(self.config.eps)),
                0.0,
                1.0,
            )
        )

        stale_override = False
        if self._last_cloud_receipt_step is not None:
            stale_override = (
                step - self._last_cloud_receipt_step
                > int(self.config.nmax) * int(self.config.cloud_latency)
            )

        self.last_probability = probability
        self.last_ratio = float(ratio)
        self.last_score = float(score)
        self.last_instability = float(instability)
        self.last_effort = float(effort)
        self.last_stale_override = bool(stale_override)

        if self.config.causal_feature_scales:
            self._sigma_e = self.beta * sigma_e_prev + (1.0 - self.beta) * np.abs(delta_edge)
            self._sigma_u = self.beta * sigma_u_prev + (1.0 - self.beta) * np.abs(delta_command)
        if self.config.causal_baseline:
            self._baseline = self.beta * baseline_prev + (1.0 - self.beta) * score

        if cloud_in_flight:
            return self._record_reason("none")
        if stale_override:
            return self._record_reason("force_age")
        if self._rng.random() < probability:
            return self._record_reason("pressure")
        return self._record_reason("none")


class SelfNormalizingMomentumScheduler(ShiftResponsePolicyScheduler):
    """Legacy compatibility alias for :class:`ShiftResponsePolicyScheduler`."""

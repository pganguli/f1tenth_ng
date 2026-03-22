"""
Safety-driven scheduling policy implementations.

Paper strategy mapping for the single-tier edge/cloud planner:

- Strategy 1 (Always-hit) -> ``AlwaysCloudScheduler``
- Strategy 2 (Bernoulli) -> ``FixedBernoulliPolicy``
- Strategy 3 (Bernoulli + max-miss guard) -> ``BernoulliMaxMissPolicy``
- Strategy 4i (Logistic) -> ``LogisticRiskPolicy``
- Strategy 4ii (Exponential) -> ``ExponentialRiskPolicy``
- Strategy 4iii (Piecewise ramp) -> ``PiecewiseLinearRampPolicy``
- Strategy 5 (Threshold) -> ``DeterministicDeviationPolicy``
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from ..base import CloudTier
from .safety_signals import SchedulingSignals


@dataclass(frozen=True)
class BaseSafetyConfig:
    """Shared controls for all safety-driven scheduling strategies."""

    min_interval: int = 1
    warmup_steps: int = 1
    fallback_interval: int = 0
    max_miss: int | None = None
    seed: int | None = None


@dataclass(frozen=True)
class DeterministicDeviationConfig(BaseSafetyConfig):
    """Configuration for deterministic deviation-threshold scheduling."""

    threshold: float = 0.03


@dataclass(frozen=True)
class FixedBernoulliConfig(BaseSafetyConfig):
    """Configuration for fixed-probability Bernoulli scheduling."""

    p: float = 0.6


@dataclass(frozen=True)
class BernoulliMaxMissConfig(BaseSafetyConfig):
    """Configuration for Bernoulli scheduling with a max-miss guard."""

    p: float = 0.6
    max_miss: int = 5


@dataclass(frozen=True)
class LogisticRiskConfig(BaseSafetyConfig):
    """Configuration for logistic risk-to-probability scheduling."""

    center: float = 0.03
    slope: float = 40.0
    p_min: float = 0.0
    p_max: float = 1.0


@dataclass(frozen=True)
class ExponentialRiskConfig(BaseSafetyConfig):
    """Configuration for exponential risk-to-probability scheduling."""

    center: float = 0.03
    rate: float = 25.0
    p_min: float = 0.0
    p_max: float = 1.0


@dataclass(frozen=True)
class PiecewiseLinearRampConfig(BaseSafetyConfig):
    """Configuration for piecewise-linear deviation-to-probability scheduling."""

    d_low: float = 0.01
    d_high: float = 0.05


@dataclass(frozen=True)
class TieredProbabilisticRiskConfig(BaseSafetyConfig):
    """Configuration for probabilistic medium/large cloud routing."""

    p_medium: float = 0.3
    p_large: float = 0.2


@dataclass(frozen=True)
class TieredThresholdConfig(BaseSafetyConfig):
    """Configuration for thresholded edge/medium/large routing."""

    medium_threshold: float = 0.03
    large_threshold: float = 0.08
    use_relative_thresholds: bool = False
    history_window: int = 256
    min_history: int = 64
    medium_quantile: float = 0.6
    large_quantile: float = 0.9


@dataclass
class PolicyState:
    """Mutable state shared across scheduling policies."""

    last_call_step: int | None = None
    consecutive_eligible_misses: int = 0


class SchedulingPolicy(ABC):  # pylint: disable=too-few-public-methods
    """Abstract scheduling policy interface."""

    @abstractmethod
    def should_call(
        self,
        step: int,
        signals: SchedulingSignals,
        latest_cloud_action_available: bool,
    ) -> bool:
        """Return whether to issue a cloud request on this step."""


class TieredSchedulingPolicy(ABC):  # pylint: disable=too-few-public-methods
    """Abstract policy for routing between no-call, medium, and large cloud."""

    @abstractmethod
    def choose_tier(
        self,
        step: int,
        signals: SchedulingSignals,
        latest_cloud_action_available: bool,
    ) -> CloudTier | None:
        """Return the requested cloud tier or ``None`` to stay on edge."""


class _BaseSafetyPolicy(SchedulingPolicy, ABC):  # pylint: disable=too-few-public-methods
    """Shared warmup / interval / fallback / max-miss mechanics."""

    def __init__(self, config: BaseSafetyConfig) -> None:
        self.config = config
        self.state = PolicyState()
        self._rng = np.random.default_rng(config.seed)

    def _can_call(self, step: int) -> bool:
        if self.state.last_call_step is None:
            return True
        return (step - self.state.last_call_step) >= max(1, int(self.config.min_interval))

    def _must_call(self, step: int, latest_cloud_action_available: bool) -> bool:
        if not latest_cloud_action_available or step < max(0, int(self.config.warmup_steps)):
            return True
        if (
            self.config.max_miss is not None
            and self.state.consecutive_eligible_misses >= int(self.config.max_miss)
        ):
            return True
        return False

    def _postprocess(self, step: int, called: bool) -> bool:
        if not called and self.config.fallback_interval > 0 and self.state.last_call_step is not None:
            called = (step - self.state.last_call_step) >= int(self.config.fallback_interval)

        if called:
            self.state.last_call_step = step
            self.state.consecutive_eligible_misses = 0
        else:
            self.state.consecutive_eligible_misses += 1
        return called

    @abstractmethod
    def _decide_strategy(self, signals: SchedulingSignals) -> bool:
        """Strategy-specific decision logic."""

    def should_call(
        self,
        step: int,
        signals: SchedulingSignals,
        latest_cloud_action_available: bool,
    ) -> bool:
        if not self._can_call(step):
            return False

        if self._must_call(step, latest_cloud_action_available):
            return self._postprocess(step, True)

        return self._postprocess(step, self._decide_strategy(signals))


class _BaseTieredSafetyPolicy(TieredSchedulingPolicy, ABC):  # pylint: disable=too-few-public-methods
    """Shared warmup / interval / fallback / max-miss mechanics for tiered routing."""

    def __init__(self, config: BaseSafetyConfig) -> None:
        self.config = config
        self.state = PolicyState()
        self._rng = np.random.default_rng(config.seed)

    def _can_call(self, step: int) -> bool:
        if self.state.last_call_step is None:
            return True
        return (step - self.state.last_call_step) >= max(1, int(self.config.min_interval))

    def _must_call(self, step: int, latest_cloud_action_available: bool) -> bool:
        if not latest_cloud_action_available or step < max(0, int(self.config.warmup_steps)):
            return True
        if (
            self.config.max_miss is not None
            and self.state.consecutive_eligible_misses >= int(self.config.max_miss)
        ):
            return True
        return False

    def _postprocess(self, step: int, tier: CloudTier | None) -> CloudTier | None:
        if (
            tier is None
            and self.config.fallback_interval > 0
            and self.state.last_call_step is not None
            and (step - self.state.last_call_step) >= int(self.config.fallback_interval)
        ):
            tier = CloudTier.LARGE

        if tier is not None:
            self.state.last_call_step = step
            self.state.consecutive_eligible_misses = 0
        else:
            self.state.consecutive_eligible_misses += 1
        return tier

    @abstractmethod
    def _choose_tier_strategy(self, signals: SchedulingSignals) -> CloudTier | None:
        """Strategy-specific tier routing logic."""

    def _observe_signals(self, signals: SchedulingSignals) -> None:
        """Hook for policies that maintain running statistics."""

    def choose_tier(
        self,
        step: int,
        signals: SchedulingSignals,
        latest_cloud_action_available: bool,
    ) -> CloudTier | None:
        if not self._can_call(step):
            self._observe_signals(signals)
            return None

        if self._must_call(step, latest_cloud_action_available):
            tier = self._postprocess(step, CloudTier.LARGE)
            self._observe_signals(signals)
            return tier

        tier = self._postprocess(step, self._choose_tier_strategy(signals))
        self._observe_signals(signals)
        return tier


class DeterministicDeviationPolicy(_BaseSafetyPolicy):  # pylint: disable=too-few-public-methods
    """Call cloud when deviation exceeds a deterministic threshold."""

    def __init__(self, config: DeterministicDeviationConfig | None = None) -> None:
        self.deterministic_config = config or DeterministicDeviationConfig()
        super().__init__(self.deterministic_config)

    def _decide_strategy(self, signals: SchedulingSignals) -> bool:
        return signals.deviation >= float(self.deterministic_config.threshold)

class FixedBernoulliPolicy(_BaseSafetyPolicy):  # pylint: disable=too-few-public-methods
    """Sample cloud calls from a fixed Bernoulli probability."""

    def __init__(self, config: FixedBernoulliConfig | None = None) -> None:
        self.fixed_bernoulli_config = config or FixedBernoulliConfig()
        super().__init__(self.fixed_bernoulli_config)

    def probability(self, signals: SchedulingSignals) -> float:
        del signals
        return float(np.clip(self.fixed_bernoulli_config.p, 0.0, 1.0))

    def _decide_strategy(self, signals: SchedulingSignals) -> bool:
        return bool(self._rng.random() < self.probability(signals))


class BernoulliMaxMissPolicy(_BaseSafetyPolicy):  # pylint: disable=too-few-public-methods
    """Fixed-probability Bernoulli policy with a bounded miss streak."""

    def __init__(self, config: BernoulliMaxMissConfig | None = None) -> None:
        self.bernoulli_max_miss_config = config or BernoulliMaxMissConfig()
        super().__init__(self.bernoulli_max_miss_config)

    def probability(self, signals: SchedulingSignals) -> float:
        del signals
        return float(np.clip(self.bernoulli_max_miss_config.p, 0.0, 1.0))

    def _decide_strategy(self, signals: SchedulingSignals) -> bool:
        return bool(self._rng.random() < self.probability(signals))


class LogisticRiskPolicy(_BaseSafetyPolicy):  # pylint: disable=too-few-public-methods
    """
    Logistic policy: map deviation to call probability using a sigmoid.
    """

    def __init__(self, config: LogisticRiskConfig | None = None) -> None:
        self.logistic_config = config or LogisticRiskConfig()
        super().__init__(self.logistic_config)

    def probability(self, signals: SchedulingSignals) -> float:
        z = self.logistic_config.slope * (float(signals.deviation) - self.logistic_config.center)
        sigmoid = 1.0 / (1.0 + math.exp(-z))
        span = max(0.0, self.logistic_config.p_max - self.logistic_config.p_min)
        p = self.logistic_config.p_min + span * sigmoid
        return float(np.clip(p, 0.0, 1.0))

    def _decide_strategy(self, signals: SchedulingSignals) -> bool:
        p = self.probability(signals)
        return bool(self._rng.random() < p)


class ExponentialRiskPolicy(_BaseSafetyPolicy):  # pylint: disable=too-few-public-methods
    """
    Exponential policy: rapidly increasing call probability for high deviation.
    """

    def __init__(self, config: ExponentialRiskConfig | None = None) -> None:
        self.exponential_config = config or ExponentialRiskConfig()
        super().__init__(self.exponential_config)

    def probability(self, signals: SchedulingSignals) -> float:
        excess = max(0.0, float(signals.deviation) - self.exponential_config.center)
        activation = 1.0 - math.exp(-self.exponential_config.rate * excess)
        span = max(0.0, self.exponential_config.p_max - self.exponential_config.p_min)
        p = self.exponential_config.p_min + span * activation
        return float(np.clip(p, 0.0, 1.0))

    def _decide_strategy(self, signals: SchedulingSignals) -> bool:
        p = self.probability(signals)
        return bool(self._rng.random() < p)


class PiecewiseLinearRampPolicy(_BaseSafetyPolicy):  # pylint: disable=too-few-public-methods
    """Increase call probability linearly between two deviation thresholds."""

    def __init__(self, config: PiecewiseLinearRampConfig | None = None) -> None:
        self.piecewise_linear_ramp_config = config or PiecewiseLinearRampConfig()
        if self.piecewise_linear_ramp_config.d_high <= self.piecewise_linear_ramp_config.d_low:
            raise ValueError("d_high must be greater than d_low")
        super().__init__(self.piecewise_linear_ramp_config)

    def probability(self, signals: SchedulingSignals) -> float:
        deviation = float(signals.deviation)
        d_low = float(self.piecewise_linear_ramp_config.d_low)
        d_high = float(self.piecewise_linear_ramp_config.d_high)
        if deviation <= d_low:
            return 0.0
        if deviation >= d_high:
            return 1.0
        return float((deviation - d_low) / (d_high - d_low))

    def _decide_strategy(self, signals: SchedulingSignals) -> bool:
        return bool(self._rng.random() < self.probability(signals))


class TieredProbabilisticRiskPolicy(_BaseTieredSafetyPolicy):  # pylint: disable=too-few-public-methods
    """Route requests using fixed medium/large call probabilities."""

    def __init__(self, config: TieredProbabilisticRiskConfig | None = None) -> None:
        self.tiered_probabilistic_config = config or TieredProbabilisticRiskConfig()
        total_prob = (
            float(self.tiered_probabilistic_config.p_medium)
            + float(self.tiered_probabilistic_config.p_large)
        )
        if total_prob > 1.0 + 1e-9:
            raise ValueError("p_medium + p_large must be <= 1.0")
        super().__init__(self.tiered_probabilistic_config)

    def probabilities(self, signals: SchedulingSignals) -> tuple[float, float]:
        del signals
        cfg = self.tiered_probabilistic_config
        return float(np.clip(cfg.p_medium, 0.0, 1.0)), float(np.clip(cfg.p_large, 0.0, 1.0))

    def _choose_tier_strategy(self, signals: SchedulingSignals) -> CloudTier | None:
        p_medium, p_large = self.probabilities(signals)
        draw = float(self._rng.random())
        if draw < p_large:
            return CloudTier.LARGE
        if draw < p_large + p_medium:
            return CloudTier.MEDIUM
        return None


class TieredThresholdPolicy(_BaseTieredSafetyPolicy):  # pylint: disable=too-few-public-methods
    """Route between edge, medium, and large using a scalar disagreement threshold."""

    def __init__(self, config: TieredThresholdConfig | None = None) -> None:
        self.tiered_threshold_config = config or TieredThresholdConfig()
        self._recent_deviations: deque[float] = deque(
            maxlen=max(1, int(self.tiered_threshold_config.history_window))
        )
        super().__init__(self.tiered_threshold_config)

    def _observe_signals(self, signals: SchedulingSignals) -> None:
        self._recent_deviations.append(abs(float(signals.deviation)))

    def _relative_thresholds(self) -> tuple[float, float] | None:
        cfg = self.tiered_threshold_config
        if not cfg.use_relative_thresholds:
            return None
        if len(self._recent_deviations) < max(1, int(cfg.min_history)):
            return None
        history = np.array(self._recent_deviations, dtype=float)
        medium = float(
            max(
                cfg.medium_threshold,
                np.quantile(history, np.clip(cfg.medium_quantile, 0.0, 1.0)),
            )
        )
        large = float(
            max(
                cfg.large_threshold,
                np.quantile(history, np.clip(cfg.large_quantile, 0.0, 1.0)),
            )
        )
        return medium, max(large, medium)

    def _choose_tier_strategy(self, signals: SchedulingSignals) -> CloudTier | None:
        deviation = float(signals.deviation)
        thresholds = self._relative_thresholds()
        if thresholds is None:
            medium_threshold = float(self.tiered_threshold_config.medium_threshold)
            large_threshold = float(self.tiered_threshold_config.large_threshold)
        else:
            medium_threshold, large_threshold = thresholds
        if deviation < medium_threshold:
            return None
        if deviation < large_threshold:
            return CloudTier.MEDIUM
        return CloudTier.LARGE

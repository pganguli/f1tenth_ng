"""
Archived scheduling policies — superseded by paper-accurate implementations.

These policies were early prototypes that do NOT match the DAC 2026 paper
strategies. They are preserved here for reference but are no longer exported
from the package or available via the CLI.

- ``BernoulliDeviationPolicy``: deviation-aware Bernoulli (p = base_prob +
  risk_gain * excess). Paper Strategy 2 is a *fixed-p* Bernoulli that ignores
  deviation entirely. Replaced by ``FixedBernoulliPolicy``.

- ``MaxMissDeterministicPolicy``: deterministic threshold with a max-miss
  bound. Paper Strategy 3 is a *probabilistic* Bernoulli with a max-miss
  guard. Replaced by ``BernoulliMaxMissPolicy``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .safety_policies import BaseSafetyConfig, SchedulingSignals, _BaseSafetyPolicy


@dataclass(frozen=True)
class BernoulliDeviationConfig(BaseSafetyConfig):
    """Configuration for deviation-aware Bernoulli scheduling (ARCHIVED)."""

    threshold: float = 0.03
    base_prob: float = 0.05
    risk_gain: float = 8.0


@dataclass(frozen=True)
class MaxMissDeterministicConfig(BaseSafetyConfig):
    """Deterministic threshold strategy with a mandatory max-miss bound (ARCHIVED)."""

    threshold: float = 0.03
    max_miss: int = 5


class MaxMissDeterministicPolicy(_BaseSafetyPolicy):  # pylint: disable=too-few-public-methods
    """
    Deterministic threshold strategy with explicit max-miss guarantees (ARCHIVED).

    This is NOT paper Strategy 3. Paper Strategy 3 is a probabilistic Bernoulli
    with a max-miss guard. See ``BernoulliMaxMissPolicy`` for the paper version.
    """

    def __init__(self, config: MaxMissDeterministicConfig | None = None) -> None:
        self.max_miss_config = config or MaxMissDeterministicConfig()
        super().__init__(self.max_miss_config)

    def _decide_strategy(self, signals: SchedulingSignals) -> bool:
        return signals.deviation >= float(self.max_miss_config.threshold)


class BernoulliDeviationPolicy(_BaseSafetyPolicy):  # pylint: disable=too-few-public-methods
    """
    Bernoulli policy: sample cloud invocation probability from deviation (ARCHIVED).

    This is NOT paper Strategy 2. Paper Strategy 2 is a fixed-probability
    Bernoulli that ignores deviation. See ``FixedBernoulliPolicy`` for the
    paper version.
    """

    def __init__(self, config: BernoulliDeviationConfig | None = None) -> None:
        self.bernoulli_config = config or BernoulliDeviationConfig()
        super().__init__(self.bernoulli_config)

    def probability(self, signals: SchedulingSignals) -> float:
        deviation_excess = max(0.0, float(signals.deviation) - self.bernoulli_config.threshold)
        p = self.bernoulli_config.base_prob + self.bernoulli_config.risk_gain * deviation_excess
        return float(np.clip(p, 0.0, 1.0))

    def _decide_strategy(self, signals: SchedulingSignals) -> bool:
        p = self.probability(signals)
        return bool(self._rng.random() < p)

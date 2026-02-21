"""
Scheduler classes that decide **when** to issue cloud inference requests.
"""

from .always_call_scheduler import AlwaysCallScheduler
from .fixed_interval_scheduler import FixedIntervalScheduler

__all__ = [
    "AlwaysCallScheduler",
    "FixedIntervalScheduler",
]

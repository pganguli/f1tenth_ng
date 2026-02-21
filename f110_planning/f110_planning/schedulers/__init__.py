"""
Scheduler classes that decide **when** to issue cloud inference requests.
"""

from .fixed_interval_scheduler import FixedIntervalScheduler

__all__ = [
    "FixedIntervalScheduler",
]

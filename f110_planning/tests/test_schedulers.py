"""Tests for cloud scheduler implementations."""

from f110_planning.schedulers import FixedIntervalScheduler, RLScheduler


def test_fixed_interval():
    sched = FixedIntervalScheduler(interval=3)
    # should call on step 0,3,6,...
    calls = [sched.should_call_cloud(i, {}, None) for i in range(10)]
    expected = [i % 3 == 0 for i in range(10)]
    assert calls == expected


def test_rl_scheduler_basic():
    sched = RLScheduler()
    # no action set -> defaults to False
    assert not sched.should_call_cloud(0, {}, None)
    # set action to True
    sched.set_action(True)
    assert sched.should_call_cloud(5, {}, None)
    # clearing via reset
    sched.reset()
    assert not sched.should_call_cloud(0, {}, None)

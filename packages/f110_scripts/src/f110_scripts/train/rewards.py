"""Reward function registry for RL cloud-scheduler training.

Add a new entry to :data:`REGISTRY` to make it selectable via the
``--reward`` CLI flag in ``train_rl.py``.

Each factory in the registry is called as::

    factory(waypoints=waypoints, **extra_kwargs)

and must return a callable with signature::

    reward_fn(obs: dict, call_mask: list[bool]) -> float

where
* ``obs`` is the simulator observation dict (base + augmented keys),
* ``call_mask`` is the list of per-DNN call decisions for this step.

Examples
--------
Adding a custom reward that penalises calling any DNN::

    from f110_scripts.train.rewards import REGISTRY

    def _my_factory(waypoints, cost=0.05, **_):
        from f110_planning.metrics import crosstrack_error
        import numpy as np
        def _fn(obs, call_mask):
            pos = np.array([obs["poses_x"][0], obs["poses_y"][0]])
            dist = crosstrack_error(pos, waypoints)
            n_calls = sum(call_mask)
            return -float(dist**2) - cost * n_calls
        return _fn

    REGISTRY["cte_plus_cost"] = _my_factory
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


# ---------------------------------------------------------------------------
# Built-in reward factories
# ---------------------------------------------------------------------------

def _cte_only_factory(
    waypoints: np.ndarray,
    **_kwargs: Any,
) -> Callable[[dict, list[bool]], float]:
    """–CTE²: minimise cross-track RMSE; no cloud-cost term."""
    from f110_planning.metrics import crosstrack_error  # pylint: disable=import-outside-toplevel

    wpts = waypoints.copy()

    def _reward(obs: dict[str, Any], _call_mask: list[bool]) -> float:
        pos = np.array([obs["poses_x"][0], obs["poses_y"][0]], dtype=np.float64)
        dist = crosstrack_error(pos, wpts)
        return -float(dist ** 2)

    return _reward


def _cte_plus_call_cost_factory(
    waypoints: np.ndarray,
    cost_per_call: float = 0.05,
    **_kwargs: Any,
) -> Callable[[dict, list[bool]], float]:
    """–CTE² – λ·(calls/m): flat penalty proportional to fraction of DNNs called."""
    from f110_planning.metrics import crosstrack_error  # pylint: disable=import-outside-toplevel

    wpts = waypoints.copy()
    m = 3  # number of DNN slots

    def _reward(obs: dict[str, Any], call_mask: list[bool]) -> float:
        pos = np.array([obs["poses_x"][0], obs["poses_y"][0]], dtype=np.float64)
        dist = crosstrack_error(pos, wpts)
        cte_term = -float(dist ** 2)
        cost_term = -cost_per_call * (sum(call_mask) / m)
        return cte_term + cost_term

    return _reward


def _cte_plus_no_call_penalty_factory(
    waypoints: np.ndarray,
    penalty_per_no_call: float = 0.05,
    **_kwargs: Any,
) -> Callable[[dict, list[bool]], float]:
    """–CTE² – λ·(non-calls/m): penalise skipping DNN calls.

    The complement of :func:`_cte_plus_call_cost_factory`.  Where that
    function discourages calling DNNs, this one discourages *not* calling
    them.  An agent that never triggers a particular DNN slot will receive an
    extra penalty proportional to the fraction of slots left uncalled each
    step, nudging it to distribute cloud calls across all *m* DNNs.

    Parameters
    ----------
    waypoints : np.ndarray
        Reference waypoints for CTE computation.
    penalty_per_no_call : float
        Scale factor λ applied to the uncalled fraction.  A value of 0.05
        means up to –0.05 is added when no DNNs are called at all.
    """
    from f110_planning.metrics import crosstrack_error  # pylint: disable=import-outside-toplevel

    wpts = waypoints.copy()
    m = 3  # number of DNN slots

    def _reward(obs: dict[str, Any], call_mask: list[bool]) -> float:
        pos = np.array([obs["poses_x"][0], obs["poses_y"][0]], dtype=np.float64)
        dist = crosstrack_error(pos, wpts)
        cte_term = -float(dist ** 2)
        no_call_fraction = (m - sum(call_mask)) / m
        penalty_term = -penalty_per_no_call * no_call_fraction
        return cte_term + penalty_term

    return _reward


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Maps reward name → factory callable.
#: Each factory receives ``waypoints`` as a keyword argument plus any
#: additional keyword arguments passed by the caller.
REGISTRY: dict[str, Callable[..., Callable[[dict, list[bool]], float]]] = {
    "cte_only": _cte_only_factory,
    "cte_plus_call_cost": _cte_plus_call_cost_factory,
    "cte_plus_no_call_penalty": _cte_plus_no_call_penalty_factory,
}


def make_reward(
    name: str,
    waypoints: np.ndarray,
    **kwargs: Any,
) -> Callable[[dict, list[bool]], float]:
    """Instantiate a reward function by name.

    Parameters
    ----------
    name : str
        Key in :data:`REGISTRY`.
    waypoints : np.ndarray
        Reference waypoints for CTE computation.
    **kwargs
        Extra keyword arguments forwarded to the factory (e.g. ``cost_per_call``).

    Returns
    -------
    Callable[[dict, list[bool]], float]
        Bound reward function ready to pass as ``reward_fn=`` to the env.

    Raises
    ------
    KeyError
        If ``name`` is not in :data:`REGISTRY`.
    """
    if name not in REGISTRY:
        available = sorted(REGISTRY.keys())
        raise KeyError(
            f"Unknown reward '{name}'. Available: {available}"
        )
    return REGISTRY[name](waypoints=waypoints, **kwargs)

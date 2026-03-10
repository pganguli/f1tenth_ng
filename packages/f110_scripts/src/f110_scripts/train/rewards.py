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
Adding a custom reward that blends CTE with a flat call bonus::

    from f110_scripts.train.rewards import REGISTRY

    def _my_factory(waypoints, bonus=0.02, **_):
        from f110_planning.metrics import crosstrack_error
        import numpy as np
        def _fn(obs, call_mask):
            pos = np.array([obs["poses_x"][0], obs["poses_y"][0]])
            dist = crosstrack_error(pos, waypoints)
            return -float(dist**2) + bonus * sum(call_mask)
        return _fn

    REGISTRY["cte_plus_flat_bonus"] = _my_factory
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable

import numpy as np


# ---------------------------------------------------------------------------
# Built-in reward factories
# ---------------------------------------------------------------------------

def _cte_factory(
    waypoints: np.ndarray,
    **_kwargs: Any,
) -> Callable[[dict, list[bool]], float]:
    """Pure CTE baseline: reward = –(distance to nearest waypoint)².

    The call_mask is entirely ignored — this reward contains no signal
    whatsoever about which DNNs to call or how often.  It is the simplest
    possible reward and serves as a reference point against which all other
    rewards are compared.  An agent trained with ``cte`` will minimise
    cross-track error but is given complete freedom over its cloud-calling
    strategy; any structure it discovers there is driven purely by the
    environment dynamics rather than a reward shaping signal.
    """
    from f110_planning.metrics import crosstrack_error  # pylint: disable=import-outside-toplevel

    wpts = waypoints.copy()

    def _reward(obs: dict[str, Any], _call_mask: list[bool]) -> float:
        pos = np.array([obs["poses_x"][0], obs["poses_y"][0]], dtype=np.float64)
        dist = crosstrack_error(pos, wpts)
        return -float(dist ** 2)

    return _reward


def _cte_sensitivity_reg_factory(
    waypoints: np.ndarray,
    window: int = 200,
    penalty_scale: float = 0.5,
    target_call_rates: float | list[float] | None = None,
    **_kwargs: Any,
) -> Callable[[dict, list[bool]], float]:
    """–CTE² – λ·Σ (r̂ᵢ – r*ᵢ)²: permanent sensitivity-prior regularisation.

    This reward encodes the results of an **offline sensitivity analysis** as
    a set of per-slot target call rates ``r*ᵢ``, then permanently penalises
    the agent whenever its *learned* rolling call distribution deviates from
    those targets.

    **Design intent** (smoke-test / sanity check):
    If the RL agent's discovered calling strategy is wildly inconsistent with
    what a sensitivity analysis says should matter (e.g. it almost never calls
    ``track_width`` despite that DNN having the highest offline sensitivity),
    this reward applies quadratic pressure to correct that.  The penalty is
    two-sided: both undercalling *and* overcalling a slot relative to its
    target are penalised, so the agent converges toward — but not beyond —
    the prior distribution.

    **Important distinction from** ``cte_sensitivity_annealed``:
    The regularisation term is **permanent** — it is present for the entire
    training run and is never annealed away.  The final converged policy is
    therefore *not* equivalent to plain ``cte``; it will always reflect a
    compromise between CTE minimisation and fidelity to the sensitivity prior.
    Use this reward when you trust the prior and want it enforced throughout;
    use ``cte_sensitivity_annealed`` when you only want a headstart.

    DNN slot order (matches ``SelectiveCloudSchedulerEnv.DNN_NAMES``):
        0 → left_wall_dist
        1 → track_width
        2 → heading_error

    Parameters
    ----------
    waypoints : np.ndarray
        Reference waypoints for CTE computation.
    window : int
        Length of the rolling history (in steps) used to estimate each
        slot's call rate.  Larger values (e.g. 500) react more slowly but
        give a smoother estimate; smaller values (e.g. 50) react quickly
        but can be jittery.
    penalty_scale : float
        Scale factor λ applied to the total squared deviation across all
        slots.  Tune this relative to typical CTE² magnitudes so neither
        term dominates entirely.
    target_call_rates : float, list[float], or None
        Per-slot target call rate ``r*ᵢ`` from the offline sensitivity
        analysis.  Raw importance weights are accepted and normalised
        internally to sum to 1, so you can pass the same values as
        ``cte_sensitivity_annealed``'s ``call_weights``.

        * ``None`` — uniform target ``1/m`` (≈ 0.33) for every slot.
        * ``float`` — the same target applied to all three slots.
        * ``list[float]`` of length 3 — individual targets for
          [left_wall_dist, track_width, heading_error] respectively.
          Example for the known F1 importance order
          (track_width > left_wall > heading): ``[0.3, 0.5, 0.2]``.
    """
    from f110_planning.metrics import crosstrack_error  # pylint: disable=import-outside-toplevel

    wpts = waypoints.copy()
    m = 3  # number of DNN slots

    if target_call_rates is None:
        r_stars = [1.0 / m] * m
    elif isinstance(target_call_rates, (int, float)):
        r_stars = [float(target_call_rates)] * m
    else:
        if len(target_call_rates) != m:
            raise ValueError(
                f"target_call_rates list must have exactly {m} entries "
                f"(one per DNN slot), got {len(target_call_rates)}."
            )
        total = sum(target_call_rates)
        r_stars = [float(r) / total for r in target_call_rates]

    # Independent rolling history per slot, pre-filled with zeros so the
    # penalty is applied from the very first episode.
    histories: list[deque[int]] = [
        deque([0] * window, maxlen=window) for _ in range(m)
    ]

    def _reward(obs: dict[str, Any], call_mask: list[bool]) -> float:
        pos = np.array([obs["poses_x"][0], obs["poses_y"][0]], dtype=np.float64)
        dist = crosstrack_error(pos, wpts)
        cte_term = -float(dist ** 2)

        # Update each slot's history with this step's call decision.
        for i, called in enumerate(call_mask):
            histories[i].append(1 if called else 0)

        # Penalise squared deviation of each slot's rolling rate from its target.
        deviation = sum(
            (sum(h) / window - r_stars[i]) ** 2
            for i, h in enumerate(histories)
        )
        return cte_term - penalty_scale * deviation

    return _reward


def _cte_sensitivity_annealed_factory(
    waypoints: np.ndarray,
    call_weights: list[float] | None = None,
    bonus_scale: float = 0.1,
    anneal_steps: int = 1_000_000,
    **_kwargs: Any,
) -> Callable[[dict, list[bool]], float]:
    """–CTE² + β(t)·Σ ŵᵢ·𝟙[DNNᵢ called]: annealed sensitivity-prior bootstrap.

    This reward gives training a **headstart** by temporarily rewarding the
    agent for calling DNNs in proportion to their known importance from an
    offline sensitivity analysis.  The per-step bonus decays **linearly to
    zero** over ``anneal_steps`` environment steps, after which the reward is
    identical to plain ``cte`` (pure CTE minimisation).

    **Design intent** (faster convergence toward a good calling strategy):
    Early in training the agent has no idea which DNNs matter most, so it
    may waste many episodes on calling strategies that sensitivity analysis
    already rules out as poor.  The bonus injects that prior knowledge
    directly into the reward signal, steering early exploration toward
    high-value calling patterns.  Once the agent has had enough steps to
    learn the true CTE-driven optimum, the bonus has fully annealed away and
    the policy is driven entirely by CTE — the prior leaves no permanent bias.

    **Important distinction from** ``cte_sensitivity_reg``:
    The bonus is **temporary** — it fully vanishes at ``anneal_steps``.  The
    final converged policy *is* equivalent to plain ``cte``.  Use this reward
    for faster convergence without permanently constraining the policy; use
    ``cte_sensitivity_reg`` when you want the sensitivity prior enforced
    throughout training.

    The bonus at step ``t`` is:
        β(t) = bonus_scale · max(0, 1 – t / anneal_steps)
    and the per-step contribution is:
        bonus = β(t) · Σ_{i: called} ŵᵢ
    where ŵᵢ are the sensitivity weights normalised to sum to 1.

    DNN slot order (matches ``SelectiveCloudSchedulerEnv.DNN_NAMES``):
        0 → left_wall_dist
        1 → track_width
        2 → heading_error

    Parameters
    ----------
    waypoints : np.ndarray
        Reference waypoints for CTE computation.
    call_weights : list[float] of length 3, or None
        Raw importance weights from offline sensitivity analysis for each
        DNN slot [left_wall_dist, track_width, heading_error].  Normalised
        to sum to 1 internally, so only relative magnitudes matter.
        Defaults to uniform ``[1/3, 1/3, 1/3]``, which makes this reward
        equivalent to ``cte`` with a small, decaying uniform bonus.
        Example for the known F1 importance order
        (track_width > left_wall > heading): ``[0.3, 0.5, 0.2]``.
    bonus_scale : float
        Initial magnitude β₀ of the call bonus.  Should be small relative
        to typical CTE² values so the CTE signal always dominates;
        0.05–0.2 is a reasonable range.
    anneal_steps : int
        Number of environment steps over which β(t) decays from
        ``bonus_scale`` to 0.  After this point the reward is identical
        to ``cte``.  Should be a fraction of total training timesteps
        (e.g. 20–40 %) to allow ample time for CTE-driven fine-tuning.
    """
    from f110_planning.metrics import crosstrack_error  # pylint: disable=import-outside-toplevel

    wpts = waypoints.copy()
    m = 3  # number of DNN slots

    raw_weights = [1.0 / m] * m if call_weights is None else list(call_weights)
    if len(raw_weights) != m:
        raise ValueError(
            f"call_weights must have exactly {m} entries, got {len(raw_weights)}."
        )
    total = sum(raw_weights)
    norm_weights = [w / total for w in raw_weights]

    # Mutable step counter shared across calls.
    state = {"steps": 0}

    def _reward(obs: dict[str, Any], call_mask: list[bool]) -> float:
        pos = np.array([obs["poses_x"][0], obs["poses_y"][0]], dtype=np.float64)
        dist = crosstrack_error(pos, wpts)
        cte_term = -float(dist ** 2)

        beta = bonus_scale * max(0.0, 1.0 - state["steps"] / anneal_steps)
        bonus = beta * sum(
            norm_weights[i] for i, called in enumerate(call_mask) if called
        )

        state["steps"] += 1
        return cte_term + bonus

    return _reward


def _cte_sensitivity_staleness_factory(
    waypoints: np.ndarray,
    call_weights: list[float] | None = None,
    bonus_scale: float = 0.1,
    anneal_steps: int = 1_000_000,
    age_scale_steps: int = 20,
    **_kwargs: Any,
) -> Callable[[dict, list[bool]], float]:
    """–CTE² + β(t)·Σ ŵᵢ·(age_i / L)·𝟙[DNNᵢ called]: staleness-weighted bonus.

    Extends ``cte_sensitivity_annealed`` by scaling each per-call bonus by
    the current **cloud age** of that slot.  The intuition is:

    * Calling a slot that was just refreshed gives almost no bonus — it is
      already fresh.
    * Calling a stale slot (large ``cloud_age[i]``) gives a large bonus,
      proportional to both its accuracy sensitivity *and* how overdue a
      refresh is.

    Formally, the per-step bonus contribution for slot *i* is:

    .. math::

        \\beta(t)\\cdot\\hat{w}_i \\cdot \\min\\!\\left(\\frac{\\ell_i(t)}{L},\\, 1\\right)
        \\cdot \\mathbf{1}[\\text{DNN}_i\\text{ called}]

    where :math:`\\ell_i(t)` is ``cloud_age[i]`` from the observation, capped at
    ``age_scale_steps`` *L* so a single step is always enough to earn the full
    bonus once a slot is sufficiently overdue.  The outer envelope β(t) decays
    linearly to zero over ``anneal_steps``, after which the reward is identical
    to plain ``cte``.

    This is structurally Markovian (uses current state, not a rolling rate
    window) and creates no conflicting gradient: a call that is simultaneously
    sensitivity-important *and* overdue is rewarded most; a redundant call on a
    fresh high-sensitivity slot earns almost nothing.

    Parameters
    ----------
    waypoints : np.ndarray
        Reference waypoints for CTE computation.
    call_weights : list[float] of length 3, or None
        Raw accuracy-sensitivity weights [left_wall_dist, track_width,
        heading_error].  Normalised to sum to 1.  Defaults to uniform.
    bonus_scale : float
        Initial bonus magnitude β₀.  Should be small relative to typical CTE².
    anneal_steps : int
        Steps over which β(t) decays to 0.
    age_scale_steps : int
        Age at which the staleness factor saturates to 1.0 (i.e. full bonus).
        Set this near the expected crossover latency L* (~cloud_latency × 2
        is a reasonable starting point).
    """
    from f110_planning.metrics import crosstrack_error  # pylint: disable=import-outside-toplevel

    wpts = waypoints.copy()
    m = 3

    raw_weights = [1.0 / m] * m if call_weights is None else list(call_weights)
    if len(raw_weights) != m:
        raise ValueError(
            f"call_weights must have exactly {m} entries, got {len(raw_weights)}."
        )
    total = sum(raw_weights)
    norm_weights = [w / total for w in raw_weights]

    L = float(age_scale_steps)
    state = {"steps": 0}

    def _reward(obs: dict[str, Any], call_mask: list[bool]) -> float:
        pos = np.array([obs["poses_x"][0], obs["poses_y"][0]], dtype=np.float64)
        dist = crosstrack_error(pos, wpts)
        cte_term = -float(dist ** 2)

        beta = bonus_scale * max(0.0, 1.0 - state["steps"] / anneal_steps)
        if beta > 0.0:
            cloud_age = obs.get("cloud_age", [0.0] * m)
            bonus = beta * sum(
                norm_weights[i] * min(float(cloud_age[i]) / L, 1.0)
                for i, called in enumerate(call_mask)
                if called
            )
        else:
            bonus = 0.0

        state["steps"] += 1
        return cte_term + bonus

    return _reward


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Maps reward name → factory callable.
#: Each factory receives ``waypoints`` as a keyword argument plus any
#: additional keyword arguments passed by the caller.
REGISTRY: dict[str, Callable[..., Callable[[dict, list[bool]], float]]] = {
    "cte": _cte_factory,
    "cte_sensitivity_reg": _cte_sensitivity_reg_factory,
    "cte_sensitivity_annealed": _cte_sensitivity_annealed_factory,
    "cte_sensitivity_staleness": _cte_sensitivity_staleness_factory,
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

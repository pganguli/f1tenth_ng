"""Sensitivity-proportional cloud-DNN scheduler."""


class SensitivityProportionalScheduler:
    """Selects DNNs proportionally to their sensitivity weights.

    Uses a deficit (credit-accumulation) algorithm: every step each DNN
    accumulates credits equal to its normalised weight; the *top_k*
    highest-credit DNNs are selected and have ``1.0`` subtracted from
    their credit.

    Over *T* steps DNN *i* is selected approximately ``T * weight[i]``
    times.  Because credits are bounded in ``(-1, 1)`` the deviation from
    the ideal count is at most ±1 at any point, so the long-run call rate
    converges to exactly ``weight[i]`` as T → ∞.

    ``burst_window`` groups consecutive calls to the same DNN set
    ---------------------------------------------------------------
    Setting ``burst_window=N`` makes each credit-based selection persist for
    *N* consecutive calls before the credit algorithm is consulted again.
    The credit accumulation still happens once per *N* calls (i.e. once per
    "logical step"), so the long-run call rates are preserved exactly —
    only the temporal grouping changes.

    With ``burst_window=1`` (default) you get the original interleaved pattern
    ``l, t, h, l, t, h, …``.  With ``burst_window=3`` and ``top_k=1`` you get
    ``l, l, l, t, t, t, h, h, h, …``.

    Parameters
    ----------
    weights : list[float]
        Per-DNN sensitivity weights.  They will be normalised so they sum
        to 1; ``len(weights)`` determines the mask length produced by
        :meth:`get_call_mask`.
    top_k : int
        Number of DNN slots to mark ``True`` per step.
    burst_window : int
        Number of consecutive :meth:`get_call_mask` calls that share the
        same selection before credits are updated and a new selection is
        made.  Must be ≥ 1.  Default is ``1`` (original behaviour).
    """

    def __init__(
        self,
        weights: list[float],
        top_k: int = 1,
        burst_window: int = 1,
    ) -> None:
        if burst_window < 1:
            raise ValueError("burst_window must be >= 1.")
        total = sum(weights)
        if total <= 0:
            raise ValueError("weights must contain at least one positive value.")
        self._weights = [w / total for w in weights]
        self._top_k = min(top_k, len(weights))
        self._burst_window = burst_window
        # Pre-load credits so the first call already reflects the weights.
        self._credits = list(self._weights)
        # Current mask (computed lazily on first call).
        self._current_mask: list[bool] | None = None
        # How many times the current mask has been returned in this burst.
        self._burst_count = 0

    def get_call_mask(self) -> list[bool]:
        """Return the mask for the current step and update credits."""
        # Compute a new selection only at the start of each burst window.
        if self._current_mask is None or self._burst_count >= self._burst_window:
            self._burst_count = 0
            n = len(self._weights)
            for i in range(n):
                self._credits[i] += self._weights[i]
            # Select top_k by highest credit; lower index wins ties (stable).
            sorted_idx = sorted(range(n), key=lambda i: self._credits[i], reverse=True)
            selected = sorted_idx[: self._top_k]
            for i in selected:
                self._credits[i] -= 1.0
            mask = [False] * n
            for i in selected:
                mask[i] = True
            self._current_mask = mask

        self._burst_count += 1
        return list(self._current_mask)

    def reset(self) -> None:
        """Reset credits to their initial pre-loaded state."""
        self._credits = list(self._weights)
        self._current_mask = None
        self._burst_count = 0

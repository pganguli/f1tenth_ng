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

    Parameters
    ----------
    weights : list[float]
        Per-DNN sensitivity weights.  They will be normalised so they sum
        to 1; ``len(weights)`` determines the mask length produced by
        :meth:`get_call_mask`.
    top_k : int
        Number of DNN slots to mark ``True`` per step.
    """

    def __init__(self, weights: list[float], top_k: int = 1) -> None:
        total = sum(weights)
        if total <= 0:
            raise ValueError("weights must contain at least one positive value.")
        self._weights = [w / total for w in weights]
        self._top_k = min(top_k, len(weights))
        # Pre-load credits so the first call already reflects the weights.
        self._credits = list(self._weights)

    def get_call_mask(self) -> list[bool]:
        """Return the mask for the current step and update credits."""
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
        return mask

    def reset(self) -> None:
        """Reset credits to their initial pre-loaded state."""
        self._credits = list(self._weights)

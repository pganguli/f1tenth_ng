"""Round-robin cloud-DNN scheduler for per-DNN selective calling."""


class RoundRobinScheduler:
    """Cycles through DNN slots, selecting *top_k* per step.

    For ``top_k=1`` and ``num_dnns=3`` the call pattern repeats as:
    step 0 → DNN 0, step 1 → DNN 1, step 2 → DNN 2, step 3 → DNN 0, …

    For ``top_k=2`` and ``num_dnns=3``:
    step 0 → {0, 1}, step 1 → {2, 0}, step 2 → {1, 2}, …

    In all cases each DNN is called at exactly ``top_k / num_dnns`` of steps
    over any complete period of ``num_dnns`` steps.

    Parameters
    ----------
    num_dnns : int
        Total number of DNN slots (length of the mask produced).
    top_k : int
        Number of slots to mark ``True`` per step.
    """

    def __init__(self, num_dnns: int = 3, top_k: int = 1) -> None:
        self._num_dnns = num_dnns
        self._top_k = min(top_k, num_dnns)
        self._counter = 0

    def get_call_mask(self) -> list[bool]:
        """Return the mask for the current step and advance the counter."""
        mask = [False] * self._num_dnns
        for i in range(self._top_k):
            mask[(self._counter + i) % self._num_dnns] = True
        self._counter = (self._counter + self._top_k) % self._num_dnns
        return mask

    def reset(self) -> None:
        """Reset the round-robin counter to its initial state."""
        self._counter = 0

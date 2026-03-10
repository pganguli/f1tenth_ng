"""Custom SB3 callbacks for RL cloud-scheduler training."""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CteRmseCallback(BaseCallback):
    """Log per-episode cross-track error RMSE to TensorBoard.

    Each environment step the ``SelectiveCloudSchedulerEnv`` now emits
    ``info["cte"]`` — the raw Euclidean distance from the ego position to the
    nearest waypoint.  This callback accumulates those values, and when an
    episode ends it computes the RMSE over all steps in that episode and
    records it under ``"rollout/cte_rmse"`` in TensorBoard.

    Because SB3's TensorBoard logger uses ``num_timesteps`` as the x-axis,
    the resulting plot shows CTE RMSE vs. training timesteps, which is
    exactly what you want to track whether the policy converges to tighter
    tracking over time.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        # One accumulator list per parallel environment.
        self._ep_sq_ctes: list[list[float]] = []

    def _on_training_start(self) -> None:
        n = self.training_env.num_envs
        self._ep_sq_ctes = [[] for _ in range(n)]

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [False] * len(infos))

        for i, (info, done) in enumerate(zip(infos, dones)):
            cte = info.get("cte")
            if cte is not None:
                self._ep_sq_ctes[i].append(cte ** 2)

            if done and self._ep_sq_ctes[i]:
                rmse = float(np.sqrt(np.mean(self._ep_sq_ctes[i])))
                self.logger.record("rollout/cte_rmse", rmse)
                if self.verbose >= 1:
                    print(
                        f"[CteRmseCallback] env={i} "
                        f"timesteps={self.num_timesteps} "
                        f"cte_rmse={rmse:.4f}"
                    )
                self._ep_sq_ctes[i] = []

        return True

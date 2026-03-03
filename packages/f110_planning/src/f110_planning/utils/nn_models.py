"""
Shared DNN architectures and serialisation helpers for F1TENTH.

Architecture overview (all produce a single scalar output):
    1 – tiny single-channel CNN (~134 params)
    2 – 3-stage single-channel CNN (~small)
    3 – [Conv(1→1), Conv(1→8)] → Linear(1072, 32) (~small)
    4 – [Conv(1→1), Conv(1→16)] → Linear(2144, 32) (~small-medium)
    5 – [Conv(1→8), Conv(8→16)] → Linear(2144, 64) (~medium)
    6 – [Conv(1→8), Conv(8→32)] → Linear(4288, 64) (~medium-large)
    7 – [Conv(1→16), Conv(16→32)] → Linear(4288, 128) (~large)
"""

import io
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn


def get_architecture(arch_id: int) -> nn.Module:
    """
    Factory function for F1TENTH single-output LiDAR neural network architectures.

    All architectures accept a ``(batch, 1, 1080)`` tensor and return a
    ``(batch, 1)`` scalar — suitable for heading error, left wall distance,
    or track width prediction.

    Args:
        arch_id: An integer in ``{1, 2, 3, 4, 5, 6, 7}`` identifying the
            specific layer configuration (ordered roughly by parameter count).

    Returns:
        An uninitialised ``nn.Sequential`` module.

    Raises:
        ValueError: If ``arch_id`` is not in the supported set.
    """
    factories = {
        1: lambda: nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=8),
            nn.Flatten(),
            nn.Linear(134, 1),
        ),
        2: lambda: nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(1, 1, kernel_size=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(1, 1, kernel_size=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Flatten(),
            nn.Linear(32, 8),
            nn.ELU(),
            nn.Linear(8, 1),
        ),
        3: lambda: nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(1, 8, kernel_size=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Flatten(),
            nn.Linear(1072, 32),
            nn.ELU(),
            nn.Linear(32, 1),
        ),
        4: lambda: nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(1, 16, kernel_size=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Flatten(),
            nn.Linear(2144, 32),
            nn.ELU(),
            nn.Linear(32, 1),
        ),
        5: lambda: nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(8, 16, kernel_size=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Flatten(),
            nn.Linear(2144, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        ),
        6: lambda: nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(8, 32, kernel_size=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Flatten(),
            nn.Linear(4288, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        ),
        7: lambda: nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Flatten(),
            nn.Linear(4288, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        ),
    }

    if arch_id not in factories:
        raise ValueError(
            f"Architecture ID {arch_id} is not supported. "
            f"Valid IDs are: {sorted(factories.keys())}"
        )

    return factories[arch_id]()


# ---------------------------------------------------------------------------
# Self-sufficient model serialisation helpers
# ---------------------------------------------------------------------------


def save_as_torchscript(
    model: nn.Module,
    path: str | Path,
    metadata: dict[str, Any],
) -> None:
    """Save an ``nn.Module`` as a self-sufficient TorchScript ``.pt`` file.

    The saved file embeds both the full computation graph (via
    ``torch.jit.script``) and all metadata (``arch_id``, ``target_col``,
    training config, quantization flag, …) as a JSON sidecar inside the same
    archive.  No separate architecture class or external config file is
    required to load a non-quantized model.

    For quantized models (``metadata["quantized"] is True``) the *FP32*
    computation graph is scripted and the FP32 ``state_dict`` is additionally
    embedded as a binary extra file.  This is necessary because
    ``torchao.AffineQuantizedTensor`` weights are not TorchScript-serialisable.
    At load time the caller rebuilds the module, loads the FP32 weights, and
    re-applies ``quantize_()``—which is semantically equivalent since torchao
    INT8 weight quantisation is deterministic given the FP32 weights.

    Args:
        model: The FP32 ``nn.Module`` to script and save.  Must be in eval
            mode and **not** yet quantized.
        path: Destination path for the ``.pt`` file.
        metadata: Dict of metadata to embed (must be JSON-serialisable).
            Should contain at minimum ``"quantized"`` (bool).  For quantized
            models ``"arch_id"`` is also required so the model can be
            faithfully reconstructed at load time.
    """
    path = Path(path)
    extra_files: dict[str, bytes] = {
        "metadata.json": json.dumps(metadata).encode(),
    }

    # For quantized models embed the FP32 state_dict as well.
    # This lets _load_model reconstruct the quantized module without the
    # ScriptModule state_dict key-name complications.
    if metadata.get("quantized", False):
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        extra_files["state_dict.pt"] = buf.getvalue()

    if metadata.get("quantized", False):
        # torchao AffineQuantizedTensor / LinearActivationQuantizedTensor
        # weights are not TorchScript-serialisable, so we script a freshly
        # instantiated FP32 architecture for the computation graph but store
        # the quantized state_dict so it can be loaded directly after
        # re-applying quantize_() at inference time.
        arch_id: int = metadata["arch_id"]
        fp32_for_graph = get_architecture(arch_id)
        fp32_for_graph.eval()
        scripted = torch.jit.script(fp32_for_graph)
    else:
        scripted = torch.jit.script(model)
    torch.jit.save(scripted, str(path), _extra_files=extra_files)


def load_torchscript(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[torch.ScriptModule, dict[str, Any]]:
    """Load a self-sufficient TorchScript model saved by :func:`save_as_torchscript`.

    For *non-quantized* models the returned :class:`torch.ScriptModule` is
    ready for inference with **no** additional imports required — no
    ``nn_models.py``, no ``arch_id``, no config file.

    For *quantized* models (``metadata["quantized"] is True``), the returned
    module is the FP32 TorchScript graph.  Use the ``"arch_id"`` and the
    ``"_state_dict_bytes"`` private key in *metadata* to reconstruct the
    quantized module::

        scripted, meta = load_torchscript(path)
        if meta["quantized"]:
            model = get_architecture(meta["arch_id"])
            buf = io.BytesIO(meta["_state_dict_bytes"])
            model.load_state_dict(torch.load(buf, map_location=device))
            model.eval()
            quantize_(model, Int8DynamicActivationInt8WeightConfig())

    Args:
        path: Path to the ``.pt`` file produced by :func:`save_as_torchscript`.
        map_location: Device to map tensors to on load.

    Returns:
        A ``(scripted_model, metadata)`` tuple where *metadata* mirrors the
        dict passed to :func:`save_as_torchscript`.  For quantized files an
        extra ``"_state_dict_bytes"`` key (raw ``bytes``) is injected by this
        function for downstream reconstruction.
    """
    extra: dict[str, bytes] = {"metadata.json": b"", "state_dict.pt": b""}
    scripted = torch.jit.load(str(path), _extra_files=extra, map_location=map_location)
    metadata: dict[str, Any] = json.loads(extra["metadata.json"].decode())
    if extra["state_dict.pt"]:
        # Attach raw bytes under a private key; caller decides whether to use them.
        metadata["_state_dict_bytes"] = extra["state_dict.pt"]
    return scripted, metadata

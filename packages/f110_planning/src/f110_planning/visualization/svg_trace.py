"""
Post-simulation SVG trace renderer.

Collects car-path and cloud-call data during a simulation run and renders it as
a clean vector SVG using matplotlib.  Designed as an alternative to pyglet when
a publication-quality image (white background, shaded cloud-call regions,
path drawn through them) is needed.

Typical usage
-------------
::

    from f110_planning.visualization.svg_trace import SimTrace, collect_step, render_svg

    trace = SimTrace()
    while not done:
        action = planner.plan(obs)
        obs, *_ = env.step(action)
        collect_step(trace, obs, planner)

    render_svg(trace, args.map, args.map_ext, waypoints, output_path="trace.svg")
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import yaml
from PIL import Image

if TYPE_CHECKING:
    pass

# ── Per-DNN colours (matplotlib float RGBA) ───────────────────────────────────
# Index 0 = left-wall, 1 = track-width, 2 = heading — matches SelectiveEdgeCloudPlanner
_DNN_COLORS: list[tuple[float, float, float]] = [
    (0.00, 0.82, 1.00),   # cyan      — left-wall distance
    (0.39, 0.90, 0.16),   # lime      — track-width
    (0.90, 0.24, 0.90),   # magenta   — heading error
]
_MULTI_CALL_COLOR: tuple[float, float, float] = (1.0, 0.55, 0.0)  # orange
_DNN_LABELS: list[str] = ["Left-wall DNN", "Track-width DNN", "Heading DNN"]


# ── Map helpers ────────────────────────────────────────────────────────────────

def _pool_wall_mask(img: np.ndarray, target_px: int = 1200) -> np.ndarray:
    """Downsample a grayscale map to a binary wall mask using max-pooling.

    Max-pooling (any wall pixel in a block → wall in output) preserves thin
    1-pixel-wide walls that would otherwise disappear under average-downsampling.
    The result is used with ``contourf`` to produce vector wall paths in the SVG.

    Parameters
    ----------
    img:
        Grayscale image array (H, W), uint8.  Pixels < 128 are treated as walls.
    target_px:
        Target size (longest dimension) for the output mask.  Larger = more
        detail but more SVG path nodes.
    """
    h, w = img.shape
    factor = max(1, max(h, w) // target_px)
    wall = (img < 128).view(np.uint8)
    if factor == 1:
        return wall
    nh = h // factor
    nw = w // factor
    cropped = wall[:nh * factor, :nw * factor]
    return cropped.reshape(nh, factor, nw, factor).max(axis=(1, 3))


# ── Data collection ────────────────────────────────────────────────────────────

@dataclasses.dataclass
class SimTrace:
    """Accumulated simulation data for post-run SVG export.

    Attributes
    ----------
    positions:
        Car (x, y) in metres at each recorded step.
    cloud_events:
        ``(x, y, call_mask)`` tuples.  *call_mask* is a list[bool] of length 3
        for :class:`SelectiveEdgeCloudPlanner`, or ``None`` for the binary
        :class:`EdgeCloudPlanner`.  Only steps where at least one DNN was
        called are recorded.
    """

    positions: list[tuple[float, float]] = dataclasses.field(default_factory=list)
    cloud_events: list[tuple[float, float, list[bool] | None]] = dataclasses.field(
        default_factory=list
    )


def collect_step(trace: SimTrace, obs: dict, planner: object, agent_idx: int = 0) -> None:
    """Record one simulation step into *trace*.

    Parameters
    ----------
    trace:
        The :class:`SimTrace` being accumulated.
    obs:
        Observation dict from :meth:`F110Env.step`.
    planner:
        The active planner (inspected for ``last_call_mask`` /
        ``last_cloud_call``).
    agent_idx:
        Which agent to track (default 0 = ego).
    """
    x = float(obs["poses_x"][agent_idx])
    y = float(obs["poses_y"][agent_idx])
    trace.positions.append((x, y))

    call_mask = getattr(planner, "last_call_mask", None)
    if call_mask is not None:
        mask: list[bool] = list(call_mask)
        if any(mask):
            trace.cloud_events.append((x, y, mask))
    elif getattr(planner, "last_cloud_call", False):
        trace.cloud_events.append((x, y, None))


# ── Rendering ─────────────────────────────────────────────────────────────────

def render_svg(
    trace: SimTrace,
    map_path: str,
    map_ext: str,
    waypoints: np.ndarray | None = None,
    output_path: str = "trace.svg",
    *,
    cloud_marker_size: float | None = None,
    cloud_alpha: float = 0.18,
    path_linewidth: float = 1.5,
    path_color: str = "#f0c040",
) -> None:
    """Render *trace* as a vector SVG file.

    The map walls are drawn as filled vector paths (via ``contourf`` over a
    max-pooled binary mask), so thin wall segments are never lost.  The
    background is white (free space).  Cloud-call positions are rendered as
    translucent scatter markers sized to approximately 1 m radius in world
    coordinates; they accumulate into shaded regions where calls cluster.  The
    car path is drawn **on top** of the shading so it is always legible through
    the translucent overlay.

    Parameters
    ----------
    trace:
        Collected simulation data.
    map_path:
        Map base path without extension (the ``--map`` CLI argument value).
    map_ext:
        Map image extension, e.g. ``".png"``.
    waypoints:
        Optional ``(N, 2)`` reference-line array in metres.
    output_path:
        Destination file path.  The ``.svg`` extension is enforced.
    cloud_marker_size:
        Scatter marker area in points² for cloud-call dots.  ``None`` (default)
        auto-computes the size so each marker has a ~1 m radius in world
        coordinates.
    cloud_alpha:
        Per-marker alpha for cloud-call scatter.  Low values let overlapping
        markers accumulate into visibly denser regions without fully obscuring
        the path beneath.
    path_linewidth:
        Line width of the car-path trace in points.
    path_color:
        Matplotlib colour string for the car-path trace.
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    # Enforce .svg extension.
    if not output_path.endswith(".svg"):
        output_path = output_path + ".svg"

    # ── Load map ───────────────────────────────────────────────────────────────
    with open(map_path + ".yaml", encoding="utf-8") as f:
        meta = yaml.safe_load(f)

    resolution: float = float(meta["resolution"])
    origin_x: float = float(meta["origin"][0])
    origin_y: float = float(meta["origin"][1])

    img_raw = np.array(
        Image.open(map_path + map_ext)
        .convert("L")
        .transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    )
    h, w = img_raw.shape

    map_width_m: float = w * resolution
    map_height_m: float = h * resolution

    # ── Build vector wall mask via max-pool + contourf ─────────────────────────
    # Max-pool to ~1200 px longest side: preserves thin walls, keeps SVG compact.
    wall_pooled = _pool_wall_mask(img_raw)
    ph, pw = wall_pooled.shape
    x_pool = np.linspace(origin_x, origin_x + map_width_m, pw)
    y_pool = np.linspace(origin_y, origin_y + map_height_m, ph)
    XX, YY = np.meshgrid(x_pool, y_pool)

    # ── Figure setup ───────────────────────────────────────────────────────────
    aspect_ratio = map_width_m / map_height_m
    fig_w = 14.0
    fig_h = min(fig_w / aspect_ratio, 10.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_facecolor("white")   # free-space colour
    fig.patch.set_facecolor("white")

    # Walls as filled vector paths: only draws where wall_pooled == 1.
    ax.contourf(XX, YY, wall_pooled, levels=[0.5, 1.5], colors=["#444444"], zorder=0)

    # ── Reference line ─────────────────────────────────────────────────────────
    if waypoints is not None and len(waypoints) > 0:
        ax.plot(
            waypoints[:, 0],
            waypoints[:, 1],
            color="#bbbbbb",
            linewidth=0.7,
            zorder=1,
        )

    # ── Cloud-call shaded regions ──────────────────────────────────────────────
    # Establish axes transforms so we can compute world-space marker size.
    ax.set_aspect("equal")
    fig.tight_layout(pad=0.3)
    fig.canvas.draw()

    if cloud_marker_size is None:
        bbox = ax.get_window_extent()
        xlim = ax.get_xlim()
        pts_per_m = (bbox.width * 72.0 / fig.dpi) / (xlim[1] - xlim[0])
        # Target radius ≈ 1 m in world coordinates.
        cloud_marker_size = np.pi * pts_per_m ** 2

    legend_handles: list[mpatches.Patch] = []

    if trace.cloud_events:
        has_masks = any(e[2] is not None for e in trace.cloud_events)

        if not has_masks:
            xs = [e[0] for e in trace.cloud_events]
            ys = [e[1] for e in trace.cloud_events]
            ax.scatter(
                xs, ys,
                s=cloud_marker_size,
                color=_MULTI_CALL_COLOR,
                alpha=cloud_alpha,
                linewidths=0,
                zorder=2,
            )
            legend_handles.append(
                mpatches.Patch(color=_MULTI_CALL_COLOR, alpha=0.6, label="Cloud call")
            )
        else:
            multi = [(e[0], e[1]) for e in trace.cloud_events if e[2] and sum(e[2]) > 1]
            if multi:
                ax.scatter(
                    [p[0] for p in multi], [p[1] for p in multi],
                    s=cloud_marker_size,
                    color=_MULTI_CALL_COLOR,
                    alpha=cloud_alpha,
                    linewidths=0,
                    zorder=2,
                )
                legend_handles.append(
                    mpatches.Patch(
                        color=_MULTI_CALL_COLOR, alpha=0.6, label="Multi-DNN call"
                    )
                )

            for dnn_idx, (color, label) in enumerate(zip(_DNN_COLORS, _DNN_LABELS)):
                pts = [
                    (e[0], e[1])
                    for e in trace.cloud_events
                    if e[2] and e[2][dnn_idx] and sum(e[2]) == 1
                ]
                if pts:
                    ax.scatter(
                        [p[0] for p in pts], [p[1] for p in pts],
                        s=cloud_marker_size,
                        color=color,
                        alpha=cloud_alpha,
                        linewidths=0,
                        zorder=2,
                    )
                    legend_handles.append(
                        mpatches.Patch(color=color, alpha=0.6, label=label)
                    )

    # ── Car path — drawn ABOVE the shading so it shows through ─────────────────
    if trace.positions:
        pos = np.asarray(trace.positions)
        ax.plot(
            pos[:, 0],
            pos[:, 1],
            color=path_color,
            linewidth=path_linewidth,
            alpha=0.95,
            zorder=3,
        )

    # ── Legend & cosmetics ─────────────────────────────────────────────────────
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", framealpha=0.85, fontsize=8)

    ax.axis("off")
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"SVG trace saved → {output_path}")

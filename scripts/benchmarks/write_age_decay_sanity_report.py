#!/usr/bin/env python3
"""Write Markdown and LaTeX reports for the age-decay sanity study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Markdown and LaTeX reports for the age-decay sanity study.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest-json",
        type=str,
        default="data/benchmarks/age_decay_sanity_manifest.json",
    )
    parser.add_argument(
        "--selected-lambda-json",
        type=str,
        default="data/benchmarks/lambda_sweep_sanity_optimal.json",
    )
    parser.add_argument(
        "--best-configs-json",
        type=str,
        default="data/benchmarks/strategy_sanity_5train_best_configs.json",
    )
    parser.add_argument(
        "--best-config-summary-csv",
        type=str,
        default="data/benchmarks/eval_best_configs_sanity_3eval_lambda_summary.csv",
    )
    parser.add_argument(
        "--report-md",
        type=str,
        default="docs/age_decay_sanity_report.md",
    )
    parser.add_argument(
        "--report-tex",
        type=str,
        default="docs/age_decay_sanity_report.tex",
    )
    return parser.parse_args()


def write_text(path: Path, text: str) -> None:
    """Write UTF-8 text to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def latex_escape(text: str) -> str:
    """Escape a short text snippet for LaTeX."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    escaped = text
    for old, new in replacements.items():
        escaped = escaped.replace(old, new)
    return escaped


def frame_to_markdown(frame: pd.DataFrame) -> str:
    """Render a dataframe as a basic markdown table."""
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def best_family_table(best_summary: pd.DataFrame) -> pd.DataFrame:
    """Return the held-out best-family table."""
    return (
        best_summary.sort_values(["strategy", "rank"], kind="stable")
        .groupby("strategy", as_index=False)
        .first()
        .loc[:, [
            "strategy",
            "experiment",
            "cloud_call_rate_mean",
            "crosstrack_max_m_mean",
            "collision_free_rate",
        ]]
    )


def main() -> None:
    """Write markdown and LaTeX reports from the sanity-study artifacts."""
    args = parse_args()
    manifest = json.loads(Path(args.manifest_json).read_text())
    selected = json.loads(Path(args.selected_lambda_json).read_text())
    best_configs = json.loads(Path(args.best_configs_json).read_text())
    best_summary = pd.read_csv(args.best_config_summary_csv)
    best_table = best_family_table(best_summary)

    lambda_grid = ", ".join(str(value) for value in manifest["lambda_grid"])
    command_lines = "\n".join(
        f"- `{item['name']}`: `{item['cmd']}`"
        for item in manifest["commands"]
    )
    boundary_note = (
        f"Boundary hit: `{selected['boundary_hit']}`"
        if "boundary_hit" in selected
        else "Boundary hit: unavailable"
    )
    best_config_lines = "\n".join(
        (
            f"- `{strategy}`: `{info['experiment_name']}` with params "
            f"`{json.dumps(info['params'], sort_keys=True)}` "
            f"(train max CTE {info['train_metrics']['mean_crosstrack_max']:.4f}, "
            f"CCR {info['train_metrics']['mean_cloud_call_rate']:.3f})"
        )
        for strategy, info in sorted(best_configs["best_configs"].items())
    )

    markdown = f"""# Age-Decay Sanity Study Report

## Study Goal
Run a fast sanity-check of anchored age decay using a strict non-train boundary and `max CTE` as the primary selection metric.

## Split
- Train maps: {", ".join(manifest["train_maps"])}
- Eval maps: {", ".join(manifest["eval_maps"])}

## Fixed Settings
- Cloud latency: `{manifest["cloud_latency"]}`
- Lambda grid: `{lambda_grid}`
- Selection metric: `{manifest["selection_metric"]}`
- {boundary_note}

## What Ran
{command_lines}

## Selected Lambda
- `lambda* = {selected["selected_lambda"]}`

## Best Train-Time Family Configs
{best_config_lines}

## Held-Out Best-Family Results
{frame_to_markdown(best_table)}

## Figure Inventory
- `{manifest["figure_paths"]["aggregate_pareto"]}`
- `{manifest["figure_paths"]["per_map_pareto"]}`
- `{manifest["figure_paths"]["family_leaderboard"]}`
- `{manifest["figure_paths"]["appendix_lambda_sweep"]}`

## Manifest
- `{args.manifest_json}`
"""

    best_rows_tex = "\n".join(
        (
            f"{latex_escape(str(row.strategy))} & "
            f"{latex_escape(str(row.experiment))} & "
            f"{row.cloud_call_rate_mean:.3f} & "
            f"{row.crosstrack_max_m_mean:.4f} & "
            f"{row.collision_free_rate:.3f} \\\\"
        )
        for row in best_table.itertuples(index=False)
    )
    command_tex = "\n".join(
        f"\\item \\texttt{{{latex_escape(item['name'])}}}: \\texttt{{{latex_escape(item['cmd'])}}}"
        for item in manifest["commands"]
    )

    tex = rf"""\documentclass[conference]{{IEEEtran}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{enumitem}}
\begin{{document}}
\title{{Age-Decay Sanity Study Report}}
\maketitle

\section{{Goal}}
This report documents a fast sanity-check of anchored age decay using a strict non-train boundary and max CTE as the primary metric.

\section{{Setup}}
Train maps: {latex_escape(", ".join(manifest["train_maps"]))}.\\
Eval maps: {latex_escape(", ".join(manifest["eval_maps"]))}.\\
Cloud latency: {manifest["cloud_latency"]}.\\
Lambda grid: {latex_escape(lambda_grid)}.\\
Selection metric: {latex_escape(manifest["selection_metric"])}.\\
Boundary hit: {latex_escape(str(selected.get("boundary_hit", "unavailable")))}.\\
Selected lambda: $\lambda^* = {selected["selected_lambda"]}$.

\section{{Commands Run}}
\begin{{itemize}}[leftmargin=*]
{command_tex}
\end{{itemize}}

\section{{Best Train-Time Family Configs}}
\begin{{itemize}}[leftmargin=*]
{"".join(f"\\item \\texttt{{{latex_escape(strategy)}}}: \\texttt{{{latex_escape(info['experiment_name'])}}}, train max CTE {info['train_metrics']['mean_crosstrack_max']:.4f}, CCR {info['train_metrics']['mean_cloud_call_rate']:.3f}. " for strategy, info in sorted(best_configs["best_configs"].items()))}
\end{{itemize}}

\section{{Held-Out Best-Family Results}}
\begin{{tabular}}{{lllll}}
\toprule
Strategy & Experiment & CCR & Max CTE & Collision-Free \\
\midrule
{best_rows_tex}
\bottomrule
\end{{tabular}}

\section{{Figures}}
\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{../{manifest["figure_paths"]["aggregate_pareto_png"]}}}
\caption{{Aggregate held-out Pareto view.}}
\end{{figure}}

\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{../{manifest["figure_paths"]["per_map_pareto_png"]}}}
\caption{{Per-map Pareto panels for the three eval maps.}}
\end{{figure}}

\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{../{manifest["figure_paths"]["family_leaderboard_png"]}}}
\caption{{Family leaderboard on the held-out eval maps.}}
\end{{figure}}

\appendices
\section{{Lambda Sweep}}
\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{../{manifest["figure_paths"]["appendix_lambda_sweep_png"]}}}
\caption{{Internal lambda-sweep appendix figure.}}
\end{{figure}}

\end{{document}}
"""

    write_text(Path(args.report_md), markdown)
    write_text(Path(args.report_tex), tex)
    print(f"Wrote {args.report_md}")
    print(f"Wrote {args.report_tex}")


if __name__ == "__main__":
    main()

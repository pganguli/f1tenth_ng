#!/usr/bin/env python3
"""Write Markdown and LaTeX-source reports for the anchored age-decay study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from map_split import EVAL_MAPS, TRAIN_MAPS


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Markdown and LaTeX-source reports for the anchored age-decay study.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--selected-lambda-json",
        type=str,
        default="data/benchmarks/lambda_sweep_optimal.json",
    )
    parser.add_argument(
        "--lambda-summary-csv",
        type=str,
        default="data/benchmarks/lambda_sweep_train_13maps_summary.csv",
    )
    parser.add_argument(
        "--static-summary-csv",
        type=str,
        default="data/benchmarks/single_tier_paper_strategies_10maps_static_summary.csv",
    )
    parser.add_argument(
        "--lambda-summary-heldout-csv",
        type=str,
        default="data/benchmarks/single_tier_paper_strategies_10maps_lambda_summary.csv",
    )
    parser.add_argument(
        "--best-configs-json",
        type=str,
        default="data/benchmarks/best_configs.json",
    )
    parser.add_argument(
        "--best-config-summary-csv",
        type=str,
        default="data/benchmarks/eval_best_configs_10maps_lambda_summary.csv",
    )
    parser.add_argument(
        "--methodology-md",
        type=str,
        default="docs/age_decay_lambda_methodology.md",
    )
    parser.add_argument(
        "--results-md",
        type=str,
        default="docs/age_decay_lambda_results.md",
    )
    parser.add_argument(
        "--methodology-tex",
        type=str,
        default="docs/age_decay_lambda_methodology.tex",
    )
    parser.add_argument(
        "--results-tex",
        type=str,
        default="docs/age_decay_lambda_results.tex",
    )
    return parser.parse_args()


def latex_escape(text: str) -> str:
    """Escape a string for basic LaTeX output."""
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


def _best_target_table(summary: pd.DataFrame) -> pd.DataFrame:
    in_band = summary[summary["in_target_ccr_band"]].copy()
    if in_band.empty:
        return in_band
    return (
        in_band.sort_values(["map_name", "rank"], kind="stable")
        .groupby("map_name", as_index=False)
        .first()
    )


def write_text(path: Path, text: str) -> None:
    """Write a UTF-8 text file, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def format_lambda_table(lambda_summary: pd.DataFrame) -> str:
    """Return a compact markdown table for the lambda sweep."""
    header = "| λ | Mean Collision | Mean RMSE | Mean CCR | Target-Band Fraction |\n|---:|---:|---:|---:|---:|"
    rows = [
        (
            f"| {row.age_decay_lambda:g} | {row.mean_collision_rate:.3f} | "
            f"{row.mean_crosstrack_rmse:.4f} | {row.mean_cloud_call_rate:.3f} | "
            f"{row.frac_in_target_band:.3f} |"
        )
        for row in lambda_summary.itertuples(index=False)
    ]
    return "\n".join([header, *rows])


def frame_to_markdown(frame: pd.DataFrame) -> str:
    """Render a small dataframe as a simple markdown table without tabulate."""
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def format_best_configs(best_configs: dict[str, dict[str, object]]) -> str:
    """Return a markdown bullet list of best strategy-family configs."""
    lines = []
    for strategy, info in sorted(best_configs.items()):
        metrics = info["train_metrics"]
        lines.append(
            f"- `{strategy}`: `{info['experiment_name']}` with params `{json.dumps(info['params'], sort_keys=True)}` "
            f"(train RMSE {metrics['mean_crosstrack_rmse']:.4f}, CCR {metrics['mean_cloud_call_rate']:.3f}, "
            f"collision {metrics['mean_collision_rate']:.3f})"
        )
    return "\n".join(lines)


def main() -> None:
    """Generate the age-decay methodology/results docs."""
    args = parse_args()
    selected_payload = json.loads(Path(args.selected_lambda_json).read_text())
    best_configs_payload = json.loads(Path(args.best_configs_json).read_text())
    lambda_summary = pd.read_csv(args.lambda_summary_csv)
    static_summary = pd.read_csv(args.static_summary_csv)
    lambda_summary_heldout = pd.read_csv(args.lambda_summary_heldout_csv)
    best_config_summary = pd.read_csv(args.best_config_summary_csv)

    selected_lambda = float(selected_payload["selected_lambda"])
    static_target = _best_target_table(static_summary)
    lambda_target = _best_target_table(lambda_summary_heldout)
    best_family_rows = best_config_summary.sort_values(["strategy", "rank"], kind="stable")
    best_family_rows = best_family_rows.groupby("strategy", as_index=False).first()

    methodology_md = f"""# Anchored Age-Decay Methodology

## Objective
Tune one global planner hyperparameter, `age_decay_lambda`, for the anchored single-tier cloud-fusion rule

```text
alpha_i(A) = alpha_static_i * (sigma_e_i^2 + sigma_c_i^2) / (sigma_e_i^2 + sigma_c_i^2 + lambda * A * sigma_proc_i^2)
```

This preserves the tuned static cloud weights at cloud age `A = 0` and only changes how quickly stale cloud features lose trust after arrival.

## Fixed Inputs
- Static feature alphas: `alpha_left = 0.2`, `alpha_track = 0.2`, `alpha_heading = 0.7`
- Edge variances: `sigma_e^2 = (0.028020, 0.036518, 0.019371)`
- Cloud variances: `sigma_c^2 = (0.000518, 0.001539, 0.001140)`
- Process-noise sigmas: `sigma_proc = (0.044961, 0.067937, 0.033182)`
- Cloud latency: `5` simulator steps

## Train / Eval Split
- Training maps ({len(TRAIN_MAPS)}): {", ".join(TRAIN_MAPS)}
- Eval maps ({len(EVAL_MAPS)}): {", ".join(EVAL_MAPS)}

The eval split remains the canonical 10-map paper set to preserve continuity with the earlier figures. All tuning was restricted to the remaining 13 maps.

## Lambda Sweep
- Candidate grid: `{", ".join(str(value) for value in selected_payload["lambda_grid"])}`
- Representative strategies:
  - `always_hit`
  - `fixed_interval_k5`
  - `fixed_bernoulli_p60`
  - `bernoulli_max_miss_p60_m5`
- Trials per representative configuration: `3`
- Selection rule:
  1. if any lambda has zero mean collision rate, only compare those lambdas
  2. minimize mean crosstrack RMSE
  3. break ties by minimizing mean absolute distance from target CCR `0.60`

Selected value: `lambda* = {selected_lambda:g}`

### Lambda Sweep Summary
{format_lambda_table(lambda_summary)}

## Strategy Optimization
- Expanded training grid from `scripts/benchmarks/optimize_hyperparameters.py`
- Total configs in current repo: `118`
- Coarse stage: `1` trial per config on all 13 training maps
- Confirmation stage: rerun the winning configuration for each of the 8 strategy families with `3` trials on all 13 training maps

### Best Training Config Per Strategy Family
{format_best_configs(best_configs_payload["best_configs"])}

## Held-Out Evaluation
- Static baseline full canonical suite: `27` configs, `5` trials each on the 10 eval maps
- Anchored-decay full canonical suite: `27` configs, `5` trials each on the 10 eval maps
- Anchored-decay best-config eval: `8` configs, `10` trials each on the 10 eval maps

All paper-facing held-out figures use repeated trials and uncertainty bars derived from the per-config variance across those repeated runs.
"""

    results_md = f"""# Anchored Age-Decay Results

## Headline
- Selected global decay parameter: `lambda* = {selected_lambda:g}`
- Held-out static target-band winners: `{len(static_target)}` maps
- Held-out anchored-decay target-band winners: `{len(lambda_target)}` maps
- Best-config strategy families evaluated on held-out maps: `{len(best_family_rows)}`

## Held-Out Target-Band Winners

### Static Fusion
{frame_to_markdown(static_target[['map_name', 'experiment', 'strategy', 'cloud_call_rate_mean', 'crosstrack_rmse_m_mean']])}

### Anchored Age Decay
{frame_to_markdown(lambda_target[['map_name', 'experiment', 'strategy', 'cloud_call_rate_mean', 'crosstrack_rmse_m_mean']])}

## Held-Out Best Configs
{frame_to_markdown(best_family_rows[['strategy', 'experiment', 'cloud_call_rate_mean', 'crosstrack_rmse_m_mean', 'collision_free_rate']])}

## Figure Inventory
- `data/benchmarks/paper_figures_10maps_age_decay/lambda_train_sweep.*`
- `data/benchmarks/paper_figures_10maps_age_decay/static_vs_lambda_target_band.*`
- `data/benchmarks/paper_figures_10maps_age_decay/strategy_family_tradeoff.*`
- `data/benchmarks/paper_figures_10maps_age_decay/strategy_win_summary.*`
- `data/benchmarks/paper_figures_10maps_age_decay/alpha_decay_curves.*`
- `data/benchmarks/paper_figures_10maps_lambda_curated/*` if the curated plotter is rerun on the anchored-decay held-out summary

## Notes
- `collision_free_rate_ci95`, `crosstrack_rmse_m_ci95`, `cloud_call_rate_ci95`, and `lap_time_s_ci95` are exported in the held-out summary CSVs for uncertainty-aware downstream plotting.
- The canonical exploratory/oracle plotter was not used for these publication-facing figures.
"""

    methodology_tex = rf"""\documentclass[conference]{{IEEEtran}}
\usepackage{{booktabs}}
\usepackage{{amsmath}}
\begin{{document}}
\title{{Anchored Age-Decay Methodology}}
\maketitle
\section{{Objective}}
We tune one global planner hyperparameter, $\lambda$, for the anchored age-decay fusion rule
\[
\alpha_i(A)=\alpha^{{static}}_i \frac{{\sigma^2_{{e,i}}+\sigma^2_{{c,i}}}}{{\sigma^2_{{e,i}}+\sigma^2_{{c,i}}+\lambda A \sigma^2_{{proc,i}}}}.
\]
This preserves the tuned static cloud weights at age $A=0$ and only changes how fast stale cloud features decay after arrival.

\section{{Train / Eval Split}}
Training maps ({len(TRAIN_MAPS)}): {latex_escape(", ".join(TRAIN_MAPS))}.\\
Eval maps ({len(EVAL_MAPS)}): {latex_escape(", ".join(EVAL_MAPS))}.

\section{{Lambda Sweep}}
Candidate grid: {latex_escape(", ".join(str(value) for value in selected_payload["lambda_grid"]))}.\\
Representative strategies: {latex_escape(", ".join(["always_hit", "fixed_interval_k5", "fixed_bernoulli_p60", "bernoulli_max_miss_p60_m5"]))}.\\
Selected value: $\lambda^* = {selected_lambda:g}$.

\section{{Strategy Optimization}}
The current expanded optimizer grid contains 118 configurations and was run on all 13 training maps before confirming one winning configuration for each strategy family.

\section{{Held-Out Evaluation}}
The held-out study includes the full 27-config static baseline sweep, the full 27-config anchored-decay sweep, and the 8-family best-config anchored-decay eval. Repeated trials are used for all publication-facing uncertainty bars.
\end{{document}}
"""

    results_tex = rf"""\documentclass[conference]{{IEEEtran}}
\usepackage{{booktabs}}
\begin{{document}}
\title{{Anchored Age-Decay Results}}
\maketitle
\section{{Headline}}
Selected global decay parameter: $\lambda^* = {selected_lambda:g}$.

\section{{Held-Out Target-Band Winners}}
Static fusion produced {len(static_target)} target-band winners on the 10 held-out maps; anchored age decay produced {len(lambda_target)} target-band winners.

\section{{Artifacts}}
Primary figures are written under \texttt{{data/benchmarks/paper\_figures\_10maps\_age\_decay/}}. The held-out summary CSVs contain the exported uncertainty columns used by those figures.
\end{{document}}
"""

    write_text(Path(args.methodology_md), methodology_md)
    write_text(Path(args.results_md), results_md)
    write_text(Path(args.methodology_tex), methodology_tex)
    write_text(Path(args.results_tex), results_tex)

    print(f"Wrote {args.methodology_md}")
    print(f"Wrote {args.results_md}")
    print(f"Wrote {args.methodology_tex}")
    print(f"Wrote {args.results_tex}")


if __name__ == "__main__":
    main()

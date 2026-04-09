#!/usr/bin/env python3
"""Run the full age-decay sanity-study pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from map_split import SANITY_EVAL_MAPS, SANITY_TRAIN_MAPS


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data/benchmarks"
DOCS_DIR = ROOT / "docs"
PYTHONPATH = "packages/f110_gym/src:packages/f110_planning/src:packages/f110_scripts/src"


def ensure_tectonic() -> None:
    """Install Tectonic via Homebrew if it is missing."""
    if shutil_which("tectonic") is not None:
        return
    if shutil_which("brew") is None:
        raise RuntimeError("tectonic is not installed and Homebrew is unavailable")
    subprocess.run(["brew", "install", "tectonic"], check=True)


def shutil_which(name: str) -> str | None:
    """Return an executable path if present."""
    from shutil import which

    return which(name)


def run_command(name: str, args: list[str], commands: list[dict[str, str]]) -> None:
    """Run one pipeline command and append it to the manifest command log."""
    env = os.environ.copy()
    env["PYTHONPATH"] = PYTHONPATH
    cmd = [sys.executable, *args]
    commands.append({"name": name, "cmd": shlex.join(cmd)})
    print(f"\n[{name}] {shlex.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


def main() -> None:
    """Execute the sanity-study pipeline and record a manifest."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    train_maps = ",".join(SANITY_TRAIN_MAPS)
    eval_maps = ",".join(SANITY_EVAL_MAPS)
    lambda_json = OUT_DIR / "lambda_sweep_sanity_optimal.json"
    best_configs_json = OUT_DIR / "strategy_sanity_5train_best_configs.json"
    figure_dir = OUT_DIR / "paper_figures_sanity_3eval"
    manifest_path = OUT_DIR / "age_decay_sanity_manifest.json"
    report_md = DOCS_DIR / "age_decay_sanity_report.md"
    report_tex = DOCS_DIR / "age_decay_sanity_report.tex"
    report_pdf = DOCS_DIR / "age_decay_sanity_report.pdf"

    commands: list[dict[str, str]] = []
    start = time.strftime("%Y-%m-%d %H:%M:%S")

    run_command(
        "lambda_sweep",
        [
            "scripts/benchmarks/sweep_age_decay_lambda.py",
            "--maps",
            train_maps,
            "--lambda-grid",
            "0,1,4,16,64",
            "--trials",
            "2",
            "--workers",
            "5",
            "--selection-metric",
            "max_cte",
            "--output-stem",
            "lambda_sweep_sanity_5train",
            "--selected-lambda-json",
            str(lambda_json),
        ],
        commands,
    )

    selected_lambda = float(json.loads(lambda_json.read_text())["selected_lambda"])

    run_command(
        "train_family_search",
        [
            "scripts/benchmarks/optimize_hyperparameters.py",
            "--maps",
            train_maps,
            "--grid-profile",
            "sanity",
            "--selection-metric",
            "max_cte",
            "--trials",
            "1",
            "--workers",
            "5",
            "--age-decay-lambda",
            f"{selected_lambda:g}",
            "--output-stem",
            "strategy_sanity_5train",
        ],
        commands,
    )

    run_command(
        "eval_static_full_suite",
        [
            "scripts/benchmarks/benchmark_single_tier_paper_strategies.py",
            "--maps",
            eval_maps,
            "--trials",
            "3",
            "--workers",
            "3",
            "--selection-metric",
            "max_cte",
            "--output-stem",
            "single_tier_sanity_3eval_static",
        ],
        commands,
    )

    run_command(
        "eval_lambda_full_suite",
        [
            "scripts/benchmarks/benchmark_single_tier_paper_strategies.py",
            "--maps",
            eval_maps,
            "--trials",
            "3",
            "--workers",
            "3",
            "--selection-metric",
            "max_cte",
            "--age-decay-lambda",
            f"{selected_lambda:g}",
            "--output-stem",
            "single_tier_sanity_3eval_lambda",
        ],
        commands,
    )

    run_command(
        "eval_lambda_best_configs",
        [
            "scripts/benchmarks/benchmark_eval_best_configs.py",
            "--configs-json",
            str(best_configs_json),
            "--maps",
            eval_maps,
            "--trials",
            "5",
            "--workers",
            "3",
            "--selection-metric",
            "max_cte",
            "--age-decay-lambda",
            f"{selected_lambda:g}",
            "--output-stem",
            "eval_best_configs_sanity_3eval_lambda",
        ],
        commands,
    )

    run_command(
        "plot_figures",
        [
            "scripts/benchmarks/plot_age_decay_sanity_figures.py",
            "--lambda-raw-csv",
            "data/benchmarks/lambda_sweep_sanity_5train.csv",
            "--selected-lambda-json",
            str(lambda_json),
            "--static-raw-csv",
            "data/benchmarks/single_tier_sanity_3eval_static.csv",
            "--lambda-raw-heldout-csv",
            "data/benchmarks/single_tier_sanity_3eval_lambda.csv",
            "--best-config-raw-csv",
            "data/benchmarks/eval_best_configs_sanity_3eval_lambda.csv",
            "--output-dir",
            str(figure_dir),
        ],
        commands,
    )

    manifest = {
        "generated_at": start,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip(),
        "train_maps": SANITY_TRAIN_MAPS,
        "eval_maps": SANITY_EVAL_MAPS,
        "selection_metric": "max_cte",
        "cloud_latency": 5,
        "lambda_grid": [0, 1, 4, 16, 64],
        "commands": commands,
        "selected_lambda": selected_lambda,
        "figure_paths": {
            "aggregate_pareto": "data/benchmarks/paper_figures_sanity_3eval/aggregate_strategy_pareto.pdf",
            "aggregate_pareto_png": "data/benchmarks/paper_figures_sanity_3eval/aggregate_strategy_pareto.png",
            "per_map_pareto": "data/benchmarks/paper_figures_sanity_3eval/per_map_strategy_pareto.pdf",
            "per_map_pareto_png": "data/benchmarks/paper_figures_sanity_3eval/per_map_strategy_pareto.png",
            "family_leaderboard": "data/benchmarks/paper_figures_sanity_3eval/family_comparison_leaderboard.pdf",
            "family_leaderboard_png": "data/benchmarks/paper_figures_sanity_3eval/family_comparison_leaderboard.png",
            "appendix_lambda_sweep": "data/benchmarks/paper_figures_sanity_3eval/appendix_lambda_sweep.pdf",
            "appendix_lambda_sweep_png": "data/benchmarks/paper_figures_sanity_3eval/appendix_lambda_sweep.png",
        },
        "artifact_paths": {
            "selected_lambda_json": str(lambda_json.relative_to(ROOT)),
            "best_configs_json": str(best_configs_json.relative_to(ROOT)),
            "static_raw_csv": "data/benchmarks/single_tier_sanity_3eval_static.csv",
            "lambda_raw_csv": "data/benchmarks/single_tier_sanity_3eval_lambda.csv",
            "best_config_raw_csv": "data/benchmarks/eval_best_configs_sanity_3eval_lambda.csv",
            "report_md": str(report_md.relative_to(ROOT)),
            "report_tex": str(report_tex.relative_to(ROOT)),
            "report_pdf": str(report_pdf.relative_to(ROOT)),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    run_command(
        "write_report_sources",
        [
            "scripts/benchmarks/write_age_decay_sanity_report.py",
            "--manifest-json",
            str(manifest_path),
            "--selected-lambda-json",
            str(lambda_json),
            "--best-configs-json",
            str(best_configs_json),
            "--best-config-summary-csv",
            "data/benchmarks/eval_best_configs_sanity_3eval_lambda_summary.csv",
            "--report-md",
            str(report_md),
            "--report-tex",
            str(report_tex),
        ],
        commands,
    )

    ensure_tectonic()
    compile_cmd = ["tectonic", str(report_tex)]
    commands.append({"name": "compile_pdf", "cmd": shlex.join(compile_cmd)})
    print(f"\n[compile_pdf] {shlex.join(compile_cmd)}")
    subprocess.run(compile_cmd, check=True, cwd=ROOT)

    manifest["commands"] = commands
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()

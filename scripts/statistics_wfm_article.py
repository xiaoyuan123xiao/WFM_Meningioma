"""Article-aligned paired statistics for WFM ablation results.

Inputs are per-case CSV files produced by evaluate_registered.py:

    per_case_registered_<model>.csv

The script reports paired Wilcoxon p-values, bootstrap 95% CIs for paired
mean differences, Cliff's delta, and Benjamini-Hochberg q-values.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METRIC_ALIASES = {"dice": "Dice", "roi_mae": "ROI_MAE"}


def normalize_columns(frame):
    frame = frame.rename(columns={c: METRIC_ALIASES.get(c.lower(), c) for c in frame.columns})
    return frame


def cliff_delta(first, second):
    differences = np.asarray(first) - np.asarray(second)
    return float((np.sum(differences > 0) - np.sum(differences < 0)) / len(differences))


def bootstrap_mean_difference(first, second, num_bootstrap, seed):
    rng = np.random.default_rng(seed)
    n = len(first)
    values = []
    first = np.asarray(first)
    second = np.asarray(second)
    for _ in range(num_bootstrap):
        indices = rng.integers(0, n, n)
        values.append((first[indices] - second[indices]).mean())
    return float(np.mean(values)), float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def benjamini_hochberg(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    m = len(p_values)
    adjusted = np.empty(m, dtype=float)
    running = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        running = min(running, ranked[i] * m / rank)
        adjusted[i] = running
    result = np.empty(m, dtype=float)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def load_results(results_dir):
    results = {}
    for path in sorted(Path(results_dir).glob("per_case_registered_*.csv")):
        model = path.stem.replace("per_case_registered_", "")
        frame = normalize_columns(pd.read_csv(path))
        results[model] = frame
    return results

def parse_comparison(value):
    if " vs " in value:
        left, right = value.split(" vs ", 1)
    elif ":" in value:
        left, right = value.split(":", 1)
    else:
        raise ValueError(f"Comparison must be 'A vs B' or 'A:B', got {value}")
    return left.strip(), right.strip()


def main():
    parser = argparse.ArgumentParser(description="Article-aligned paired statistics")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output_csv", default="pairwise_wilcoxon.csv")
    parser.add_argument("--baseline", default="original_500")
    parser.add_argument("--comparison", action="append", default=[])
    parser.add_argument("--metrics", nargs="*", default=["LPIPS", "ROI_MAE", "MSE", "Dice"])
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = load_results(args.results_dir)
    if not results:
        raise RuntimeError(f"No per_case_registered_*.csv files found in {args.results_dir}")

    if args.comparison:
        comparisons = [parse_comparison(value) for value in args.comparison]
    else:
        comparisons = [(model, args.baseline) for model in sorted(results) if model != args.baseline]

    rows = []
    for first_model, second_model in comparisons:
        if first_model not in results or second_model not in results:
            print(f"Skipping {first_model} vs {second_model}: missing result file")
            continue
        for metric in args.metrics:
            if metric not in results[first_model] or metric not in results[second_model]:
                continue
            merged = results[first_model][["subject", metric]].merge(
                results[second_model][["subject", metric]],
                on="subject",
                suffixes=("_first", "_second"),
            ).dropna()
            if len(merged) < 2:
                continue
            first = merged[f"{metric}_first"].to_numpy(dtype=float)
            second = merged[f"{metric}_second"].to_numpy(dtype=float)
            differences = first - second
            if np.allclose(differences, 0):
                p_value = 1.0
            else:
                p_value = float(wilcoxon(first, second, zero_method="wilcox", alternative="two-sided").pvalue)
            effect, ci_lo, ci_hi = bootstrap_mean_difference(
                first, second, args.bootstrap, args.seed
            )
            rows.append({
                "comparison": f"{first_model} vs {second_model}",
                "metric": metric,
                "effect": effect,
                "CI_lo": ci_lo,
                "CI_hi": ci_hi,
                "p": p_value,
                "Cliff_delta": cliff_delta(first, second),
                "n": len(merged),
            })

    if not rows:
        raise RuntimeError("No valid comparisons were computed")

    output = pd.DataFrame(rows)
    output["q_BH"] = benjamini_hochberg(output["p"].to_numpy(dtype=float))
    output.to_csv(args.output_csv, index=False)
    print(output.to_string(index=False))
    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()

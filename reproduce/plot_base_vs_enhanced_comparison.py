import argparse
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_COLOR = "#355C7D"
ENH_COLOR = "#F67280"
SC_COLOR = "#2E8B57"


def apply_professional_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D0D7DE",
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#E5E7EB",
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            "font.size": 12,
            "axes.titlesize": 18,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D0D7DE")
    ax.spines["bottom"].set_color("#D0D7DE")
    ax.grid(axis="y", alpha=0.9)
    ax.grid(axis="x", visible=False)


def annotate_bars(ax, bars, dy=0.012):
    for bar in bars:
        h = bar.get_height()
        if np.isnan(h):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + dy,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#111827",
        )


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_markdown_report(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    def _extract(pattern, default=np.nan):
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if not m:
            return default
        try:
            return float(m.group(1))
        except Exception:
            return default

    return {
        "faithfulness": {
            "sentence": {
                "balanced_accuracy": _extract(r"Sentence\s+bAcc:\s*([0-9]*\.?[0-9]+)"),
                "macro_f1": _extract(r"Sentence\s+Macro-F1:\s*([0-9]*\.?[0-9]+)"),
            },
            "summary": {
                "pearson": {"statistic": _extract(r"Summary\s+Pearson:\s*([0-9]*\.?[0-9]+)")},
                "spearman": {"statistic": _extract(r"Summary\s+Spearman:\s*([0-9]*\.?[0-9]+)")},
            },
            "system": {
                "rank_correlation": {
                    "statistic": _extract(r"System\s+Rank\s+Corr:\s*([0-9]*\.?[0-9]+)")
                }
            },
            "success_ratio": _extract(r"##\s*Faithfulness[\s\S]*?Success\s+Ratio:\s*([0-9]*\.?[0-9]+)"),
        },
        "completeness": {
            "summary": {
                "pearson": {"statistic": _extract(r"##\s*Completeness[\s\S]*?Summary\s+Pearson:\s*([0-9]*\.?[0-9]+)")},
                "spearman": {"statistic": _extract(r"##\s*Completeness[\s\S]*?Summary\s+Spearman:\s*([0-9]*\.?[0-9]+)")},
            },
            "system": {
                "rank_correlation": {
                    "statistic": _extract(r"##\s*Completeness[\s\S]*?System\s+Rank\s+Corr:\s*([0-9]*\.?[0-9]+)")
                }
            },
        },
        "conciseness": {
            "summary": {
                "pearson": {"statistic": _extract(r"##\s*Conciseness[\s\S]*?Summary\s+Pearson:\s*([0-9]*\.?[0-9]+)")},
                "spearman": {"statistic": _extract(r"##\s*Conciseness[\s\S]*?Summary\s+Spearman:\s*([0-9]*\.?[0-9]+)")},
            },
            "system": {
                "rank_correlation": {
                    "statistic": _extract(r"##\s*Conciseness[\s\S]*?System\s+Rank\s+Corr:\s*([0-9]*\.?[0-9]+)")
                }
            },
            "success_ratio": _extract(r"##\s*Conciseness[\s\S]*?Success\s+Ratio:\s*([0-9]*\.?[0-9]+)"),
        },
    }


def load_report(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Report file not found: {path}")
    if path.lower().endswith(".json"):
        return load_json(path)
    if path.lower().endswith(".md"):
        return _parse_markdown_report(path)
    raise ValueError(f"Unsupported report extension for {path}; use .json or .md")


def get_nested(dct, keys, default=np.nan):
    cur = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    try:
        return float(cur)
    except Exception:
        return default


def build_metric_rows(base_report, enhanced_report, sc_report):
    rows = []

    faithfulness_metrics = [
        ("Faithfulness", "bAcc", ["faithfulness", "sentence", "balanced_accuracy"]),
        ("Faithfulness", "Macro-F1", ["faithfulness", "sentence", "macro_f1"]),
        ("Faithfulness", "Pearson", ["faithfulness", "summary", "pearson", "statistic"]),
        ("Faithfulness", "Spearman", ["faithfulness", "summary", "spearman", "statistic"]),
        ("Faithfulness", "SysRank", ["faithfulness", "system", "rank_correlation", "statistic"]),
        ("Faithfulness", "Success", ["faithfulness", "success_ratio"]),
    ]

    comp_metrics = [
        ("Completeness", "Pearson", ["completeness", "summary", "pearson", "statistic"]),
        ("Completeness", "Spearman", ["completeness", "summary", "spearman", "statistic"]),
        ("Completeness", "SysRank", ["completeness", "system", "rank_correlation", "statistic"]),
    ]

    conc_metrics = [
        ("Conciseness", "Pearson", ["conciseness", "summary", "pearson", "statistic"]),
        ("Conciseness", "Spearman", ["conciseness", "summary", "spearman", "statistic"]),
        ("Conciseness", "SysRank", ["conciseness", "system", "rank_correlation", "statistic"]),
        ("Conciseness", "Success", ["conciseness", "success_ratio"]),
    ]

    for section, metric, keys in faithfulness_metrics + comp_metrics + conc_metrics:
        rows.append(
            {
                "Section": section,
                "Metric": metric,
                "Base": get_nested(base_report, keys),
                "Enhanced": get_nested(enhanced_report, keys),
                "SC": get_nested(sc_report, keys),
            }
        )

    return pd.DataFrame(rows)


def plot_faithfulness(df, out_path, method_base, method_enhanced, method_sc):
    sub = df[df["Section"] == "Faithfulness"].copy()
    labels = sub["Metric"].tolist()

    x = np.arange(len(labels))
    width = 0.22

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(
        x - width,
        sub["Base"],
        width,
        label=method_base,
        color=BASE_COLOR,
        edgecolor="white",
        linewidth=0.8,
    )
    bars2 = ax.bar(
        x,
        sub["Enhanced"],
        width,
        label=method_enhanced,
        color=ENH_COLOR,
        edgecolor="white",
        linewidth=0.8,
    )
    bars3 = ax.bar(
        x + width,
        sub["SC"],
        width,
        label=method_sc,
        color=SC_COLOR,
        edgecolor="white",
        linewidth=0.8,
    )

    ax.set_title("Faithfulness Metrics: Base vs Enhanced", pad=12)
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    style_axis(ax)
    ax.legend(loc="upper center", ncol=3, frameon=False)
    annotate_bars(ax, bars1, dy=0.010)
    annotate_bars(ax, bars2, dy=0.024)
    annotate_bars(ax, bars3, dy=0.038)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def _plot_section(ax, sub, title, method_base, method_enhanced, method_sc):
    labels = sub["Metric"].tolist()
    x = np.arange(len(labels))
    width = 0.22

    bars1 = ax.bar(
        x - width,
        sub["Base"],
        width,
        label=method_base,
        color=BASE_COLOR,
        edgecolor="white",
        linewidth=0.8,
    )
    bars2 = ax.bar(
        x,
        sub["Enhanced"],
        width,
        label=method_enhanced,
        color=ENH_COLOR,
        edgecolor="white",
        linewidth=0.8,
    )
    bars3 = ax.bar(
        x + width,
        sub["SC"],
        width,
        label=method_sc,
        color=SC_COLOR,
        edgecolor="white",
        linewidth=0.8,
    )

    ax.set_title(title, pad=10)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    style_axis(ax)
    annotate_bars(ax, bars1, dy=0.010)
    annotate_bars(ax, bars2, dy=0.024)
    annotate_bars(ax, bars3, dy=0.038)


def plot_completeness_conciseness(df, out_path, method_base, method_enhanced, method_sc):
    comp = df[df["Section"] == "Completeness"].copy()
    conc = df[df["Section"] == "Conciseness"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    _plot_section(axes[0], comp, "Completeness Metrics", method_base, method_enhanced, method_sc)
    _plot_section(axes[1], conc, "Conciseness Metrics", method_base, method_enhanced, method_sc)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot baseline vs enhanced evaluation metric comparisons")
    parser.add_argument(
        "--base-report",
        default="reproduce/results/reports/base/evaluation-report.json",
        help="Path to baseline evaluation-report.json",
    )
    parser.add_argument(
        "--enhanced-report",
        default="reproduce/results/reports/enhanced_qwen_both/evaluation-report.json",
        help="Path to enhanced evaluation-report.json",
    )
    parser.add_argument(
        "--sc-report",
        default="reproduce/results/reports/sc_runs/evaluation-report.md",
        help="Path to SC run report (.json or .md)",
    )
    parser.add_argument(
        "--output-dir",
        default="reproduce/results/reports/comparison_base_vs_enhanced_qwen_both",
        help="Directory for comparison outputs",
    )
    parser.add_argument("--base-label", default="base")
    parser.add_argument("--enhanced-label", default="enhanced_qwen_both")
    parser.add_argument("--sc-label", default="sc_runs")
    args = parser.parse_args()
    apply_professional_style()

    os.makedirs(args.output_dir, exist_ok=True)

    base_report = load_report(args.base_report)
    enhanced_report = load_report(args.enhanced_report)
    sc_report = load_report(args.sc_report)
    df = build_metric_rows(base_report, enhanced_report, sc_report)

    csv_path = os.path.join(args.output_dir, "comparison_metrics.csv")
    df.to_csv(csv_path, index=False)

    faithfulness_plot = os.path.join(args.output_dir, "plot_compare_faithfulness.pdf")
    comp_conc_plot = os.path.join(args.output_dir, "plot_compare_completeness_conciseness.pdf")

    plot_faithfulness(df, faithfulness_plot, args.base_label, args.enhanced_label, args.sc_label)
    plot_completeness_conciseness(df, comp_conc_plot, args.base_label, args.enhanced_label, args.sc_label)

    print("[Comparison Outputs]")
    print("- CSV:", csv_path)
    print("- Faithfulness plot:", faithfulness_plot)
    print("- Completeness/Conciseness plot:", comp_conc_plot)


if __name__ == "__main__":
    main()

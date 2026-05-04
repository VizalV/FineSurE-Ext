"""
Professional plotting for reference-vs-BART completeness/conciseness results.

This script reads the CSV produced by
reproduce/evaluate_completeness_conciseness_reference_vs_bart.py and generates
publication-style charts with clear spacing for titles and legends.
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_COLOR = "#355C7D"
ENH_COLOR = "#F67280"
COMP_COLOR = "#2E8B57"
CONC_COLOR = "#E76F51"
SUCCESS_COLOR = "#4F46E5"


def apply_professional_style() -> None:
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
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D0D7DE")
    ax.spines["bottom"].set_color("#D0D7DE")
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", visible=False)


def annotate_bars(ax, bars, fmt="{:.1f}%", offset=1.2) -> None:
    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9,
            color="#1F2937",
        )


def load_results(csv_path: str) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Summary CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    for col in ["SuccessRate(%)", "Completeness(%)", "Conciseness(%)"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _ordered_row(df: pd.DataFrame, domain: str, source: str) -> pd.Series:
    row = df[(df["Domain"] == domain) & (df["SummarySource"] == source)]
    if row.empty:
        raise ValueError(f"Missing row for {domain} / {source}")
    return row.iloc[0]


def plot_headline_metrics(df: pd.DataFrame, output_dir: str) -> None:
    domains = ["SAMSum", "GovReport"]
    sources = ["reference", "bart"]
    metric_specs = [
        ("Completeness(%)", "Completeness"),
        ("Conciseness(%)", "Conciseness"),
        ("SuccessRate(%)", "Alignment Success Rate"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.8), sharey=True)

    for ax, (metric_col, title) in zip(axes, metric_specs):
        x = np.arange(len(domains))
        width = 0.34

        ref_vals = [float(_ordered_row(df, domain, "reference")[metric_col]) for domain in domains]
        bart_vals = [float(_ordered_row(df, domain, "bart")[metric_col]) for domain in domains]

        ref_bars = ax.bar(
            x - width / 2,
            ref_vals,
            width,
            label="reference" if ax is axes[0] else None,
            color=BASE_COLOR,
            edgecolor="white",
            linewidth=0.8,
        )
        bart_bars = ax.bar(
            x + width / 2,
            bart_vals,
            width,
            label="bart" if ax is axes[0] else None,
            color=ENH_COLOR,
            edgecolor="white",
            linewidth=0.8,
        )

        ax.set_title(title, pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(domains)
        ax.set_ylim(0, 110)
        ax.set_ylabel("Percent")
        style_axis(ax)
        annotate_bars(ax, ref_bars)
        annotate_bars(ax, bart_bars)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Reference vs BART: Completeness / Conciseness", y=0.992, fontsize=19)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.84])

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "reference_vs_bart_headline_metrics.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_path)


def plot_detailed_metrics(df: pd.DataFrame, output_dir: str) -> None:
    domains = ["SAMSum", "GovReport"]
    sources = ["reference", "bart"]

    labels = [f"{domain}\n({source})" for domain in domains for source in sources]
    x = np.arange(len(labels))
    width = 0.34

    completeness_vals = []
    conciseness_vals = []
    for domain in domains:
        for source in sources:
            row = _ordered_row(df, domain, source)
            completeness_vals.append(float(row["Completeness(%)"]))
            conciseness_vals.append(float(row["Conciseness(%)"]))

    fig, ax = plt.subplots(1, 1, figsize=(8.6, 5.8))

    bars1 = ax.bar(x - width / 2, completeness_vals, width, label="Completeness", color=COMP_COLOR, edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + width / 2, conciseness_vals, width, label="Conciseness", color=CONC_COLOR, edgecolor="white", linewidth=0.8)
    ax.set_title("Completeness vs Conciseness", pad=12)
    ax.set_ylim(0, 110)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Percent")
    style_axis(ax)
    annotate_bars(ax, bars1)
    annotate_bars(ax, bars2)

    handles, labels = ax.get_legend_handles_labels()
    fig.suptitle("Reference vs BART: Completeness and Conciseness", y=0.992, fontsize=19)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.84])

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "reference_vs_bart_completeness_conciseness.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot completeness/conciseness results for reference vs bart")
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="reproduce/results/cross_domain_reference_vs_model/bart/completeness_conciseness/reference_vs_bart_completeness_conciseness.csv",
        help="Path to the CSV produced by the evaluation script",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reproduce/results/cross_domain_reference_vs_model/bart/completeness_conciseness/plots",
        help="Directory where plots will be written",
    )
    args = parser.parse_args()

    apply_professional_style()
    df = load_results(args.summary_csv)
    plot_headline_metrics(df, args.output_dir)
    plot_detailed_metrics(df, args.output_dir)


if __name__ == "__main__":
    main()

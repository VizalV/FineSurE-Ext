"""
Professional plotting for cross-domain robustness results.

Reads the saved cross_domain_summary.json or cross_domain_summary.csv produced by
reproduce/cross_domain_robustness.py and recreates the evaluation figures:
1) Error type distribution across domains
2) Faithfulness vs error rate, plus success rate by domain
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def apply_professional_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (13, 7),
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "axes.linewidth": 1.0,
            "axes.edgecolor": "#2f3b52",
            "axes.facecolor": "#ffffff",
            "figure.facecolor": "#f6f8fb",
            "axes.grid": True,
            "grid.alpha": 0.16,
            "grid.linestyle": "--",
            "grid.color": "#8ea0b8",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "text.color": "#1f2937",
            "axes.labelcolor": "#1f2937",
            "xtick.color": "#1f2937",
            "ytick.color": "#1f2937",
            "font.family": ["DejaVu Sans", "sans-serif"],
        }
    )


def style_axis(ax) -> None:
    ax.set_facecolor("#ffffff")
    ax.grid(True, axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")


def load_summary(output_dir: str) -> Dict[str, Any]:
    json_path = os.path.join(output_dir, "cross_domain_summary.json")
    csv_path = os.path.join(output_dir, "cross_domain_summary.csv")

    summary: Dict[str, Any] = {}
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)

    if os.path.exists(csv_path):
        summary["csv"] = pd.read_csv(csv_path)

    if not summary:
        raise FileNotFoundError(
            f"Could not find cross_domain_summary.json or cross_domain_summary.csv in {output_dir}"
        )

    return summary


def extract_analysis(summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    if "analysis" in summary:
        return summary["analysis"]

    if "csv" in summary:
        analysis: Dict[str, Dict[str, Any]] = {}
        for _, row in summary["csv"].iterrows():
            domain = row["Domain"]
            analysis[domain] = {
                "total": int(row["Total Samples"]),
                "successful": int(row["Successful"]),
                "success_rate": float(str(row["Success Rate (%)"]).replace("%", "")) / 100.0,
                "faithfulness": float(str(row["Faithfulness (%)"]).replace("%", "")) / 100.0,
                "error_rate": float(str(row["Error Rate (%)"]).replace("%", "")) / 100.0,
                "avg_sentences": float(row["Avg Sentences"]),
                "num_error_categories": int(str(row["Error Categories"]).split("/")[0]),
                "error_type_counts": {},
                "total_predictions": 0,
            }
        return analysis

    raise ValueError("Summary file does not contain analysis data.")


def plot_error_type_distribution(summary: Dict[str, Any], output_dir: str) -> None:
    analysis = extract_analysis(summary)
    domains = list(analysis.keys())
    if not domains:
        return

    error_type_order = [
        "no error",
        "entity error",
        "predicate error",
        "out-of-context error",
        "circumstantial error",
        "coreference error",
        "linking error",
        "grammatical error",
        "other error",
    ]
    display_names = {
        "no error": "No Error",
        "entity error": "Entity",
        "predicate error": "Predicate",
        "out-of-context error": "Out-of-Context",
        "circumstantial error": "Circumstantial",
        "coreference error": "Coreference",
        "linking error": "Linking",
        "grammatical error": "Grammar",
        "other error": "Other",
    }

    fig, axes = plt.subplots(1, len(domains), figsize=(5.6 * len(domains), 5.1), squeeze=False)
    axes = axes[0]

    for ax, domain in zip(axes, domains):
        counts = analysis[domain].get("error_type_counts", {})
        values = [counts.get(name, 0) for name in error_type_order]
        labels = [display_names[name] for name in error_type_order]

        colors = ["#1f77b4" if name == "no error" else "#4f7cac" for name in error_type_order]
        bars = ax.bar(labels, values, color=colors, edgecolor="#27405c", linewidth=0.7)
        ax.set_title(f"{domain}\n(n={analysis[domain]['total']})", pad=10)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=38)
        ax.margins(x=0.04)
        style_axis(ax)

        for bar in bars:
            height = bar.get_height()
            if height <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + max(values) * 0.015,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle("Cross-Domain Error Type Distribution", y=0.98, fontsize=18, fontweight="semibold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "error_type_distribution.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_faithfulness_comparison(summary: Dict[str, Any], output_dir: str) -> None:
    analysis = extract_analysis(summary)
    domains = list(analysis.keys())
    if not domains:
        return

    faithfulness = [analysis[d]["faithfulness"] for d in domains]
    error_rates = [analysis[d]["error_rate"] for d in domains]
    success_rates = [analysis[d]["success_rate"] for d in domains]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    x_pos = np.arange(len(domains))
    width = 0.36

    ax1.bar(x_pos - width / 2, faithfulness, width, label="Faithfulness", color="#2e8b57", edgecolor="#1f5e3b")
    ax1.bar(x_pos + width / 2, error_rates, width, label="Error Rate", color="#d1495b", edgecolor="#8f2f3b")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(domains)
    ax1.set_ylabel("Rate")
    ax1.set_ylim(0, 1.12)
    ax1.set_title("Faithfulness vs Error Rate", pad=10)
    ax1.legend(loc="upper right")
    style_axis(ax1)

    for idx, value in enumerate(faithfulness):
        ax1.text(idx - width / 2, value + 0.03, f"{value * 100:.0f}%", ha="center", va="bottom", fontsize=9)
    for idx, value in enumerate(error_rates):
        ax1.text(idx + width / 2, value + 0.03, f"{value * 100:.0f}%", ha="center", va="bottom", fontsize=9)

    bars = ax2.bar(domains, success_rates, color="#7353ba", edgecolor="#4b2f82")
    ax2.set_ylabel("Success Rate")
    ax2.set_ylim(0, 1.12)
    ax2.set_title("Evaluation Success Rate", pad=10)
    style_axis(ax2)

    for idx, bar in enumerate(bars):
        value = success_rates[idx]
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.03,
            f"{value * 100:.0f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.suptitle("Cross-Domain Faithfulness Summary", y=0.99, fontsize=18, fontweight="semibold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "faithfulness_comparison.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot cross-domain robustness results")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="reproduce/results/cross_domain_robustness",
        help="Directory containing cross_domain_summary.json or cross_domain_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write plots. Defaults to <input-dir>/plots",
    )
    args = parser.parse_args()

    apply_professional_style()

    input_dir = args.input_dir
    output_dir = args.output_dir or os.path.join(input_dir, "plots")
    summary = load_summary(input_dir)

    plot_error_type_distribution(summary, output_dir)
    plot_faithfulness_comparison(summary, output_dir)

    print(f"Saved plots to: {output_dir}")
    print("- error_type_distribution.png")
    print("- faithfulness_comparison.png")


if __name__ == "__main__":
    main()

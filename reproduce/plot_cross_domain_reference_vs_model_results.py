import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_COLOR = "#355C7D"
ENH_COLOR = "#F67280"

HEADLINE_METRICS = [
    ("Faithfulness", "Faithfulness(%)"),
    ("Error Rate", "ErrorRate(%)"),
    ("Success Rate", "SuccessRate(%)"),
]


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
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
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


def annotate_bars(ax, bars, fmt="{:.1f}%"):
    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1.2,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9,
            color="#1F2937",
        )


def load_summary(csv_path):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Summary CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def plot_metric_panel(ax, df, metric_col, title):
    domains = ["SAMSum", "GovReport"]
    sources = ["reference", "bart"]

    x = np.arange(len(domains))
    width = 0.36

    ref_vals = []
    bart_vals = []

    for domain in domains:
        sub = df[df["Domain"] == domain]
        ref_row = sub[sub["SummarySource"] == "reference"]
        bart_row = sub[sub["SummarySource"] == "bart"]
        ref_vals.append(float(ref_row.iloc[0][metric_col]) if not ref_row.empty else np.nan)
        bart_vals.append(float(bart_row.iloc[0][metric_col]) if not bart_row.empty else np.nan)

    ref_bars = ax.bar(
        x - width / 2,
        ref_vals,
        width,
        label="reference",
        color=BASE_COLOR,
        edgecolor="white",
        linewidth=0.8,
    )
    bart_bars = ax.bar(
        x + width / 2,
        bart_vals,
        width,
        label="bart",
        color=ENH_COLOR,
        edgecolor="white",
        linewidth=0.8,
    )

    ax.set_title(title, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percent")
    style_axis(ax)
    annotate_bars(ax, ref_bars)
    annotate_bars(ax, bart_bars)


def plot_reference_vs_bart(csv_path, output_dir):
    df = load_summary(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    # Convert percentage columns to numeric values for plotting.
    plot_df = df.copy()
    for col in ["Faithfulness(%)", "ErrorRate(%)", "SuccessRate(%)"]:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

    fig, axes = plt.subplots(1, 3, figsize=(16, 6.0), sharey=True)

    plot_metric_panel(axes[0], plot_df, "Faithfulness(%)", "Faithfulness")
    plot_metric_panel(axes[1], plot_df, "ErrorRate(%)", "Error Rate")
    plot_metric_panel(axes[2], plot_df, "SuccessRate(%)", "Evaluation Success Rate")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Cross-Domain Reference vs BART Comparison", y=0.985, fontsize=19)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=2, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.86])

    out_path = os.path.join(output_dir, "reference_vs_bart_grouped_metrics.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_path)


def plot_headline_metrics(csv_path, output_dir):
    df = load_summary(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    plot_df = df.copy()
    for col in ["Faithfulness(%)", "ErrorRate(%)", "SuccessRate(%)"]:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

    sources = ["reference", "bart"]
    x = np.arange(len(HEADLINE_METRICS))
    width = 0.34

    ref_vals = []
    bart_vals = []
    for _, metric_col in HEADLINE_METRICS:
        metric_series = plot_df[["SummarySource", metric_col]].dropna()
        ref_vals.append(float(metric_series[metric_series["SummarySource"] == "reference"][metric_col].mean()))
        bart_vals.append(float(metric_series[metric_series["SummarySource"] == "bart"][metric_col].mean()))

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    ref_bars = ax.bar(
        x - width / 2,
        ref_vals,
        width,
        label="reference",
        color=BASE_COLOR,
        edgecolor="white",
        linewidth=0.8,
    )
    bart_bars = ax.bar(
        x + width / 2,
        bart_vals,
        width,
        label="bart",
        color=ENH_COLOR,
        edgecolor="white",
        linewidth=0.8,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in HEADLINE_METRICS])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percent")
    style_axis(ax)
    annotate_bars(ax, ref_bars)
    annotate_bars(ax, bart_bars)

    handles, labels = ax.get_legend_handles_labels()
    fig.suptitle("Cross-Domain Headline Metrics", y=0.985, fontsize=19)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    out_path = os.path.join(output_dir, "reference_vs_bart_headline_metrics.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_path)


def main():
    parser = argparse.ArgumentParser(description="Create professional plots from cross-domain reference-vs-model results")
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="reproduce/results/cross_domain_reference_vs_model/bart/reference_vs_model_summary.csv",
        help="Path to reference_vs_model_summary.csv produced by the analysis script",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reproduce/results/cross_domain_reference_vs_model/bart/plots",
        help="Directory where plots will be written",
    )
    args = parser.parse_args()

    apply_professional_style()
    plot_headline_metrics(args.summary_csv, args.output_dir)
    plot_reference_vs_bart(args.summary_csv, args.output_dir)


if __name__ == "__main__":
    main()

"""
Generate per-report plots from a single FineSurE evaluation-report.json folder.

Usage:
    python reproduce/plot_evaluation_reports.py \
            --report-dir reproduce/results/reports/base

Batch usage:
    python reproduce/plot_evaluation_reports.py \
            --reports-root reproduce/results/reports
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np


FAITH_METRIC_KEYS = [
    ("bAcc", ("faithfulness", "sentence", "balanced_accuracy")),
    ("Macro-F1", ("faithfulness", "sentence", "macro_f1")),
    ("Pearson", ("faithfulness", "summary", "pearson", "statistic")),
    ("Spearman", ("faithfulness", "summary", "spearman", "statistic")),
    ("SysRank", ("faithfulness", "system", "rank_correlation", "statistic")),
    ("Success", ("faithfulness", "success_ratio")),
]

DEFAULT_REPORT_ORDER = [
    "base",
    "gpt4",
    "enhanced",
    "enhanced_sc_qwen_100",
    "keyfact_two_stage_sc_qwen_100",
    "keyfact_machine_sc_qwen_100",
]


def _apply_professional_style():
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


def _style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D0D7DE")
    ax.spines["bottom"].set_color("#D0D7DE")
    ax.grid(axis="y", alpha=0.9)
    ax.grid(axis="x", visible=False)


def _get_nested(dct: Dict, path: Tuple[str, ...], default=0.0):
    cur = dct
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def _load_report(report_json_path: str) -> Dict:
    with open(report_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _report_dirs(reports_root: str) -> List[str]:
    if not os.path.isdir(reports_root):
        return []

    subdirs = []
    for name in sorted(os.listdir(reports_root)):
        d = os.path.join(reports_root, name)
        if not os.path.isdir(d):
            continue
        p = os.path.join(d, "evaluation-report.json")
        if os.path.isfile(p):
            subdirs.append(d)
    return subdirs


def _resolve_report_dirs(report_dir: Optional[str] = None, reports_root: Optional[str] = None) -> List[str]:
    if report_dir:
        return [report_dir]

    if not reports_root:
        return []

    root_path = Path(reports_root)
    ordered = []
    seen = set()

    for name in DEFAULT_REPORT_ORDER:
        candidate = root_path / name
        report_json = candidate / "evaluation-report.json"
        if candidate.is_dir() and report_json.is_file():
            ordered.append(str(candidate))
            seen.add(str(candidate))

    for candidate in _report_dirs(reports_root):
        if candidate not in seen:
            ordered.append(candidate)

    return ordered


def _plot_faithfulness_headline(report: Dict, out_path: str, title_suffix: str):
    labels = [k for k, _ in FAITH_METRIC_KEYS]
    vals = [_safe_float(_get_nested(report, path, 0.0)) for _, path in FAITH_METRIC_KEYS]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color="#355C7D", edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(f"Faithfulness Headline Metrics ({title_suffix})", pad=12)
    _style_axis(ax)

    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(val + 0.02, 0.98),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_per_category_f1_support(report: Dict, out_path: str, title_suffix: str):
    per_cat = _get_nested(report, ("faithfulness", "per_category"), [])
    if not per_cat:
        return

    cats = [row.get("category", "unknown") for row in per_cat]
    f1 = [_safe_float(row.get("f1", 0.0)) for row in per_cat]
    support = [_safe_float(row.get("support", 0.0)) for row in per_cat]

    x = np.arange(len(cats))

    fig, ax1 = plt.subplots(figsize=(12, 5.2))
    bars = ax1.bar(x, f1, color="#0F766E", alpha=0.92, edgecolor="white", linewidth=0.8)
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("F1")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cats, rotation=35, ha="right")
    ax1.set_title(f"Per-Category F1 and Support ({title_suffix})", pad=12)
    _style_axis(ax1)

    ax2 = ax1.twinx()
    ax2.plot(x, support, color="#b45309", marker="o", linewidth=2)
    ax2.set_ylabel("Support")

    for bar, val in zip(bars, f1):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            min(val + 0.02, 0.98),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _build_confusion_matrix(report: Dict):
    cm_rows = _get_nested(report, ("faithfulness", "confusion_matrix"), [])
    per_cat = _get_nested(report, ("faithfulness", "per_category"), [])

    category_order = [row.get("category", "unknown") for row in per_cat]
    if not category_order:
        category_order = [row.get("gt", "unknown") for row in cm_rows]

    idx = {cat: i for i, cat in enumerate(category_order)}
    mat = np.zeros((len(category_order), len(category_order)), dtype=float)

    for row in cm_rows:
        gt = row.get("gt")
        pred_counts = row.get("pred_counts", {})
        if gt not in idx:
            continue
        r = idx[gt]
        for pred_cat, count in pred_counts.items():
            if pred_cat not in idx:
                continue
            c = idx[pred_cat]
            mat[r, c] = _safe_float(count, 0.0)

    return category_order, mat


def _plot_confusion_heatmap(report: Dict, out_path: str, title_suffix: str):
    cats, mat = _build_confusion_matrix(report)
    if mat.size == 0:
        return

    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    vmax = np.max(mat) if np.max(mat) > 0 else 1.0
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=vmax)
    fig.colorbar(im, fraction=0.046, pad=0.04, label="Count")

    ax.set_xticks(np.arange(len(cats)))
    ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(cats)))
    ax.set_yticklabels(cats, fontsize=9)
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("Ground truth category")
    ax.set_title(f"Category Confusion Heatmap ({title_suffix})", pad=8)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_comp_conc_corr(report: Dict, out_path: str, title_suffix: str):
    labels = ["Pearson", "Spearman", "SysRank", "Success"]

    comp_vals = [
        _safe_float(_get_nested(report, ("completeness", "summary", "pearson", "statistic"), 0.0)),
        _safe_float(_get_nested(report, ("completeness", "summary", "spearman", "statistic"), 0.0)),
        _safe_float(_get_nested(report, ("completeness", "system", "rank_correlation", "statistic"), 0.0)),
        _safe_float(_get_nested(report, ("completeness", "success_ratio"), np.nan), np.nan),
    ]

        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        _safe_float(_get_nested(report, ("conciseness", "summary", "pearson", "statistic"), 0.0)),
        _safe_float(_get_nested(report, ("conciseness", "summary", "spearman", "statistic"), 0.0)),
        _safe_float(_get_nested(report, ("conciseness", "system", "rank_correlation", "statistic"), 0.0)),
        _safe_float(_get_nested(report, ("conciseness", "success_ratio"), np.nan), np.nan),
    ]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.2))
    b1 = ax.bar(x - w / 2, comp_vals, width=w, label="Completeness", color="#6F42C1", edgecolor="white", linewidth=0.8)
    b2 = ax.bar(x + w / 2, conc_vals, width=w, label="Conciseness", color="#2A6FDB", edgecolor="white", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(f"Completeness and Conciseness Metrics ({title_suffix})", pad=12)
    _style_axis(ax)
    ax.legend(frameon=False, loc="upper left")

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, min(h + 0.02, 0.98), f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_cross_run_faithfulness(all_reports: List[Tuple[str, Dict]], out_path: str):
    if not all_reports:
        return

    run_names = [name for name, _ in all_reports]
    metric_labels = [k for k, _ in FAITH_METRIC_KEYS]
    values = np.array(
        [
            [_safe_float(_get_nested(report, path, 0.0)) for _, path in FAITH_METRIC_KEYS]
            for _, report in all_reports
        ],
        dtype=float,
    )

    x = np.arange(len(metric_labels))
    width = 0.8 / max(len(run_names), 1)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, run_name in enumerate(run_names):
        offset = (i - (len(run_names) - 1) / 2) * width
        ax.bar(x + offset, values[i], width=width, label=run_name)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Cross-Run Faithfulness Comparison", pad=12)
    _style_axis(ax)
    ax.legend(ncol=2, fontsize=9, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_cross_run_success(all_reports: List[Tuple[str, Dict]], out_path: str):
    if not all_reports:
        return

    run_names = [name for name, _ in all_reports]
    faith_success = [_safe_float(_get_nested(r, ("faithfulness", "success_ratio"), np.nan), np.nan) for _, r in all_reports]
    conc_success = [_safe_float(_get_nested(r, ("conciseness", "success_ratio"), np.nan), np.nan) for _, r in all_reports]

    x = np.arange(len(run_names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(x - w / 2, faith_success, width=w, label="Faithfulness", color="#0891B2", edgecolor="white", linewidth=0.8)
    ax.bar(x + w / 2, conc_success, width=w, label="Conciseness", color="#F59E0B", edgecolor="white", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(run_names, rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Success ratio")
    ax.set_title("Cross-Run Success Ratios", pad=12)
    _style_axis(ax)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_cross_run_summary_table(all_reports: List[Tuple[str, Dict]], out_path: str):
    if not all_reports:
        return

    rows = []
    for run_name, report in all_reports:
        faith = report.get("faithfulness", {})
        comp = report.get("completeness", {})
        conc = report.get("conciseness", {})
        rows.append([
            run_name,
            _safe_float(_get_nested(faith, ("sentence", "balanced_accuracy"), 0.0)),
            _safe_float(_get_nested(faith, ("sentence", "macro_f1"), 0.0)),
            _safe_float(_get_nested(faith, ("summary", "pearson", "statistic"), 0.0)),
            _safe_float(_get_nested(faith, ("system", "rank_correlation", "statistic"), 0.0)),
            _safe_float(_get_nested(comp, ("summary", "pearson", "statistic"), 0.0)),
            _safe_float(_get_nested(conc, ("summary", "pearson", "statistic"), 0.0)),
        ])

    fig, ax = plt.subplots(figsize=(13, 0.55 + 0.35 * len(rows)))
    ax.axis("off")

    col_labels = [
        "Run",
        "Faith bAcc",
        "Faith Macro-F1",
        "Faith Pearson",
        "Faith SysRank",
        "Comp Pearson",
        "Conc Pearson",
    ]

    cell_text = [
        [
            row[0],
            f"{row[1]:.3f}",
            f"{row[2]:.3f}",
            f"{row[3]:.3f}",
            f"{row[4]:.3f}",
            f"{row[5]:.3f}",
            f"{row[6]:.3f}",
        ]
        for row in rows
    ]

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)
    ax.set_title("Cross-Run Evaluation Summary", pad=18)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_batch_plots(report_dirs: List[str], output_root: str):
    if not report_dirs:
        print("No report directories found.")
        return False

    os.makedirs(output_root, exist_ok=True)

    all_reports = []
    for report_dir in report_dirs:
        report_json = os.path.join(report_dir, "evaluation-report.json")
        if not os.path.isfile(report_json):
            print(f"[skip] missing report: {report_json}")
            continue

        report = _load_report(report_json)
        report_name = os.path.basename(report_dir)
        all_reports.append((report_name, report))

        run_out_dir = os.path.join(output_root, report_name)
        os.makedirs(run_out_dir, exist_ok=True)

        _plot_faithfulness_headline(report, os.path.join(run_out_dir, "plot_faithfulness_headline.png"), report_name)
        _plot_per_category_f1_support(report, os.path.join(run_out_dir, "plot_per_category_f1_support.png"), report_name)
        _plot_confusion_heatmap(report, os.path.join(run_out_dir, "plot_category_confusion_heatmap.png"), report_name)
        _plot_comp_conc_corr(report, os.path.join(run_out_dir, "plot_completeness_conciseness.png"), report_name)

    if not all_reports:
        print("No valid evaluation reports found.")
        return False

    _plot_cross_run_faithfulness(all_reports, os.path.join(output_root, "plot_cross_run_faithfulness.png"))
    _plot_cross_run_success(all_reports, os.path.join(output_root, "plot_cross_run_success.png"))
    _plot_cross_run_summary_table(all_reports, os.path.join(output_root, "plot_cross_run_summary_table.png"))

    print(f"[ok] generated batch plots in: {output_root}")
    return True


def generate_plots_for_report(report_dir: str):
    report_json = os.path.join(report_dir, "evaluation-report.json")
    report_name = os.path.basename(report_dir)

    if not os.path.isfile(report_json):
        print(f"[skip] missing report: {report_json}")
        return False

    report = _load_report(report_json)

    _plot_faithfulness_headline(
        report,
        os.path.join(report_dir, "plot_faithfulness_headline.png"),
        report_name,
    )
    _plot_per_category_f1_support(
        report,
        os.path.join(report_dir, "plot_per_category_f1_support.png"),
        report_name,
    )
    _plot_confusion_heatmap(
        report,
        os.path.join(report_dir, "plot_category_confusion_heatmap.png"),
        report_name,
    )
    _plot_comp_conc_corr(
        report,
        os.path.join(report_dir, "plot_completeness_conciseness.png"),
        report_name,
    )

    print(f"[ok] generated plots in: {report_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate plots from FineSurE evaluation reports")
    parser.add_argument(
        "--report-dir",
        type=str,
        required=True,
        help="Directory containing one evaluation-report.json",
    )
    parser.add_argument(
        "--reports-root",
        type=str,
        default=None,
        help="Directory containing multiple report folders to plot together",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Directory for generated plots when using --reports-root",
    )
    args = parser.parse_args()
    _apply_professional_style()

    report_dirs = _resolve_report_dirs(args.report_dir, args.reports_root)

    if len(report_dirs) == 0:
        if args.report_dir:
            print(f"Report directory not found or invalid: {args.report_dir}")
        else:
            print(f"No report directories found under: {args.reports_root}")
        return

    if args.report_dir and not args.reports_root:
        report_dir = args.report_dir
        report_json = os.path.join(report_dir, "evaluation-report.json")
        if not os.path.isdir(report_dir):
            print(f"Report directory not found: {report_dir}")
            return
        if not os.path.isfile(report_json):
            print(f"evaluation-report.json not found in: {report_dir}")
            return

        ok = generate_plots_for_report(report_dir)
        if not ok:
            print("No plots were generated.")
        return

    output_root = args.output_root or str(Path(args.reports_root) / "plots")
    ok = generate_batch_plots(report_dirs, output_root)
    if not ok:
        print("No plots were generated.")


if __name__ == "__main__":
    main()

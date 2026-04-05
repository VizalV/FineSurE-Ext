"""
Generate per-report plots from a single FineSurE evaluation-report.json folder.

Usage:
    python reproduce/plot_evaluation_reports.py \
            --report-dir reproduce/results/reports/base
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

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


def _plot_faithfulness_headline(report: Dict, out_path: str, title_suffix: str):
    labels = [k for k, _ in FAITH_METRIC_KEYS]
    vals = [_safe_float(_get_nested(report, path, 0.0)) for _, path in FAITH_METRIC_KEYS]

    plt.figure(figsize=(10, 4.5))
    x = np.arange(len(labels))
    bars = plt.bar(x, vals, color="#2f6ea6")
    plt.xticks(x, labels)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title(f"Faithfulness Headline Metrics ({title_suffix})")

    for bar, val in zip(bars, vals):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            min(val + 0.02, 0.98),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_per_category_f1_support(report: Dict, out_path: str, title_suffix: str):
    per_cat = _get_nested(report, ("faithfulness", "per_category"), [])
    if not per_cat:
        return

    cats = [row.get("category", "unknown") for row in per_cat]
    f1 = [_safe_float(row.get("f1", 0.0)) for row in per_cat]
    support = [_safe_float(row.get("support", 0.0)) for row in per_cat]

    x = np.arange(len(cats))

    fig, ax1 = plt.subplots(figsize=(12, 4.8))
    bars = ax1.bar(x, f1, color="#0f766e", alpha=0.9)
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("F1")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cats, rotation=35, ha="right")
    ax1.set_title(f"Per-Category F1 and Support ({title_suffix})")

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
    fig.savefig(out_path, dpi=200)
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

    plt.figure(figsize=(7.8, 6.6))
    vmax = np.max(mat) if np.max(mat) > 0 else 1.0
    plt.imshow(mat, cmap="YlOrRd", vmin=0, vmax=vmax)
    plt.colorbar(fraction=0.046, pad=0.04, label="Count")

    plt.xticks(np.arange(len(cats)), cats, rotation=45, ha="right", fontsize=8)
    plt.yticks(np.arange(len(cats)), cats, fontsize=8)
    plt.xlabel("Predicted category")
    plt.ylabel("Ground truth category")
    plt.title(f"Category Confusion Heatmap ({title_suffix})")

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_comp_conc_corr(report: Dict, out_path: str, title_suffix: str):
    labels = ["Pearson", "Spearman", "SysRank", "Success"]

    comp_vals = [
        _safe_float(_get_nested(report, ("completeness", "summary", "pearson", "statistic"), 0.0)),
        _safe_float(_get_nested(report, ("completeness", "summary", "spearman", "statistic"), 0.0)),
        _safe_float(_get_nested(report, ("completeness", "system", "rank_correlation", "statistic"), 0.0)),
        _safe_float(_get_nested(report, ("completeness", "success_ratio"), np.nan), np.nan),
    ]

    conc_vals = [
        _safe_float(_get_nested(report, ("conciseness", "summary", "pearson", "statistic"), 0.0)),
        _safe_float(_get_nested(report, ("conciseness", "summary", "spearman", "statistic"), 0.0)),
        _safe_float(_get_nested(report, ("conciseness", "system", "rank_correlation", "statistic"), 0.0)),
        _safe_float(_get_nested(report, ("conciseness", "success_ratio"), np.nan), np.nan),
    ]

    x = np.arange(len(labels))
    w = 0.35

    plt.figure(figsize=(9.5, 4.8))
    plt.bar(x - w / 2, comp_vals, width=w, label="Completeness", color="#7c3aed")
    plt.bar(x + w / 2, conc_vals, width=w, label="Conciseness", color="#2563eb")

    plt.xticks(x, labels)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title(f"Completeness and Conciseness Metrics ({title_suffix})")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


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

    plt.figure(figsize=(11.5, 5.2))
    for i, run_name in enumerate(run_names):
        offset = (i - (len(run_names) - 1) / 2) * width
        plt.bar(x + offset, values[i], width=width, label=run_name)

    plt.xticks(x, metric_labels)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("Cross-Run Faithfulness Comparison")
    plt.legend(ncol=2, fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_cross_run_success(all_reports: List[Tuple[str, Dict]], out_path: str):
    if not all_reports:
        return

    run_names = [name for name, _ in all_reports]
    faith_success = [_safe_float(_get_nested(r, ("faithfulness", "success_ratio"), np.nan), np.nan) for _, r in all_reports]
    conc_success = [_safe_float(_get_nested(r, ("conciseness", "success_ratio"), np.nan), np.nan) for _, r in all_reports]

    x = np.arange(len(run_names))
    w = 0.35

    plt.figure(figsize=(10.5, 4.8))
    plt.bar(x - w / 2, faith_success, width=w, label="Faithfulness", color="#0891b2")
    plt.bar(x + w / 2, conc_success, width=w, label="Conciseness", color="#f59e0b")

    plt.xticks(x, run_names, rotation=20, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Success ratio")
    plt.title("Cross-Run Success Ratios")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


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
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()

import argparse
import csv
import json
import os
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np


CATEGORY_CODE_TO_NAME = {
    "NoE": "no error",
    "OutE": "out-of-context error",
    "EntE": "entity error",
    "RelE": "predicate error",
    "CircE": "circumstantial error",
    "GramE": "grammatical error",
    "CorefE": "coreference error",
    "LinkE": "linking error",
    "OtherE": "other error",
}

CATEGORY_ORDER = [
    "no error",
    "out-of-context error",
    "entity error",
    "predicate error",
    "circumstantial error",
    "grammatical error",
    "coreference error",
    "linking error",
    "other error",
]

ALIASES = {
    "out of context error": "out-of-context error",
    "out-of-context": "out-of-context error",
    "no factual error": "no error",
    "factual": "no error",
    "entity mismatch": "entity error",
    "predicate mismatch": "predicate error",
    "coref error": "coreference error",
}

HUMAN_COLOR = "#2F5AA8"
SYSTEM_COLOR = "#F59E0B"
DELTA_COLOR = "#B91C1C"


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
            "axes.titlesize": 19,
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


def _annotate_bar_values(ax, bars, fmt="{:.2f}", dy=0.01, color="#374151"):
    for bar in bars:
        h = bar.get_height()
        if np.isnan(h):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + dy,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=9,
            color=color,
        )


def normalize_category(value):
    if value is None:
        return "other error"

    text = str(value).strip()
    if text in CATEGORY_CODE_TO_NAME:
        return CATEGORY_CODE_TO_NAME[text]

    text = text.lower().replace("_", " ")
    text = " ".join(text.split())
    text = ALIASES.get(text, text)

    if text in CATEGORY_ORDER:
        return text

    return "other error"


def _safe_float(value, default=None):
    try:
        return float(value)
    except Exception:
        return default


def labels_from_type_cell(cell):
    if cell is None:
        return []
    if isinstance(cell, list):
        items = cell
    else:
        items = [cell]

    out = []
    for item in items:
        if item is None:
            continue
        txt = str(item)
        if txt == "None":
            continue
        out.append(normalize_category(item))
    return out


def sentence_category_set(label_value, type_cell):
    label_num = _safe_float(label_value, default=None)
    labels = set(labels_from_type_cell(type_cell))

    if label_num == 0.0:
        return {"no error"}

    if label_num == 1.0 and len(labels) == 0:
        return {"other error"}

    if label_num == 1.0 and "no error" in labels:
        labels.discard("no error")

    if len(labels) == 0:
        return {"no error"}

    return labels


def parse_annotator_sets(raw_annotations):
    annotator_sentence_sets = {}

    for annotator_id, ann in raw_annotations.items():
        labels = ann.get("factuality_labels", [])
        types = ann.get("factuality_types", [])
        max_len = max(len(labels), len(types))
        sent_sets = {}

        for idx in range(max_len):
            label_val = labels[idx] if idx < len(labels) else None
            type_cell = types[idx] if idx < len(types) else None

            if label_val == "None":
                continue

            sent_sets[idx] = sentence_category_set(label_val, type_cell)

        annotator_sentence_sets[annotator_id] = sent_sets

    return annotator_sentence_sets


def parse_prediction_sets(record):
    pred_labels = record.get("pred_faithfulness_labels", [])
    pred_types = record.get("pred_faithfulness_error_type", [])

    max_len = max(len(pred_labels), len(pred_types))
    pred_sets = {}

    for idx in range(max_len):
        label_val = pred_labels[idx] if idx < len(pred_labels) else None
        type_cell = pred_types[idx] if idx < len(pred_types) else None

        if label_val == "None":
            continue

        pred_sets[idx] = sentence_category_set(label_val, type_cell)

    return pred_sets


def compute_agreement_stats(records):
    pair_matches = {c: 0 for c in CATEGORY_ORDER}
    pair_total = {c: 0 for c in CATEGORY_ORDER}
    sys_matches = {c: 0 for c in CATEGORY_ORDER}
    sys_total = {c: 0 for c in CATEGORY_ORDER}
    support_count = {c: 0 for c in CATEGORY_ORDER}

    human_exact_matches = 0
    human_exact_total = 0
    sys_exact_matches = 0
    sys_exact_total = 0

    for rec in records:
        raw_annotations = rec.get("raw_annotations")
        if not isinstance(raw_annotations, dict) or len(raw_annotations) < 2:
            continue

        ann_sets = parse_annotator_sets(raw_annotations)
        pred_sets = parse_prediction_sets(rec)

        sentence_ids = set()
        for sent_map in ann_sets.values():
            sentence_ids.update(sent_map.keys())
        sentence_ids.update(pred_sets.keys())

        for sid in sentence_ids:
            available = []
            for a_id in ann_sets:
                if sid in ann_sets[a_id]:
                    available.append(ann_sets[a_id][sid])

            if len(available) >= 2:
                for s1, s2 in combinations(available, 2):
                    human_exact_total += 1
                    if s1 == s2:
                        human_exact_matches += 1

                    for c in CATEGORY_ORDER:
                        v1 = c in s1
                        v2 = c in s2
                        pair_total[c] += 1
                        if v1 == v2:
                            pair_matches[c] += 1

            for s in available:
                for c in s:
                    if c in support_count:
                        support_count[c] += 1

            if sid in pred_sets and len(available) > 0:
                ps = pred_sets[sid]
                for s in available:
                    sys_exact_total += 1
                    if ps == s:
                        sys_exact_matches += 1

                    for c in CATEGORY_ORDER:
                        pv = c in ps
                        hv = c in s
                        sys_total[c] += 1
                        if pv == hv:
                            sys_matches[c] += 1

    rows = []
    for c in CATEGORY_ORDER:
        human_agreement = (pair_matches[c] / pair_total[c]) if pair_total[c] > 0 else float("nan")
        system_agreement = (sys_matches[c] / sys_total[c]) if sys_total[c] > 0 else float("nan")

        human_disagreement = (1.0 - human_agreement) if pair_total[c] > 0 else float("nan")
        system_disagreement = (1.0 - system_agreement) if sys_total[c] > 0 else float("nan")

        delta = (
            system_disagreement - human_disagreement
            if pair_total[c] > 0 and sys_total[c] > 0
            else float("nan")
        )

        rows.append(
            {
                "category": c,
                "human_agreement": human_agreement,
                "system_agreement": system_agreement,
                "human_disagreement": human_disagreement,
                "system_disagreement": system_disagreement,
                "disagreement_delta_system_minus_human": delta,
                "human_pairwise_examples": pair_total[c],
                "system_vs_human_examples": sys_total[c],
                "support_count": support_count[c],
            }
        )

    summary = {
        "overall_human_exact_agreement": (human_exact_matches / human_exact_total)
        if human_exact_total > 0
        else float("nan"),
        "overall_system_vs_human_exact_agreement": (sys_exact_matches / sys_exact_total)
        if sys_exact_total > 0
        else float("nan"),
        "overall_human_exact_total": human_exact_total,
        "overall_system_vs_human_exact_total": sys_exact_total,
    }

    return rows, summary


def save_csv(rows, out_csv):
    fields = [
        "category",
        "human_agreement",
        "system_agreement",
        "human_disagreement",
        "system_disagreement",
        "disagreement_delta_system_minus_human",
        "human_pairwise_examples",
        "system_vs_human_examples",
        "support_count",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _valid(v):
    return isinstance(v, (int, float)) and not np.isnan(v)


def save_plot(rows, summary, out_png, title):
    labels = [r["category"] for r in rows]
    human = [r["human_disagreement"] if _valid(r["human_disagreement"]) else 0.0 for r in rows]
    system = [r["system_disagreement"] if _valid(r["system_disagreement"]) else 0.0 for r in rows]
    delta = [
        r["disagreement_delta_system_minus_human"]
        if _valid(r["disagreement_delta_system_minus_human"])
        else 0.0
        for r in rows
    ]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(14.5, 7.2))
    human_bars = ax.bar(
        x - width / 2,
        human,
        width,
        label="Human vs Human disagreement",
        color=HUMAN_COLOR,
        edgecolor="white",
        linewidth=0.8,
    )
    system_bars = ax.bar(
        x + width / 2,
        system,
        width,
        label="FineSurE vs Human disagreement",
        color=SYSTEM_COLOR,
        edgecolor="white",
        linewidth=0.8,
    )

    _annotate_bar_values(ax, human_bars, fmt="{:.2f}", dy=0.008)
    _annotate_bar_values(ax, system_bars, fmt="{:.2f}", dy=0.008)

    for i, d in enumerate(delta):
        if _valid(d) and d > 0.0:
            ax.text(
                x[i],
                max(human[i], system[i]) + 0.038,
                f"+{d:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color=DELTA_COLOR,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Disagreement rate (lower is better)")
    ax.set_title(title, pad=14)
    _style_axis(ax)
    ax.legend(loc="upper right", frameon=False)

    overall_hh = summary.get("overall_human_exact_agreement", float("nan"))
    overall_sh = summary.get("overall_system_vs_human_exact_agreement", float("nan"))
    subtitle = (
        f"Overall exact agreement: human-human={overall_hh:.3f} | FineSurE-human={overall_sh:.3f}"
        if _valid(overall_hh) and _valid(overall_sh)
        else "Overall exact agreement: insufficient data"
    )
    fig.text(
        0.5,
        0.015,
        subtitle,
        ha="center",
        fontsize=10.5,
        color="#1F2937",
        bbox={"facecolor": "#F9FAFB", "edgecolor": "#E5E7EB", "boxstyle": "round,pad=0.25"},
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_jsonl(input_path, max_records=None):
    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
            if max_records is not None and len(rows) >= max_records:
                break
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Plot category-wise Human-Human vs FineSurE-Human agreement on faithfulness labels"
    )
    parser.add_argument(
        "--input-jsonl",
        required=True,
        help="Path to raw-data.json (jsonl) containing raw_annotations and prediction fields",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where outputs (png/csv/json) will be written",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=-1,
        help="Use only first N records for quick subset experiments; set <=0 for all",
    )
    parser.add_argument(
        "--title",
        default="Agreement Alignment by Category",
        help="Plot title",
    )
    args = parser.parse_args()
    _apply_professional_style()

    max_records = args.max_records if args.max_records and args.max_records > 0 else None
    data = load_jsonl(args.input_jsonl, max_records=max_records)

    if len(data) == 0:
        raise SystemExit("No parseable rows found in input jsonl")

    rows, summary = compute_agreement_stats(data)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "agreement_alignment_by_category.csv")
    png_path = os.path.join(args.output_dir, "agreement_alignment_by_category.png")
    json_path = os.path.join(args.output_dir, "agreement_alignment_summary.json")

    save_csv(rows, csv_path)
    save_plot(rows, summary, png_path, args.title)

    payload = {
        "input_jsonl": args.input_jsonl,
        "records_used": len(data),
        "summary": summary,
        "categories": rows,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("[Agreement Alignment Outputs]")
    print("- Plot:", png_path)
    print("- CSV:", csv_path)
    print("- Summary:", json_path)


if __name__ == "__main__":
    main()

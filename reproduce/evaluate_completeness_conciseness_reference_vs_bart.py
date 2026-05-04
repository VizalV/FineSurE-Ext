import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "finesure"))

from finesure.utils_opensource import (
    compute_completeness_percentage_score,
    compute_conciseness_percentage_score,
)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_alignment(
    input_jsonl: str,
    keyfact_jsonl: str,
    output_dir: str,
    model_key: str,
    reasoning_mode: str,
    n_samples: int,
    temperature: float,
    max_tokens: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        sys.executable,
        "finesure/keyfact-alignment-opensource-enhanced.py",
        input_jsonl,
        keyfact_jsonl,
        output_dir,
        model_key,
        "--reasoning-mode",
        reasoning_mode,
        "--n-samples",
        str(n_samples),
        "--temperature",
        str(temperature),
        "--max-tokens",
        str(max_tokens),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def summarize_alignment(raw_data_json: str) -> Dict[str, Any]:
    rows = read_jsonl(raw_data_json)

    total = len(rows)
    successful = 0
    completeness_scores = []
    conciseness_scores = []

    for row in rows:
        labels = row.get("pred_alignment_labels", [])
        line_nums = row.get("pred_sentence_line_numbers", [])
        sent_count = len(row.get("sentences", []))

        is_ok = len(labels) > 0 and sent_count > 0
        if not is_ok:
            continue

        successful += 1
        completeness_scores.append(compute_completeness_percentage_score(labels))
        conciseness_scores.append(compute_conciseness_percentage_score(line_nums, sent_count))

    return {
        "total": total,
        "successful": successful,
        "success_rate": (successful / total) if total else 0.0,
        "completeness": float(np.mean(completeness_scores)) if completeness_scores else 0.0,
        "conciseness": float(np.mean(conciseness_scores)) if conciseness_scores else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate FineSurE completeness + conciseness for reference vs bart summaries."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="reproduce/results/cross_domain_reference_vs_model/bart",
        help="Folder containing SAMSum___reference.jsonl, SAMSum___bart.jsonl, GovReport___reference.jsonl, GovReport___bart.jsonl",
    )
    parser.add_argument(
        "--keyfact-dir",
        type=str,
        default="reproduce/results/cross_domain_reference_vs_model/bart/generated_keyfacts",
        help="Folder containing SAMSum_keyfacts.jsonl and GovReport_keyfacts.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reproduce/results/cross_domain_reference_vs_model/bart/completeness_conciseness",
    )
    parser.add_argument("--finesure-model", type=str, default="qwen2.5-7b-direct")
    parser.add_argument("--reasoning-mode", type=str, default="two_stage", choices=["single_stage", "two_stage"])
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Do not run alignment; summarize existing raw-data.json files only.",
    )
    args = parser.parse_args()

    tasks = [
        ("SAMSum", "reference", "SAMSum___reference.jsonl", "SAMSum_keyfacts.jsonl"),
        ("SAMSum", "bart", "SAMSum___bart.jsonl", "SAMSum_keyfacts.jsonl"),
        ("GovReport", "reference", "GovReport___reference.jsonl", "GovReport_keyfacts.jsonl"),
        ("GovReport", "bart", "GovReport___bart.jsonl", "GovReport_keyfacts.jsonl"),
    ]

    rows = []
    for domain, source, input_name, keyfact_name in tasks:
        input_path = os.path.join(args.input_dir, input_name)
        keyfact_path = os.path.join(args.keyfact_dir, keyfact_name)
        out_dir = os.path.join(args.output_dir, f"{domain}_{source}")

        if not args.skip_run:
            run_alignment(
                input_jsonl=input_path,
                keyfact_jsonl=keyfact_path,
                output_dir=out_dir,
                model_key=args.finesure_model,
                reasoning_mode=args.reasoning_mode,
                n_samples=args.n_samples,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )

        raw_data_path = os.path.join(out_dir, "raw-data.json")
        stats = summarize_alignment(raw_data_path)

        rows.append(
            {
                "Domain": domain,
                "SummarySource": source,
                "Total": stats["total"],
                "Successful": stats["successful"],
                "SuccessRate": stats["success_rate"],
                "SuccessRate(%)": f"{100 * stats['success_rate']:.1f}",
                "Completeness": stats["completeness"],
                "Completeness(%)": f"{100 * stats['completeness']:.1f}",
                "Conciseness": stats["conciseness"],
                "Conciseness(%)": f"{100 * stats['conciseness']:.1f}",
            }
        )

    df = pd.DataFrame(rows)
    os.makedirs(args.output_dir, exist_ok=True)

    csv_path = os.path.join(args.output_dir, "reference_vs_bart_completeness_conciseness.csv")
    json_path = os.path.join(args.output_dir, "reference_vs_bart_completeness_conciseness.json")
    df.to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=2)

    print("Saved:", csv_path)
    print("Saved:", json_path)


if __name__ == "__main__":
    main()

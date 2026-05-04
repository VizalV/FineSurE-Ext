import argparse
import json
import os
import re
import time
from typing import Any, Dict, List

from openai import OpenAI


PROMPT_TEMPLATE = """You will be provided with a summary. Your task is to decompose
the summary into a set of \"key facts\". A \"key fact\" is a single
fact written as briefly and clearly as possible, encompassing at
most 2-3 entities.
Here are nine examples of key facts to illustrate the desired
level of granularity:
* Kevin Carr set off on his journey from Haytor.
* Kevin Carr set off on his journey from Dartmoor.
* Kevin Carr set off on his journey in July 2013.
* Kevin Carr is less than 24 hours away from completing his trip.
* Kevin Carr ran around the world unsupported.
* Kevin Carr ran with his tent.
* Kevin Carr is set to break the previous record.
* Kevin Carr is set to break the record by 24 hours.
* The previous record was held by an Australian.

Instruction:
First, read the summary carefully.
Second, decompose the summary into (at most 16) key facts.
Provide your answer in JSON format. The answer should be a dictionary
with the key \"key facts\" containing the key facts as a list:
{\"key facts\": [\"first key fact\", \"second key fact\", \"third key fact\"]}

Summary:
{summary}
"""


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}

    block = match.group(0)
    try:
        return json.loads(block)
    except Exception:
        return {}


def normalize_keyfacts(obj: Dict[str, Any], max_facts: int = 16) -> List[str]:
    if not isinstance(obj, dict):
        return []

    candidates = None
    if "key facts" in obj:
        candidates = obj["key facts"]
    elif "key_facts" in obj:
        candidates = obj["key_facts"]
    elif "facts" in obj:
        candidates = obj["facts"]

    if not isinstance(candidates, list):
        return []

    out = []
    for x in candidates:
        s = str(x).strip()
        if s:
            out.append(s)
    return out[:max_facts]


def build_messages(summary: str) -> List[Dict[str, str]]:
    # Avoid str.format on the whole prompt because JSON braces in examples
    # (e.g., {"key facts": [...]}) can be misread as format placeholders.
    prompt = PROMPT_TEMPLATE.replace("{summary}", summary)
    return [
        {
            "role": "system",
            "content": (
                "You are a careful information extraction assistant. "
                "Return strict JSON only, no markdown."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]


def annotate_rows(
    rows: List[Dict[str, Any]],
    client: OpenAI,
    model: str,
    sleep_sec: float,
    max_retries: int,
) -> List[Dict[str, Any]]:
    annotated = []

    for i, row in enumerate(rows, start=1):
        doc_id = row.get("doc_id")
        summary = row.get("reference", "")
        if not summary:
            print(f"[WARN] {doc_id}: missing reference summary; skipping")
            annotated.append(
                {
                    "doc_id": doc_id,
                    "key_facts": [],
                    "success": False,
                    "error": "missing reference summary",
                }
            )
            continue

        success = False
        err = ""
        key_facts = []

        for attempt in range(1, max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    temperature=0.0,
                    messages=build_messages(summary),
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content or ""
                parsed = safe_extract_json_object(content)
                key_facts = normalize_keyfacts(parsed, max_facts=16)
                if key_facts:
                    print(
                        f"[OK] {doc_id}: extracted {len(key_facts)} keyfacts "
                        f"(attempt {attempt}/{max_retries})"
                    )
                    success = True
                    break
                err = "empty_or_unparseable_keyfacts"
                preview = content.replace("\n", " ").strip()[:220]
                print(
                    f"[WARN] {doc_id}: attempt {attempt}/{max_retries} returned no usable keyfacts. "
                    f"Response preview: {preview}"
                )
            except Exception as exc:
                err = str(exc)
                print(
                    f"[WARN] {doc_id}: attempt {attempt}/{max_retries} failed with error: {err}"
                )

            time.sleep(min(2.0 * attempt, 10.0))

        if not success:
            print(f"[FAIL] {doc_id}: keyfact generation failed after {max_retries} attempt(s).")

        annotated.append(
            {
                "doc_id": doc_id,
                "key_facts": key_facts,
                "success": success,
                "error": "" if success else err,
            }
        )

        if i % 25 == 0:
            print(f"Processed {i}/{len(rows)}")
        time.sleep(sleep_sec)

    return annotated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate keyfacts from reference summaries using gpt-4.1-nano"
    )
    parser.add_argument(
        "--samsum-reference-jsonl",
        type=str,
        default="reproduce/results/cross_domain_reference_vs_model/bart/SAMSum___reference.jsonl",
    )
    parser.add_argument(
        "--govreport-reference-jsonl",
        type=str,
        default="reproduce/results/cross_domain_reference_vs_model/bart/GovReport___reference.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reproduce/results/cross_domain_reference_vs_model/bart/generated_keyfacts",
    )
    parser.add_argument("--model", type=str, default="gpt-4.1-nano")
    parser.add_argument("--sleep-sec", type=float, default=0.05)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    client = OpenAI()

    pairs = [
        ("SAMSum", args.samsum_reference_jsonl),
        ("GovReport", args.govreport_reference_jsonl),
    ]

    for domain, path in pairs:
        print(f"\nGenerating keyfacts for {domain} from {path}")
        rows = read_jsonl(path)
        ann = annotate_rows(
            rows=rows,
            client=client,
            model=args.model,
            sleep_sec=args.sleep_sec,
            max_retries=args.max_retries,
        )

        out_rows = []
        success_count = 0
        for src, kf in zip(rows, ann):
            out = {
                "doc_id": src.get("doc_id"),
                "key_facts": kf.get("key_facts", []),
                "source_summary": "reference",
                "domain": domain,
                "success": kf.get("success", False),
                "error": kf.get("error", ""),
            }
            if out["success"]:
                success_count += 1
            out_rows.append(out)

        out_path = os.path.join(args.output_dir, f"{domain}_keyfacts.jsonl")
        write_jsonl(out_path, out_rows)
        print(
            f"Saved: {out_path} | success {success_count}/{len(out_rows)} "
            f"({(100.0 * success_count / max(1, len(out_rows))):.1f}%)"
        )


if __name__ == "__main__":
    main()

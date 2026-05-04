"""
Cross-domain evaluator: reference summaries vs FRANK-style model-generated summaries.

For SAMSum and GovReport:
1) Evaluate FineSurE on dataset reference summaries.
2) Generate summaries using HF models mapped to FRANK-style aliases.
3) Evaluate FineSurE on those generated summaries.
4) Compare capability metrics side-by-side.
"""

import argparse
import json
import os
import random
import sys
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from transformers.utils import logging as hf_logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FINESURE_DIR = os.path.join(PROJECT_ROOT, "finesure")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, FINESURE_DIR)

from finesure.utils_opensource import get_response_opensource, parsing_llm_fact_checking_output
from finesure.utils_categorization import build_strict_factcheck_prompt, parse_factcheck_json, normalize_category


# Keep run logs focused on meaningful failures.
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", message=r"Your max_length is set to .*")
warnings.filterwarnings("ignore", message=r"You seem to be using the pipelines sequentially on GPU.*")


FRANK_STYLE_MODELS = {
	"bart": {
		"hf_model": "facebook/bart-large-cnn",
		"task": "summarization",
	},
	"s2s": {
		"hf_model": "facebook/bart-large-xsum",
		"task": "summarization",
	},
	# FRANK-style proxies where original training checkpoints are not directly available in this repo.
	"bert_sum": {
		"hf_model": "sshleifer/distilbart-cnn-12-6",
		"task": "summarization",
	},
	"pgn": {
		"hf_model": "google/pegasus-xsum",
		"task": "summarization",
	},
	"bus": {
		"hf_model": "t5-small",
		"task": "text2text-generation",
		"prefix": "summarize: ",
	},
}


def split_sentences(text, max_sentences=4):
	parts = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
	if max_sentences:
		parts = parts[:max_sentences]
	return parts


def load_huggingface_dataset(dataset_name: str, split: str = "test", max_samples: int = 50, seed: int = 42):
	try:
		from datasets import load_dataset
	except ImportError:
		raise RuntimeError("Please install datasets: pip install datasets")

	ds = load_dataset(dataset_name, split=split)
	if max_samples and len(ds) > max_samples:
		rng = random.Random(seed)
		idxs = list(range(len(ds)))
		rng.shuffle(idxs)
		ds = ds.select(idxs[:max_samples])
	return ds


def prepare_domain_records(dataset, domain_name):
	records = []
	for idx, item in enumerate(dataset):
		if domain_name == "samsum":
			transcript = item.get("dialogue", "")
			reference = item.get("summary", "")
			max_sents = 3
		else:
			transcript = item.get("report", "")
			reference = item.get("summary", "")
			max_sents = 4

		if not transcript or not reference:
			continue

		records.append(
			{
				"doc_id": f"{domain_name}-{idx}",
				"source": domain_name,
				"split": "test",
				"model": "ground-truth",
				"transcript": transcript,
				"reference": reference,
				"sentences": split_sentences(reference, max_sentences=max_sents),
				"raw_annotations": {},
			}
		)
	return records


def build_generator(alias, device):
	from transformers import pipeline

	cfg = FRANK_STYLE_MODELS[alias]
	task = cfg["task"]
	return pipeline(task, model=cfg["hf_model"], device=device)


def safe_generator_input(generator, text, max_input_tokens=256, reserve_new_tokens=64):
	"""Tokenizer-aware truncation to prevent model position overflows."""
	tokenizer = getattr(generator, "tokenizer", None)
	if tokenizer is None:
		return text

	model_max_len = getattr(tokenizer, "model_max_length", None)
	if model_max_len is None or model_max_len <= 0 or model_max_len > 100000:
		model_max_len = max_input_tokens

	allowed = min(max_input_tokens, max(128, model_max_len - reserve_new_tokens - 8))
	enc = tokenizer(text, truncation=True, max_length=allowed, return_tensors=None)
	input_ids = enc.get("input_ids", [])
	if not input_ids:
		return text[:2000]
	return tokenizer.decode(input_ids, skip_special_tokens=True)


def generate_summary(generator, alias, transcript, max_new_tokens=64, max_input_tokens=256):
	cfg = FRANK_STYLE_MODELS[alias]
	task = cfg["task"]
	safe_text = safe_generator_input(
		generator,
		transcript,
		max_input_tokens=max_input_tokens,
		reserve_new_tokens=max_new_tokens,
	)

	if task == "summarization":
		out = generator(
			safe_text,
			max_new_tokens=max_new_tokens,
			min_new_tokens=20,
			do_sample=False,
		)
		return out[0]["summary_text"].strip()

	prompt = cfg.get("prefix", "") + safe_text
	out = generator(
		prompt,
		max_new_tokens=max_new_tokens,
		min_new_tokens=20,
		do_sample=False,
	)
	return out[0]["generated_text"].strip()


def _cleanup_cuda():
	try:
		import torch

		if torch.cuda.is_available():
			torch.cuda.synchronize()
			torch.cuda.empty_cache()
	except Exception:
		pass


def build_generated_records(base_records, alias, device=0, max_new_tokens=64, max_input_tokens=256):
	print(f"\nGenerating summaries with {alias} ({FRANK_STYLE_MODELS[alias]['hf_model']})...")
	gen = build_generator(alias, device=device)

	generated = []
	for i, rec in enumerate(base_records):
		try:
			summary = generate_summary(
				gen,
				alias,
				rec["transcript"],
				max_new_tokens=max_new_tokens,
				max_input_tokens=max_input_tokens,
			)
			item = dict(rec)
			item["model"] = alias
			item["reference"] = summary
			item["sentences"] = split_sentences(summary, max_sentences=len(rec["sentences"]) or 3)
			if len(item["sentences"]) == 0:
				item["sentences"] = [summary[:200]] if summary else []
			generated.append(item)
		except Exception as exc:
			msg = str(exc)
			if "device-side assert" in msg.lower() or "cuda" in msg.lower():
				_cleanup_cuda()
				print(
					f"  [{i+1}/{len(base_records)}] CUDA failure for alias '{alias}'. "
					"Stopping this alias and continuing with others."
				)
				break
			else:
				print(f"  [{i+1}/{len(base_records)}] generation failed: {msg[:140]}")

	del gen
	_cleanup_cuda()
	return generated


def run_finesure_fact_checking(data_list, model_key, use_strict=True):
	results = []
	success_count = 0

	for idx, item in enumerate(data_list):
		print(f"  [{idx+1}/{len(data_list)}] {item['doc_id']} ({item['model']})")
		sentences = item.get("sentences", [])
		transcript = item.get("transcript", "")

		if not sentences:
			failed = dict(item)
			failed["success"] = False
			failed["error"] = "no sentences to evaluate"
			results.append(failed)
			continue

		try:
			prompt = build_strict_factcheck_prompt(transcript=transcript, sentences=sentences) if use_strict else None
			if prompt is None:
				raise RuntimeError("Only strict mode is supported in this runner")

			output = get_response_opensource(
				prompt=prompt,
				model_key=model_key,
				temperature=0.0,
				max_tokens=2048,
			)

			parsed = parse_factcheck_json(output, expected_num_sentences=len(sentences))
			if parsed.get("success", False):
				pred_labels = parsed.get("labels", [])
				pred_types = [normalize_category(c) for c in parsed.get("categories", [])]
				success = len(pred_labels) == len(sentences) and len(pred_labels) > 0
			else:
				pred_labels, pred_types = parsing_llm_fact_checking_output(output)
				success = len(pred_labels) == len(sentences) and len(pred_labels) > 0

			row = dict(item)
			row["llm_output"] = output
			row["pred_faithfulness_labels"] = pred_labels
			row["pred_faithfulness_error_type"] = pred_types
			row["success"] = success
			results.append(row)

			if success:
				success_count += 1
		except Exception as exc:
			failed = dict(item)
			failed["success"] = False
			failed["error"] = str(exc)
			results.append(failed)

	print(f"  FineSurE success rate: {success_count}/{len(data_list)} ({(100.0*success_count/max(len(data_list),1)):.1f}%)")
	return results


def analyze_results(results_dict):
	analysis = {}
	for group, rows in results_dict.items():
		total = len(rows)
		ok = [r for r in rows if r.get("success", False)]

		types = []
		labels = []
		sent_lens = []
		for r in ok:
			types.extend(r.get("pred_faithfulness_error_type", []))
			labels.extend(r.get("pred_faithfulness_labels", []))
			sent_lens.append(len(r.get("sentences", [])))

		counts = Counter(types)
		total_preds = len(types)
		no_error = counts.get("no error", 0)

		labels_arr = np.array(labels) if labels else np.array([])
		err_rate = float((labels_arr == 1).sum() / len(labels_arr)) if len(labels_arr) else 0.0

		analysis[group] = {
			"total": total,
			"successful": len(ok),
			"success_rate": (len(ok) / total) if total else 0.0,
			"faithfulness": (no_error / total_preds) if total_preds else 0.0,
			"error_rate": err_rate,
			"avg_sentences": float(np.mean(sent_lens)) if sent_lens else 0.0,
			"error_type_counts": dict(counts),
		}

	return analysis


def save_outputs(results_dict, analysis, out_dir):
	os.makedirs(out_dir, exist_ok=True)

	with open(os.path.join(out_dir, "reference_vs_model_results.json"), "w", encoding="utf-8") as f:
		json.dump({"analysis": analysis}, f, indent=2)

	csv_rows = []
	for group, stats in analysis.items():
		domain, summary_source = group.split(" | ")
		csv_rows.append(
			{
				"Domain": domain,
				"SummarySource": summary_source,
				"Total": stats["total"],
				"Successful": stats["successful"],
				"SuccessRate(%)": f"{100*stats['success_rate']:.1f}",
				"Faithfulness(%)": f"{100*stats['faithfulness']:.1f}",
				"ErrorRate(%)": f"{100*stats['error_rate']:.1f}",
				"AvgSentences": f"{stats['avg_sentences']:.2f}",
			}
		)

	pd.DataFrame(csv_rows).to_csv(os.path.join(out_dir, "reference_vs_model_summary.csv"), index=False)

	for group, rows in results_dict.items():
		safe = group.replace(" ", "_").replace("|", "_").replace("(", "").replace(")", "").replace("/", "_")
		with open(os.path.join(out_dir, f"{safe}.jsonl"), "w", encoding="utf-8") as f:
			for r in rows:
				f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _parse_group_key(group_key):
	parts = [p.strip() for p in group_key.split("|")]
	if len(parts) != 2:
		return group_key, "unknown"
	return parts[0], parts[1]


def _mean_error_rate(labels):
	if not labels:
		return 0.0
	arr = np.array(labels)
	return float((arr == 1).sum() / len(arr))


def save_example_comparisons(results_dict, out_dir, model_alias="bart", max_examples=5):
	os.makedirs(out_dir, exist_ok=True)

	all_examples = []
	reference_keys = [k for k in results_dict.keys() if k.endswith("| reference")]

	for ref_key in reference_keys:
		domain = _parse_group_key(ref_key)[0]
		model_key = f"{domain} | {model_alias}"
		if model_key not in results_dict:
			continue

		ref_rows = {r.get("doc_id"): r for r in results_dict[ref_key] if r.get("doc_id")}
		mdl_rows = {r.get("doc_id"): r for r in results_dict[model_key] if r.get("doc_id")}
		common_ids = [d for d in ref_rows.keys() if d in mdl_rows]

		for doc_id in common_ids:
			ref = ref_rows[doc_id]
			mdl = mdl_rows[doc_id]

			ref_summary = ref.get("reference", "")
			mdl_summary = mdl.get("reference", "")
			ref_types = ref.get("pred_faithfulness_error_type", []) if ref.get("success") else []
			mdl_types = mdl.get("pred_faithfulness_error_type", []) if mdl.get("success") else []
			ref_labels = ref.get("pred_faithfulness_labels", []) if ref.get("success") else []
			mdl_labels = mdl.get("pred_faithfulness_labels", []) if mdl.get("success") else []

			added_types = sorted(set(mdl_types) - set(ref_types))
			removed_types = sorted(set(ref_types) - set(mdl_types))
			summary_changed = ref_summary.strip() != mdl_summary.strip()

			diff_score = 0
			diff_score += 1 if summary_changed else 0
			diff_score += len(added_types) + len(removed_types)
			diff_score += 1 if ref.get("success") != mdl.get("success") else 0

			all_examples.append(
				{
					"domain": domain,
					"doc_id": doc_id,
					"model_alias": model_alias,
					"summary_changed": summary_changed,
					"diff_score": diff_score,
					"reference_success": bool(ref.get("success", False)),
					"bart_success": bool(mdl.get("success", False)),
					"ground_truth_summary": ref_summary,
					"bart_summary": mdl_summary,
					"reference_pred_labels": ref_labels,
					"bart_pred_labels": mdl_labels,
					"reference_pred_error_types": ref_types,
					"bart_pred_error_types": mdl_types,
					"reference_error_rate": _mean_error_rate(ref_labels),
					"bart_error_rate": _mean_error_rate(mdl_labels),
					"added_error_types_in_bart": added_types,
					"removed_error_types_in_bart": removed_types,
					"reference_llm_output": str(ref.get("llm_output", ""))[:2000],
					"bart_llm_output": str(mdl.get("llm_output", ""))[:2000],
				}
			)

	all_examples.sort(key=lambda x: (x["diff_score"], x["domain"], x["doc_id"]), reverse=True)
	selected = all_examples[:max_examples]

	with open(os.path.join(out_dir, "reference_vs_bart_examples.json"), "w", encoding="utf-8") as f:
		json.dump({"model_alias": model_alias, "count": len(selected), "examples": selected}, f, indent=2)

	if selected:
		flat = []
		for e in selected:
			flat.append(
				{
					"domain": e["domain"],
					"doc_id": e["doc_id"],
					"summary_changed": e["summary_changed"],
					"reference_success": e["reference_success"],
					"bart_success": e["bart_success"],
					"reference_error_rate": f"{100*e['reference_error_rate']:.1f}%",
					"bart_error_rate": f"{100*e['bart_error_rate']:.1f}%",
					"added_error_types_in_bart": "; ".join(e["added_error_types_in_bart"]),
					"removed_error_types_in_bart": "; ".join(e["removed_error_types_in_bart"]),
				}
			)
		pd.DataFrame(flat).to_csv(os.path.join(out_dir, "reference_vs_bart_examples.csv"), index=False)

	with open(os.path.join(out_dir, "reference_vs_bart_examples.md"), "w", encoding="utf-8") as f:
		f.write(f"# Reference vs {model_alias} Example Comparisons\n\n")
		f.write(f"Saved examples: {len(selected)}\n\n")
		for i, e in enumerate(selected, start=1):
			f.write(f"## Example {i}: {e['domain']} / {e['doc_id']}\n\n")
			f.write(f"- Summary changed: {e['summary_changed']}\n")
			f.write(f"- Reference success: {e['reference_success']}\n")
			f.write(f"- {model_alias} success: {e['bart_success']}\n")
			f.write(f"- Added error types in {model_alias}: {', '.join(e['added_error_types_in_bart']) or 'none'}\n")
			f.write(f"- Removed error types in {model_alias}: {', '.join(e['removed_error_types_in_bart']) or 'none'}\n\n")
			f.write("Ground-truth summary:\n")
			f.write(e["ground_truth_summary"] + "\n\n")
			f.write(f"{model_alias} summary:\n")
			f.write(e["bart_summary"] + "\n\n")
			f.write("FineSurE response on ground-truth summary (truncated):\n")
			f.write(e["reference_llm_output"] + "\n\n")
			f.write(f"FineSurE response on {model_alias} summary (truncated):\n")
			f.write(e["bart_llm_output"] + "\n\n")


def main(args):
	print("=" * 80)
	print("Cross-domain: reference vs FRANK-style model summaries")
	print("=" * 80)

	samsum_ds = load_huggingface_dataset("knkarthick/samsum", split="test", max_samples=args.samsum_samples, seed=args.seed)
	govreport_ds = load_huggingface_dataset("ccdv/govreport-summarization", split="test", max_samples=args.govreport_samples, seed=args.seed)

	samsum_ref = prepare_domain_records(samsum_ds, "samsum")
	gov_ref = prepare_domain_records(govreport_ds, "govreport")

	selected_aliases = [m.strip() for m in args.summary_models.split(",") if m.strip()]
	for alias in selected_aliases:
		if alias not in FRANK_STYLE_MODELS:
			raise ValueError(f"Unknown summary model alias: {alias}. Choices: {list(FRANK_STYLE_MODELS)}")

	groups = {
		"SAMSum | reference": samsum_ref,
		"GovReport | reference": gov_ref,
	}

	for alias in selected_aliases:
		groups[f"SAMSum | {alias}"] = build_generated_records(
			samsum_ref,
			alias=alias,
			device=args.hf_device,
			max_new_tokens=args.gen_max_new_tokens,
			max_input_tokens=args.max_input_tokens,
		)
		groups[f"GovReport | {alias}"] = build_generated_records(
			gov_ref,
			alias=alias,
			device=args.hf_device,
			max_new_tokens=args.gen_max_new_tokens,
			max_input_tokens=args.max_input_tokens,
		)

	results_dict = {}
	for name, rows in groups.items():
		print(f"\nRunning FineSurE on: {name} (n={len(rows)})")
		results_dict[name] = run_finesure_fact_checking(rows, model_key=args.finesure_model, use_strict=True)

	analysis = analyze_results(results_dict)
	save_outputs(results_dict, analysis, args.output_dir)
	save_example_comparisons(
		results_dict,
		args.output_dir,
		model_alias=args.example_model,
		max_examples=args.example_count,
	)

	print("\nSaved outputs to:", args.output_dir)
	print("Summary CSV:", os.path.join(args.output_dir, "reference_vs_model_summary.csv"))
	print("Examples:", os.path.join(args.output_dir, "reference_vs_bart_examples.json"))
	print("Examples:", os.path.join(args.output_dir, "reference_vs_bart_examples.md"))


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Compare FineSurE on reference summaries vs FRANK-style model summaries.")
	parser.add_argument("--output-dir", type=str, default="reproduce/results/cross_domain_reference_vs_model")
	parser.add_argument("--finesure-model", type=str, default="qwen2.5-7b-direct")
	parser.add_argument("--summary-models", type=str, default="bart,s2s")
	parser.add_argument("--samsum-samples", type=int, default=20)
	parser.add_argument("--govreport-samples", type=int, default=20)
	parser.add_argument("--gen-max-new-tokens", type=int, default=64)
	parser.add_argument("--max-input-tokens", type=int, default=256)
	parser.add_argument("--hf-device", type=int, default=0, help="HF pipeline device (GPU index). Use 0 for single-GPU runs.")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--example-model", type=str, default="bart")
	parser.add_argument("--example-count", type=int, default=5)

	main(parser.parse_args())

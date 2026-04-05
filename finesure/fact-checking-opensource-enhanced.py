"""
Enhanced factuality categorization runner for open-source models.
This file is additive and keeps baseline scripts unchanged.
"""

import argparse
import json
import os

from utils_opensource import (
    compute_faithfulness_percentage_score,
    get_fact_checking_prompt,
    get_response_opensource,
    parsing_llm_fact_checking_output,
)
from utils_categorization import (
    aggregate_self_consistency,
    build_entailment_prompt,
    build_error_subtype_prompt,
    build_strict_factcheck_prompt,
    parse_factcheck_json,
    parse_entailment_json,
    normalize_error_only_category,
)
from factuality_experiment_presets import get_preset


def _print_results(model_labels, cnt_success_inference, cnt_total_inference):
    sentence_level_errors = {}
    summary_level_scores = {}
    mean_confidence = {}

    for model_name, values in model_labels.items():
        sentence_level_errors[model_name] = sum(values["binary_labels"]) / len(values["binary_labels"])
        summary_level_scores[model_name] = sum(values["faithfulness_scores"]) / len(values["faithfulness_scores"])

        conf = values.get("category_confidence", [])
        mean_confidence[model_name] = sum(conf) / len(conf) if conf else 1.0

    text_output = "\n\n\n[Evaluation Results]\n"
    text_output += "* sentence-level factuality error ratio per model (lower is better)\n"
    for model_name, error_rate in sentence_level_errors.items():
        text_output += model_name + "\t" + str("{:.1%}".format(error_rate)) + "\n"

    text_output += "\n* summary-level faithfulness score per model (higher is better)\n"
    for model_name, score in summary_level_scores.items():
        text_output += model_name + "\t" + str("{:.1%}".format(score)) + "\n"

    text_output += "\n* mean category confidence (higher is better)\n"
    for model_name, score in mean_confidence.items():
        text_output += model_name + "\t" + str("{:.1%}".format(score)) + "\n"

    text_output += "\n* system-level model ranking (left is better)\n"
    sorted_dict = dict(sorted(summary_level_scores.items(), key=lambda item: item[1], reverse=True))
    model_ranking = list(sorted_dict.keys())
    text_output += str(model_ranking) + "\n"

    success_ratio = "{:.1%}".format(cnt_success_inference / float(cnt_total_inference)) if cnt_total_inference else "0.0%"
    text_output += "\n* success rate: " + str(success_ratio) + "\n\n\n"

    print(text_output)
    return text_output


def _build_enhanced_log_entry(
    doc_id,
    source_model,
    success_flag,
    run_diagnostics,
    final_labels,
    final_categories,
    category_confidence,
    faithfulness_score,
):
    sample_count = len(run_diagnostics)
    successful_samples = sum(1 for item in run_diagnostics if item.get("categorization_success"))

    return {
        "doc_id": doc_id,
        "source_model": source_model,
        "doc_success": success_flag,
        "sample_count": sample_count,
        "successful_samples": successful_samples,
        "run_diagnostics": run_diagnostics,
        "final": {
            "labels": final_labels,
            "categories": final_categories,
            "category_confidence": category_confidence,
            "faithfulness_score": faithfulness_score,
        },
    }


def main(args):
    print("Using model:", args.model_key)
    print("Input:", args.input_path)
    print("Output:", args.output_path)
    print("Preset:", args.preset)
    print("Reasoning mode:", args.reasoning_mode)
    print("Prompt style:", args.prompt_style)
    print("n_samples:", args.n_samples)
    print("temperature:", args.temperature)

    inputs = []
    with open(args.input_path, "r") as reader:
        for line in reader:
            inputs.append(json.loads(line))

    os.makedirs(args.output_path, exist_ok=True)
    raw_data_path = os.path.join(args.output_path, "raw-data.json")
    result_path = os.path.join(args.output_path, "result.json")
    enhanced_diag_path = os.path.join(args.output_path, "enhanced-diagnostics.jsonl")
    enhanced_summary_path = os.path.join(args.output_path, "enhanced-summary.json")

    cnt_total_inference = 0
    cnt_success_inference = 0
    model_labels = {}
    enhanced_stats = {
        "documents": 0,
        "successful_documents": 0,
        "two_stage_documents": 0,
        "strict_mode_documents": 0,
        "self_consistency_documents": 0,
        "samples_total": 0,
        "samples_successful": 0,
    }

    with open(raw_data_path, "w") as raw_data_writer, open(enhanced_diag_path, "w") as enhanced_diag_writer:
        for input_id, input_json in enumerate(inputs):
            doc_id = input_json["doc_id"]
            model_name = input_json["model"]
            src = input_json["transcript"]
            sentences = input_json["sentences"]

            enhanced_stats["documents"] += 1
            if args.reasoning_mode == "two_stage":
                enhanced_stats["two_stage_documents"] += 1
            if args.prompt_style == "strict":
                enhanced_stats["strict_mode_documents"] += 1
            if args.n_samples > 1:
                enhanced_stats["self_consistency_documents"] += 1

            if args.reasoning_mode == "two_stage":
                # Stage-1 in two-stage mode must produce entailment labels.
                prompt = build_entailment_prompt(transcript=src, sentences=sentences)
            elif args.prompt_style == "strict":
                prompt = build_strict_factcheck_prompt(transcript=src, sentences=sentences)
            else:
                prompt = get_fact_checking_prompt(input=src, sentences=sentences)

            print("\n" + "=" * 80)
            print("[{}/{}] Processing doc_id: {}, model: {}".format(input_id + 1, len(inputs), doc_id, model_name))
            print("=" * 80)

            run_outputs = []
            parsed_runs = []
            run_diagnostics = []

            cnt_total_inference += 1

            for sample_idx in range(args.n_samples):
                try:
                    output = get_response_opensource(
                        prompt=prompt,
                        model_key=args.model_key,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                    )

                    if args.prompt_style == "strict":
                        if args.reasoning_mode == "two_stage":
                            entailment_parsed = parse_entailment_json(output, expected_num_sentences=len(sentences))
                            run_outputs.append(output)

                            if not entailment_parsed.get("success", False):
                                run_diagnostics.append(
                                    {
                                        "sample_idx": sample_idx,
                                        "mode": "two_stage",
                                        "categorization_success": False,
                                        "categorization_error": "entailment_stage_failed: {}".format(
                                            entailment_parsed.get("error", "unknown")
                                        ),
                                    }
                                )
                            else:
                                stage1_labels = entailment_parsed.get("labels", [])
                                error_indices = [idx for idx, value in enumerate(stage1_labels) if value == 1]
                                final_categories = ["no error"] * len(sentences)
                                subtype_success = True
                                subtype_error = ""

                                if error_indices:
                                    error_sentences = [sentences[idx] for idx in error_indices]
                                    subtype_prompt = build_error_subtype_prompt(src, error_sentences)
                                    subtype_output = get_response_opensource(
                                        prompt=subtype_prompt,
                                        model_key=args.model_key,
                                        temperature=args.temperature,
                                        max_tokens=args.max_tokens,
                                    )
                                    subtype_parsed = parse_factcheck_json(
                                        subtype_output,
                                        expected_num_sentences=len(error_sentences),
                                    )
                                    run_outputs.append(subtype_output)

                                    if not subtype_parsed.get("success", False):
                                        subtype_success = False
                                        subtype_error = subtype_parsed.get("error", "subtype_stage_failed")
                                    else:
                                        subtype_categories = subtype_parsed.get("categories", [])
                                        for local_idx, sent_idx in enumerate(error_indices):
                                            final_categories[sent_idx] = normalize_error_only_category(
                                                subtype_categories[local_idx]
                                            )

                                labels = stage1_labels
                                parsed_success = subtype_success and len(labels) == len(sentences)

                                run_diagnostics.append(
                                    {
                                        "sample_idx": sample_idx,
                                        "mode": "two_stage",
                                        "categorization_success": parsed_success,
                                        "categorization_error": "" if parsed_success else subtype_error,
                                        "error_sentence_count": len(error_indices),
                                    }
                                )

                                if parsed_success:
                                    parsed_runs.append(
                                        {
                                            "success": True,
                                            "labels": labels,
                                            "categories": final_categories,
                                            "reasons": [""] * len(final_categories),
                                        }
                                    )

                        else:
                            parsed = parse_factcheck_json(output, expected_num_sentences=len(sentences))
                            run_outputs.append(output)
                            labels = parsed.get("labels", [])
                            categories = parsed.get("categories", [])
                            parsed_success = bool(parsed.get("success", False))
                            run_diagnostics.append(
                                {
                                    "sample_idx": sample_idx,
                                    "mode": "single_stage",
                                    "categorization_success": parsed_success,
                                    "categorization_error": "" if parsed_success else parsed.get("error", "strict_categorization_parse_failed"),
                                }
                            )
                            if parsed_success:
                                parsed_runs.append(
                                    {
                                        "success": True,
                                        "labels": labels,
                                        "categories": categories,
                                        "reasons": [""] * len(categories),
                                    }
                                )
                    else:
                        labels, categories = parsing_llm_fact_checking_output(output)
                        run_outputs.append(output)
                        parsed_success = len(labels) == len(sentences) and len(labels) > 0
                        run_diagnostics.append(
                            {
                                "sample_idx": sample_idx,
                                "mode": "baseline",
                                "categorization_success": parsed_success,
                                "categorization_error": "" if parsed_success else "baseline_categorization_length_mismatch_or_empty",
                            }
                        )
                        if parsed_success:
                            parsed_runs.append(
                                {
                                    "success": True,
                                    "labels": labels,
                                    "categories": categories,
                                    "reasons": [""] * len(categories),
                                }
                            )

                except Exception as exc:
                    run_diagnostics.append(
                        {
                            "sample_idx": sample_idx,
                            "categorization_success": False,
                            "categorization_error": "runtime_error: {}".format(exc),
                        }
                    )

            success_flag = len(parsed_runs) > 0
            print("Success:", success_flag)

            sample_success_count = sum(1 for item in run_diagnostics if item.get("categorization_success"))

            enhanced_stats["samples_total"] += len(run_diagnostics)
            enhanced_stats["samples_successful"] += sample_success_count

            if not success_flag:
                print("\tSkipping - no successful parsed run")
                input_json["llm_output"] = run_outputs
                input_json["run_diagnostics"] = run_diagnostics

                failed_entry = _build_enhanced_log_entry(
                    doc_id=doc_id,
                    source_model=model_name,
                    success_flag=False,
                    run_diagnostics=run_diagnostics,
                    final_labels=[],
                    final_categories=[],
                    category_confidence=[],
                    faithfulness_score=None,
                )
                json.dump(failed_entry, enhanced_diag_writer)
                enhanced_diag_writer.write("\n")
                enhanced_diag_writer.flush()

                json.dump(input_json, raw_data_writer)
                raw_data_writer.write("\n")
                raw_data_writer.flush()
                continue

            cnt_success_inference += 1
            enhanced_stats["successful_documents"] += 1

            if len(parsed_runs) == 1:
                final_labels = parsed_runs[0]["labels"]
                final_categories = parsed_runs[0]["categories"]
                category_confidence = [1.0] * len(final_categories)
            else:
                aggregated = aggregate_self_consistency(parsed_runs)
                final_labels = aggregated["labels"]
                final_categories = aggregated["categories"]
                category_confidence = aggregated["confidence"]

            input_json["llm_output"] = run_outputs
            input_json["run_diagnostics"] = run_diagnostics
            input_json["pred_faithfulness_labels"] = final_labels
            input_json["pred_faithfulness_error_type"] = final_categories
            input_json["pred_category_confidence"] = category_confidence

            faithfulness_score = compute_faithfulness_percentage_score(final_labels)
            print("\t[Error Label]:", final_labels)
            print("\t[Error Type]:", final_categories)
            print("\t[Faithfulness Score]: {:.1%}".format(faithfulness_score))

            enhanced_entry = _build_enhanced_log_entry(
                doc_id=doc_id,
                source_model=model_name,
                success_flag=True,
                run_diagnostics=run_diagnostics,
                final_labels=final_labels,
                final_categories=final_categories,
                category_confidence=category_confidence,
                faithfulness_score=faithfulness_score,
            )
            json.dump(enhanced_entry, enhanced_diag_writer)
            enhanced_diag_writer.write("\n")
            enhanced_diag_writer.flush()

            if model_name not in model_labels:
                model_labels[model_name] = {
                    "faithfulness_scores": [],
                    "binary_labels": [],
                    "category_confidence": [],
                }
            model_labels[model_name]["faithfulness_scores"].append(faithfulness_score)
            model_labels[model_name]["binary_labels"].extend(final_labels)
            model_labels[model_name]["category_confidence"].extend(category_confidence)

            if cnt_total_inference % args.print_interval == 0:
                _print_results(model_labels, cnt_success_inference, cnt_total_inference)

            json.dump(input_json, raw_data_writer)
            raw_data_writer.write("\n")
            raw_data_writer.flush()

    with open(result_path, "w") as result_writer:
        if model_labels:
            text_output = _print_results(model_labels, cnt_success_inference, cnt_total_inference)
            result_writer.write(text_output)
        else:
            print("No successful evaluations.")

    summary_payload = {
        "mode": {
            "preset": args.preset,
            "reasoning_mode": args.reasoning_mode,
            "prompt_style": args.prompt_style,
            "n_samples": args.n_samples,
        },
        "aggregate": enhanced_stats,
        "derived": {
            "document_success_rate": (
                float(enhanced_stats["successful_documents"]) / float(enhanced_stats["documents"])
                if enhanced_stats["documents"]
                else 0.0
            ),
            "sample_success_rate": (
                float(enhanced_stats["samples_successful"]) / float(enhanced_stats["samples_total"])
                if enhanced_stats["samples_total"]
                else 0.0
            ),
        },
    }

    with open(enhanced_summary_path, "w") as enhanced_summary_writer:
        json.dump(summary_payload, enhanced_summary_writer, indent=2)

    print("Results saved to", args.output_path)
    print("Enhanced diagnostics saved to", enhanced_diag_path)
    print("Enhanced summary saved to", enhanced_summary_path)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Enhanced FineSurE factuality categorization runner")
    parser.add_argument("input_path", type=str, help="Input jsonl path")
    parser.add_argument("output_path", type=str, help="Output folder")
    parser.add_argument("--model-key", type=str, default="qwen2.5-7b", help="Model key from model_config.py")
    parser.add_argument(
        "--preset",
        type=str,
        default="baseline",
        choices=["baseline", "enhanced", "enhanced_sc"],
        help="Experiment preset",
    )
    parser.add_argument(
        "--reasoning-mode",
        type=str,
        default=None,
        choices=["single_stage", "two_stage"],
        help="Use single-stage categorization or two-stage entailment-first categorization",
    )
    parser.add_argument(
        "--prompt-style",
        type=str,
        default=None,
        choices=["baseline", "strict"],
        help="Override prompt style",
    )
    parser.add_argument("--n-samples", type=int, default=None, help="Override number of samples")
    parser.add_argument("--temperature", type=float, default=None, help="Override sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Generation max tokens")
    parser.add_argument("--print-interval", type=int, default=10, help="Print rolling results every N docs")
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    preset = get_preset(args.preset)

    if args.reasoning_mode is None:
        args.reasoning_mode = preset.get("reasoning_mode", "single_stage")
    if args.prompt_style is None:
        args.prompt_style = preset["prompt_style"]
    if args.n_samples is None:
        args.n_samples = preset["n_samples"]
    if args.temperature is None:
        args.temperature = preset["temperature"]

    if args.n_samples < 1:
        raise ValueError("n_samples must be >= 1")

    main(args)

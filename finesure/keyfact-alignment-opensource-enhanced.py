"""
Enhanced keyfact alignment evaluation with open-source model support.
Adds optional two-stage judging and self-consistency voting.
"""
import argparse
import ast
import math
import json
import os
import random
import time
from utils_opensource import (
    get_response_opensource,
    get_keyfact_alighment_prompt,
    parsing_llm_keyfact_alighment_output,
    compute_completeness_percentage_score,
    compute_conciseness_percentage_score,
)


def _build_numbered_summary(sentences):
    numbered = ["[" + str(idx + 1) + "] " + s for idx, s in enumerate(sentences)]
    return "\n".join(numbered)


def _build_stage1_prompt(keyfacts, sentences):
    summary = _build_numbered_summary(sentences)
    key_facts = "\n".join(keyfacts)
    num_key_facts = len(keyfacts)
    return f"""
You will receive a summary and a set of key facts for the same transcript.
Task: decide whether each key fact is inferable from the summary.

Instruction:
1) For each key fact, answer only "Yes" or "No".
2) Do not provide line numbers in this step.

Return ONLY valid JSON as a list of objects with keys "key fact" and "response":
[{{"key fact": "...", "response": "Yes"}}, {{"key fact": "...", "response": "No"}}]

The list must contain exactly {num_key_facts} objects.

Summary:
{summary}

{num_key_facts} key facts:
{key_facts}
"""


def _build_stage2_prompt(inferable_keyfacts, sentences):
    summary = _build_numbered_summary(sentences)
    key_facts = "\n".join(inferable_keyfacts)
    num_key_facts = len(inferable_keyfacts)
    return f"""
You will receive a summary and key facts that are already known to be inferable from the summary.
Task: for each key fact, provide all supporting summary line numbers.

Instruction:
1) Return line numbers only from the provided summary indices.
2) Use integers in ascending order.

Return ONLY valid JSON as a list of objects with keys "key fact" and "line number":
[{{"key fact": "...", "line number": [1, 3]}}]

The list must contain exactly {num_key_facts} objects.

Summary:
{summary}

Inferable key facts ({num_key_facts}):
{key_facts}
"""


def _safe_parse_json_list(output):
    text = output.replace("```json", "").replace("```", "").strip()
    start_idx = text.find("[")
    if start_idx == -1:
        return []
    end_idx = text.rfind("]")
    if end_idx == -1:
        return []
    payload = text[start_idx : end_idx + 1]
    return ast.literal_eval(payload)


def _parse_stage1_yes_no(output, expected_count):
    try:
        rows = _safe_parse_json_list(output)
        if not isinstance(rows, list) or len(rows) != expected_count:
            return []

        labels = []
        for row in rows:
            response = str(row.get("response", "")).strip().lower()
            labels.append(1 if response == "yes" else 0)
        return labels
    except Exception:
        return []


def _parse_stage2_lines(output, expected_count, num_sentences):
    try:
        rows = _safe_parse_json_list(output)
        if not isinstance(rows, list) or len(rows) != expected_count:
            return None

        matched = set()
        for row in rows:
            for line_num in row.get("line number", []):
                if isinstance(line_num, str):
                    line_num = line_num.replace("[", "").replace("]", "").strip()
                idx = int(line_num)
                if 1 <= idx <= num_sentences:
                    matched.add(idx)
        return sorted(matched)
    except Exception:
        return None


def _aggregate_samples(sample_predictions, num_keyfacts, num_sentences):
    successful = [s for s in sample_predictions if s.get("success")]
    if not successful:
        return [], []

    vote_threshold = int(math.ceil(len(successful) / 2.0))

    aggregated_labels = []
    for k_idx in range(num_keyfacts):
        yes_votes = sum(s["labels"][k_idx] for s in successful)
        aggregated_labels.append(1 if yes_votes >= vote_threshold else 0)

    line_counts = {}
    for sample in successful:
        for ln in sample["matched_lines"]:
            if 1 <= ln <= num_sentences:
                line_counts[ln] = line_counts.get(ln, 0) + 1

    aggregated_lines = sorted([ln for ln, c in line_counts.items() if c >= vote_threshold])
    return aggregated_labels, aggregated_lines


def _print_results_completeness(model_labels, cnt_success_inference, cnt_total_inference):
    summary_level_completeness_scores = {}
    summary_level_conciseness_scores = {}

    for model_name, error_labels in model_labels.items():
        summary_level_completeness_scores[model_name] = sum(error_labels["completeness_scores"]) / len(
            error_labels["completeness_scores"]
        )
        summary_level_conciseness_scores[model_name] = sum(error_labels["conciseness_scores"]) / len(
            error_labels["conciseness_scores"]
        )

    text_output = "\n\n\n[Evaluation Results]\n"
    text_output += "\n* completeness score per model (higher is better)\n"
    for model_name, score in summary_level_completeness_scores.items():
        text_output += model_name + "\t" + str("{:.1%}".format(score)) + "\n"

    text_output += "\n* completeness model ranking (left is better)\n"
    sorted_dict = dict(sorted(summary_level_completeness_scores.items(), key=lambda item: item[1], reverse=True))
    model_ranking = list(sorted_dict.keys())
    text_output += str(model_ranking) + "\n"

    text_output += "\n* conciseness score per model (higher is better)\n"
    for model_name, score in summary_level_conciseness_scores.items():
        text_output += model_name + "\t" + str("{:.1%}".format(score)) + "\n"

    text_output += "\n* conciseness model ranking (left is better)\n"
    sorted_dict = dict(sorted(summary_level_conciseness_scores.items(), key=lambda item: item[1], reverse=True))
    model_ranking = list(sorted_dict.keys())
    text_output += str(model_ranking) + "\n"

    success_ratio = "{:.1%}".format(cnt_success_inference / float(cnt_total_inference)) if cnt_total_inference else "0.0%"
    text_output += "\n* success rate: " + str(success_ratio) + "\n\n\n"

    print(text_output)
    return text_output


def _build_example_key(input_json):
    return f"{input_json.get('doc_id', '')}::{input_json.get('model', '')}"


def _bootstrap_from_existing_raw(raw_data_path, keyfacts, model_labels):
    """Load prior progress from raw-data.json for resume support."""
    done_keys = set()
    cnt_total_inference = 0
    cnt_success_inference = 0

    if not os.path.exists(raw_data_path):
        return done_keys, cnt_total_inference, cnt_success_inference

    with open(raw_data_path, "r") as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue

            done_keys.add(_build_example_key(row))
            cnt_total_inference += 1

            doc_id = row.get("doc_id")
            model_name = row.get("model")
            pred_labels = row.get("pred_alignment_labels", [])
            pred_line_numbers = row.get("pred_sentence_line_numbers", [])

            if doc_id not in keyfacts or model_name is None:
                continue

            expected_count = len(keyfacts[doc_id])
            success_flag = len(pred_labels) == expected_count and len(pred_labels) > 0
            if not success_flag:
                continue

            cnt_success_inference += 1
            completeness_score = compute_completeness_percentage_score(pred_labels)
            conciseness_score = compute_conciseness_percentage_score(pred_line_numbers, len(row.get("sentences", [])))

            if model_name not in model_labels:
                model_labels[model_name] = {"completeness_scores": [], "conciseness_scores": []}
            model_labels[model_name]["completeness_scores"].append(completeness_score)
            model_labels[model_name]["conciseness_scores"].append(conciseness_score)

    return done_keys, cnt_total_inference, cnt_success_inference


def main(args):
    """
    Argument:
        input_path: path for input data
        keyfact_path: path for human or machine keyfacts
        output_path: path for output data (saving the logs and the eval results)
        model_key: model to use (from model_config.py)
        print_interval: print the percentage scores every 'print_interval'
    """

    print(f"Using model: {args.model_key}")
    print(f"Input: {args.input_path}")
    print(f"Keyfacts: {args.keyfact_path}")
    print(f"Output: {args.output_path}")
    print(f"Reasoning mode: {args.reasoning_mode}")
    print(f"n_samples: {args.n_samples}\n")
    run_start = time.perf_counter()

    # loads data for completeness and conciseness evaluation using FineSurE
    inputs = []
    for line in open(args.input_path, "r"):
        line = json.loads(line)
        inputs.append(line)

    if args.shuffle_input:
        random.Random(args.sample_seed).shuffle(inputs)
    if args.max_examples is not None:
        inputs = inputs[: args.max_examples]

    print(f"Loaded examples: {len(inputs)}")

    # loads keyfacts
    keyfacts = {}
    for line in open(args.keyfact_path, "r"):
        line = json.loads(line)
        keyfacts[line["doc_id"]] = line["key_facts"]

    # variables for evaluation
    model_labels = {}

    # writer to store the output from LLM evaluation
    os.makedirs(args.output_path, exist_ok=True)
    raw_data_path = os.path.join(args.output_path, "raw-data.json")
    done_keys, cnt_total_inference, cnt_success_inference = _bootstrap_from_existing_raw(
        raw_data_path=raw_data_path,
        keyfacts=keyfacts,
        model_labels=model_labels,
    )
    if done_keys:
        print(f"Resuming from existing raw-data.json: {len(done_keys)} already processed example(s)")

    raw_data_writer = open(raw_data_path, "a")
    result_writer = open(os.path.join(args.output_path, "result.json"), "w")
    summary_path = os.path.join(args.output_path, "enhanced-summary.json")

    runtime_stats = {
        "new_documents_processed": 0,
        "new_successful_documents": 0,
        "new_documents_runtime_sec": 0.0,
        "new_sample_calls": 0,
    }

    # processes each data instance using for loop
    for input_id, input_json in enumerate(inputs):

        # input json parsing
        doc_id = input_json["doc_id"]
        model_name = input_json["model"]
        sentences = input_json["sentences"]
        list_keyfacts = keyfacts[doc_id]
        example_key = _build_example_key(input_json)

        if example_key in done_keys:
            continue

        print(f"\n{'='*80}")
        print(f"[{input_id+1}/{len(inputs)}] Processing doc_id: {doc_id}, model: {model_name}")
        print(f"{'='*80}")

        cnt_total_inference += 1
        doc_start = time.perf_counter()

        try:
            sample_predictions = []
            run_outputs = []

            for _sample_idx in range(args.n_samples):
                runtime_stats["new_sample_calls"] += 1
                print(
                    f"\t[Sample {_sample_idx+1}/{args.n_samples}] mode={args.reasoning_mode}",
                    flush=True,
                )
                if args.reasoning_mode == "single_stage":
                    prompt = get_keyfact_alighment_prompt(keyfacts=list_keyfacts, sentences=sentences)
                    output = get_response_opensource(
                        prompt=prompt,
                        model_key=args.model_key,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                    )
                    run_outputs.append(output)
                    print(f"\t[Sample {_sample_idx+1}] single-stage response received", flush=True)
                    labels, matched_lines = parsing_llm_keyfact_alighment_output(output)
                    success = len(labels) == len(list_keyfacts) and len(labels) > 0
                    sample_predictions.append(
                        {
                            "success": success,
                            "labels": labels if success else [],
                            "matched_lines": matched_lines if success else [],
                        }
                    )
                else:
                    stage1_prompt = _build_stage1_prompt(keyfacts=list_keyfacts, sentences=sentences)
                    stage1_output = get_response_opensource(
                        prompt=stage1_prompt,
                        model_key=args.model_key,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                    )
                    run_outputs.append(stage1_output)
                    print(f"\t[Sample {_sample_idx+1}] stage-1 response received", flush=True)

                    stage1_labels = _parse_stage1_yes_no(stage1_output, expected_count=len(list_keyfacts))
                    if len(stage1_labels) != len(list_keyfacts):
                        sample_predictions.append({"success": False, "labels": [], "matched_lines": []})
                        continue

                    yes_keyfacts = [kf for kf, y in zip(list_keyfacts, stage1_labels) if y == 1]
                    if yes_keyfacts:
                        stage2_prompt = _build_stage2_prompt(inferable_keyfacts=yes_keyfacts, sentences=sentences)
                        stage2_output = get_response_opensource(
                            prompt=stage2_prompt,
                            model_key=args.model_key,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                        )
                        run_outputs.append(stage2_output)
                        print(f"\t[Sample {_sample_idx+1}] stage-2 response received", flush=True)
                        matched_lines = _parse_stage2_lines(
                            stage2_output,
                            expected_count=len(yes_keyfacts),
                            num_sentences=len(sentences),
                        )
                        if matched_lines is None:
                            sample_predictions.append({"success": False, "labels": [], "matched_lines": []})
                            continue
                    else:
                        matched_lines = []

                    sample_predictions.append(
                        {
                            "success": True,
                            "labels": stage1_labels,
                            "matched_lines": matched_lines,
                        }
                    )

            pred_labels, pred_line_numbers = _aggregate_samples(
                sample_predictions,
                num_keyfacts=len(list_keyfacts),
                num_sentences=len(sentences),
            )

            input_json["llm_output"] = run_outputs[0] if len(run_outputs) == 1 else run_outputs
            input_json["pred_alignment_labels"] = pred_labels
            input_json["pred_sentence_line_numbers"] = pred_line_numbers

            # check if the parsing is success
            success_flag = len(pred_labels) == len(list_keyfacts) and len(pred_labels) > 0

            print(f"Success: {success_flag}")
            print(f'\t[Alignment Label]: {input_json["pred_alignment_labels"]}')
            print(f'\t[Matched Sentence Line Numbers]: {input_json["pred_sentence_line_numbers"]}')

            # count the success cases
            if success_flag:
                cnt_success_inference += 1
                runtime_stats["new_successful_documents"] += 1
            else:
                # fail to evaluate -> skip
                print("\t⚠️  Skipping - parsing failed")
                json.dump(input_json, raw_data_writer)
                raw_data_writer.write("\n")
                raw_data_writer.flush()
                runtime_stats["new_documents_processed"] += 1
                runtime_stats["new_documents_runtime_sec"] += (time.perf_counter() - doc_start)
                continue

            # compute the percentage score for completeness and conciseness
            completeness_score = compute_completeness_percentage_score(input_json["pred_alignment_labels"])
            conciseness_score = compute_conciseness_percentage_score(input_json["pred_sentence_line_numbers"], len(sentences))

            # put the score into the aggregation dictionary
            if model_name not in model_labels:
                model_labels[model_name] = {"completeness_scores": [], "conciseness_scores": []}
            model_labels[model_name]["completeness_scores"].append(completeness_score)
            model_labels[model_name]["conciseness_scores"].append(conciseness_score)

            print(f'\t[Completeness Score]: {completeness_score:.1%}')
            print(f'\t[Conciseness Score]: {conciseness_score:.1%}')

        except Exception as e:
            print(f"\t❌ Error processing example: {e}")
            import traceback
            traceback.print_exc()
            continue

        # print percentage score
        if cnt_total_inference % args.print_interval == 0:
            _print_results_completeness(model_labels, cnt_success_inference, cnt_total_inference)

        json.dump(input_json, raw_data_writer)
        raw_data_writer.write("\n")
        raw_data_writer.flush()
        done_keys.add(example_key)
        runtime_stats["new_documents_processed"] += 1
        runtime_stats["new_documents_runtime_sec"] += (time.perf_counter() - doc_start)

    raw_data_writer.close()

    # print final results
    if model_labels:
        text_output = _print_results_completeness(model_labels, cnt_success_inference, cnt_total_inference)
        result_writer.write(text_output)
    else:
        print("\n⚠️  No successful evaluations!")

    result_writer.close()

    total_runtime_sec = time.perf_counter() - run_start
    summary_payload = {
        "mode": {
            "reasoning_mode": args.reasoning_mode,
            "n_samples": args.n_samples,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "max_examples": args.max_examples,
            "shuffle_input": args.shuffle_input,
            "sample_seed": args.sample_seed,
        },
        "resume": {
            "already_processed_examples": len(done_keys) - runtime_stats["new_documents_processed"],
            "total_processed_examples": len(done_keys),
        },
        "aggregate": {
            "cnt_total_inference": cnt_total_inference,
            "cnt_success_inference": cnt_success_inference,
            "success_rate": (float(cnt_success_inference) / float(cnt_total_inference)) if cnt_total_inference else 0.0,
        },
        "runtime": {
            "total_runtime_sec": total_runtime_sec,
            "new_documents_processed": runtime_stats["new_documents_processed"],
            "new_successful_documents": runtime_stats["new_successful_documents"],
            "new_documents_runtime_sec": runtime_stats["new_documents_runtime_sec"],
            "avg_sec_per_new_document": (
                runtime_stats["new_documents_runtime_sec"] / runtime_stats["new_documents_processed"]
                if runtime_stats["new_documents_processed"] else 0.0
            ),
            "avg_sec_per_new_successful_document": (
                runtime_stats["new_documents_runtime_sec"] / runtime_stats["new_successful_documents"]
                if runtime_stats["new_successful_documents"] else 0.0
            ),
            "new_sample_calls": runtime_stats["new_sample_calls"],
            "avg_sec_per_sample_call": (
                runtime_stats["new_documents_runtime_sec"] / runtime_stats["new_sample_calls"]
                if runtime_stats["new_sample_calls"] else 0.0
            ),
        },
    }
    with open(summary_path, "w") as summary_writer:
        json.dump(summary_payload, summary_writer, indent=2)

    print(f"\n✅ Results saved to {args.output_path}")
    print(f"✅ Summary saved to {summary_path}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Enhanced keyfact alignment evaluation with open-source model support")
    parser.add_argument("input_path", type=str, help="Path to input jsonl")
    parser.add_argument("keyfact_path", type=str, help="Path to keyfact jsonl")
    parser.add_argument("output_path", type=str, help="Output folder")
    parser.add_argument("model_key", nargs="?", default="qwen2.5-7b", help="Model key from model_config.py")
    parser.add_argument(
        "--reasoning-mode",
        type=str,
        default="two_stage",
        choices=["single_stage", "two_stage"],
        help="single_stage uses original prompt; two_stage does Yes/No then supporting line extraction",
    )
    parser.add_argument("--n-samples", type=int, default=1, help="Number of samples for self-consistency majority vote")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum generation tokens")
    parser.add_argument("--print-interval", type=int, default=10, help="Print rolling results every N docs")
    parser.add_argument("--max-examples", type=int, default=None, help="Run only first N examples after optional shuffle")
    parser.add_argument("--shuffle-input", action="store_true", help="Shuffle input before slicing to max-examples")
    parser.add_argument("--sample-seed", type=int, default=42, help="Seed for input shuffle")
    return parser


if __name__ == "__main__":
    """
    Running Command:
        python finesure/keyfact-alignment-opensource-enhanced.py [input-path] [keyfact-path] [output-folder] [model-key]

        e.g., python finesure/keyfact-alignment-opensource-enhanced.py dataset/realsumm/realsumm-data-sample-10.json dataset/realsumm/human-keyfact-list.json result/keyfact-qwen-enhanced qwen2.5-7b-direct
    """
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.n_samples < 1:
        raise ValueError("n_samples must be >= 1")

    main(args)

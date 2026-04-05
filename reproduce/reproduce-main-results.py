
import json
import sys
import os
import argparse
import re
import numpy as np
from utils import compute_faithfulness_percentage_score, compute_completeness_percentage_score
from scipy.stats import pearsonr, spearmanr
import scipy.stats as ss


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

CATEGORY_NAME_TO_INDEX = {
    "no error": 0,
    "out-of-context error": 1,
    "entity error": 2,
    "predicate error": 3,
    "circumstantial error": 4,
    "grammatical error": 5,
    "coreference error": 6,
    "linking error": 7,
    "other error": 8,
}

INDEX_TO_CATEGORY_NAME = {v: k for k, v in CATEGORY_NAME_TO_INDEX.items()}


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return value


def _corr_to_dict(corr_obj):
    return {
        "statistic": _safe_float(getattr(corr_obj, "statistic", None)),
        "pvalue": _safe_float(getattr(corr_obj, "pvalue", None)),
    }


def _render_markdown_report(payload):
    faith = payload.get("faithfulness", {})
    comp = payload.get("completeness", {})
    conc = payload.get("conciseness", {})

    per_cat_rows = ""
    for row in faith.get("per_category", []):
        per_cat_rows += (
            f"| {row.get('category')} | {row.get('precision', 0.0):.4f} | {row.get('recall', 0.0):.4f} | {row.get('f1', 0.0):.4f} | {row.get('support', 0)} |\n"
        )

    confusion_rows = ""
    for row in faith.get("confusion_matrix", []):
        confusion_rows += (
            f"- {row.get('gt')}: {json.dumps(row.get('pred_counts', {}), ensure_ascii=True)}\n"
        )

    markdown = f"""# FineSurE Evaluation Report

## Faithfulness
- Sentence bAcc: {faith.get('sentence', {}).get('balanced_accuracy', 0.0):.4f}
- Sentence Macro-F1: {faith.get('sentence', {}).get('macro_f1', 0.0):.4f}
- Summary Pearson: {faith.get('summary', {}).get('pearson', {}).get('statistic', 0.0):.4f}
- Summary Spearman: {faith.get('summary', {}).get('spearman', {}).get('statistic', 0.0):.4f}
- System Rank Corr: {faith.get('system', {}).get('rank_correlation', {}).get('statistic', 0.0):.4f}
- Success Ratio: {faith.get('success_ratio', 0.0):.4f}

### Per-category Metrics
| Category | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
{per_cat_rows}

### Category Confusion Matrix (gt -> pred counts)
{confusion_rows}

## Completeness
- Summary Pearson: {comp.get('summary', {}).get('pearson', {}).get('statistic', 0.0):.4f}
- Summary Spearman: {comp.get('summary', {}).get('spearman', {}).get('statistic', 0.0):.4f}
- System Rank Corr: {comp.get('system', {}).get('rank_correlation', {}).get('statistic', 0.0):.4f}

## Conciseness
- Summary Pearson: {conc.get('summary', {}).get('pearson', {}).get('statistic', 0.0):.4f}
- Summary Spearman: {conc.get('summary', {}).get('spearman', {}).get('statistic', 0.0):.4f}
- System Rank Corr: {conc.get('system', {}).get('rank_correlation', {}).get('statistic', 0.0):.4f}
- Success Ratio: {conc.get('success_ratio', 0.0):.4f}
"""
    return markdown


def write_reports(payload, report_dir):
    os.makedirs(report_dir, exist_ok=True)
    json_path = os.path.join(report_dir, "evaluation-report.json")
    markdown_path = os.path.join(report_dir, "evaluation-report.md")

    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    with open(markdown_path, "w") as f:
        f.write(_render_markdown_report(payload))

    print("\n[Report Files]")
    print("- JSON:", json_path)
    print("- Markdown:", markdown_path)

def main(frank_result_path, realsumm_result_path, report_dir=None):

    # 1. load frank data
    frank_results = []
    for line in open(frank_result_path, 'r'):
        line = json.loads(line)
        frank_results.append(line)

    faith_metrics = faithfulness_eval(frank_results)

    # load realsumm data
    realsumm_results = []
    for line in open(realsumm_result_path, 'r'):
        line = json.loads(line)
        realsumm_results.append(line)
        
    comp_metrics, conc_metrics = completeness_and_conciseness_eval(realsumm_results)

    payload = {
        "faithfulness": faith_metrics,
        "completeness": comp_metrics,
        "conciseness": conc_metrics,
    }

    if report_dir:
        write_reports(payload, report_dir)


def faithfulness_eval(results):
    '''
    A function to evaluate the results from FineSurE on faithfulness at the three different levels
    '''

    # faithfulness eval
    model_wise_results = {}
    conv_ids = []
    task_keys = []
    summary_keys = []
    full_results = {
        # general factaulity (sentence-level)¸¸
        'gt_faithfulness_binary_labels': [],
        'pred_faithfulness_binary_labels': [],
        # general factaulity (summary-level)
        'gt_faithfulness_scores': [],
        'pred_faithfulness_scores': [],
        'gt_faithfulness_type_labels': [],
        'pred_faithfulness_type_labels': [],
    }

    cnt_success_inference = 0

    for result in results:
        conv_id = result['doc_id']
        dataset_name = result['source']
        model = result['model']

        # dict for computing the overall scores, and system-wise ranking
        if model not in model_wise_results:
            model_wise_results[model] = {
                'gt_faithfulness_binary_labels': [],
                'pred_faithfulness_binary_labels': [],
                'gt_faithfulness_scores': [],
                'pred_faithfulness_scores': [],
            }
    
        conv_ids.append(conv_id + model)

        # get gt labels and pred labels
        gt_faithfulness_binary_labels = get_aggregate_gt_labels(result['raw_annotations'], key="factuality_labels")
        pred_faithfulness_binary_labels = result.get('pred_faithfulness_labels', [])

        gt_faithfulness_type_labels = get_aggregate_gt_type_labels(result['raw_annotations'])
        pred_faithfulness_type_labels = result.get('pred_faithfulness_error_type', None)
        recovered_pred_types = extract_pred_types_from_llm_output(result.get('llm_output', None))

        if len(gt_faithfulness_binary_labels) != len(pred_faithfulness_binary_labels):
            # failure cases
            continue

        _gt_faithfulness_binary_labels, _pred_faithfulness_binary_labels = [], []
        _gt_faithfulness_type_labels, _pred_faithfulness_type_labels = [], []
        for idx, item in enumerate(gt_faithfulness_binary_labels):
            # exception handler
            if item != 'None':
                _gt_faithfulness_binary_labels.append(gt_faithfulness_binary_labels[idx])
                _pred_faithfulness_binary_labels.append(pred_faithfulness_binary_labels[idx])

                if idx < len(gt_faithfulness_type_labels):
                    gt_type = gt_faithfulness_type_labels[idx]
                else:
                    gt_type = 'None'

                if pred_faithfulness_type_labels is not None and idx < len(pred_faithfulness_type_labels):
                    pred_type = normalize_pred_type(pred_faithfulness_type_labels[idx])
                elif idx < len(recovered_pred_types):
                    pred_type = normalize_pred_type(recovered_pred_types[idx])
                else:
                    pred_type = 'other error' if float(pred_faithfulness_binary_labels[idx]) == 1.0 else 'no error'

                if gt_type != 'None':
                    _gt_faithfulness_type_labels.append(gt_type)
                    _pred_faithfulness_type_labels.append(pred_type)

        gt_faithfulness_binary_labels, pred_faithfulness_binary_labels = np.array(_gt_faithfulness_binary_labels), np.array(_pred_faithfulness_binary_labels)
        
        if len(gt_faithfulness_binary_labels) == 0:
            # failure cases
            continue

        key = dataset_name + '-' + conv_id + '-' + model
        for sentence_id in range(len(gt_faithfulness_binary_labels)):
            task_keys.append(key + '-' + str(sentence_id + 1))
        summary_keys.append(key)

        cnt_success_inference += 1
  
        #### compute summary-level
        gt_faithfulness_score = compute_faithfulness_percentage_score(gt_faithfulness_binary_labels) 
        pred_faithfulness_score = compute_faithfulness_percentage_score(pred_faithfulness_binary_labels) 

        full_results['gt_faithfulness_binary_labels'].extend(gt_faithfulness_binary_labels)
        full_results['pred_faithfulness_binary_labels'].extend(pred_faithfulness_binary_labels)
        full_results['gt_faithfulness_scores'].append(gt_faithfulness_score)
        full_results['pred_faithfulness_scores'].append(pred_faithfulness_score)
        full_results['gt_faithfulness_type_labels'].extend(_gt_faithfulness_type_labels)
        full_results['pred_faithfulness_type_labels'].extend(_pred_faithfulness_type_labels)

        model_wise_results[model]['gt_faithfulness_binary_labels'].extend(gt_faithfulness_binary_labels)
        model_wise_results[model]['pred_faithfulness_binary_labels'].extend(pred_faithfulness_binary_labels)
        model_wise_results[model]['gt_faithfulness_scores'].append(gt_faithfulness_score)
        model_wise_results[model]['pred_faithfulness_scores'].append(pred_faithfulness_score)

    print('[Faithfulness Evaluation]')

    print('* Sentence-level')
    bAcc = balancedAcc(full_results['gt_faithfulness_binary_labels'], full_results['pred_faithfulness_binary_labels'])
    print('\t-Balanced Accuracy:', '{:.1%}'.format(bAcc))

    if len(full_results['gt_faithfulness_type_labels']) > 0:
        print('\t-Macro F1 (Category):', '{:.1%}'.format(macro_f1(full_results['gt_faithfulness_type_labels'], full_results['pred_faithfulness_type_labels'])))
        print('\t-Per-category Precision/Recall/F1:')
        per_category = per_category_stats(full_results['gt_faithfulness_type_labels'], full_results['pred_faithfulness_type_labels'])
        for item in per_category:
            print('\t\t{}\tP:{:.1%}\tR:{:.1%}\tF1:{:.1%}\tN:{}'.format(
                item['category'], item['precision'], item['recall'], item['f1'], item['support']
            ))

        print('\t-Category Confusion Matrix (gt -> pred counts):')
        confusion = category_confusion_matrix(full_results['gt_faithfulness_type_labels'], full_results['pred_faithfulness_type_labels'])
        for row in confusion:
            print('\t\t{} -> {}'.format(row['gt'], row['pred_counts']))

    pearson_corr = pearsonr(full_results['gt_faithfulness_scores'], full_results['pred_faithfulness_scores'])
    spearman_corr = spearmanr(full_results['gt_faithfulness_scores'], full_results['pred_faithfulness_scores'])

    print('* Summary-level:')
    print("\t-Pearson:", pearson_corr)
    print("\t-Spearman:", spearman_corr)

    print('* System-level:')
    # model-wise ranking 
    _rank_correlation = rank_correlation(model_wise_results, key="faithfulness_scores")
    print("\t-Rank Correlation:", _rank_correlation)

    success_rate = cnt_success_inference / len(conv_ids)
    print('* Success ratio', '{:.1%}'.format(success_rate))

    faith_metrics = {
        "sentence": {
            "balanced_accuracy": _safe_float(bAcc),
            "macro_f1": _safe_float(macro_f1(full_results['gt_faithfulness_type_labels'], full_results['pred_faithfulness_type_labels'])) if len(full_results['gt_faithfulness_type_labels']) > 0 else 0.0,
        },
        "summary": {
            "pearson": _corr_to_dict(pearson_corr),
            "spearman": _corr_to_dict(spearman_corr),
        },
        "system": {
            "rank_correlation": _corr_to_dict(_rank_correlation),
        },
        "success_ratio": _safe_float(success_rate),
        "per_category": per_category_stats(full_results['gt_faithfulness_type_labels'], full_results['pred_faithfulness_type_labels']) if len(full_results['gt_faithfulness_type_labels']) > 0 else [],
        "confusion_matrix": category_confusion_matrix(full_results['gt_faithfulness_type_labels'], full_results['pred_faithfulness_type_labels']) if len(full_results['gt_faithfulness_type_labels']) > 0 else [],
    }

    return faith_metrics


def completeness_and_conciseness_eval(results):
    '''
    A function to evaluate the results from FineSurE on faithfulness at the three different levels
    '''

    # faithfulness eval
    model_wise_results = {}
    conv_ids = []
    task_keys = []
    summary_keys = []
    full_results = {
        'gt_completeness_scores': [],
        'pred_completeness_scores': [],
        'gt_conciseness_scores': [],
        'pred_conciseness_scores': [],
    }

    cnt_success_inference = 0

    for result in results:
        conv_id = result['doc_id']
        dataset_name = result['source']
        model = result['model']

        # dict for computing the overall scores, and system-wise ranking
        if model not in model_wise_results:
            model_wise_results[model] = {
                'gt_completeness_scores': [],
                'pred_completeness_scores': [],
                'gt_conciseness_scores': [],
                'pred_conciseness_scores': [],
            }
    
        conv_ids.append(conv_id + model)

        # get gt labels and pred labels
        gt_alignment_labels = get_aggregate_gt_labels(result['raw_annotations'], key="key_fact_labels")
        gt_sentence_line_numbers = get_aggregate_gt_labels(result['raw_annotations'], key="sentence_labels")
        pred_alignment_labels = result['pred_alignment_labels']
        pred_sentence_line_numbers = result['pred_sentence_line_numbers']

        # failure cases
        if len(gt_alignment_labels) != len(pred_alignment_labels):
            continue
        _gt_alignment_labels, _pred_alignment_labels = [], []
        for idx, item in enumerate(gt_alignment_labels):
            if item != 'None':
                _gt_alignment_labels.append(gt_alignment_labels[idx])
                _pred_alignment_labels.append(pred_alignment_labels[idx])
        gt_alignment_labels, pred_alignment_labels = np.array(_gt_alignment_labels), np.array(_pred_alignment_labels)

        key = dataset_name + '-' + conv_id + '-' + model
        for key_fact_id in range(len(gt_alignment_labels)):
            task_keys.append(key + '-' + str(key_fact_id + 1))
        summary_keys.append(key)

        cnt_success_inference += 1

        # compute completeness percentage
        gt_completeness_score = compute_completeness_percentage_score(gt_alignment_labels)
        pred_completeness_score = compute_completeness_percentage_score(pred_alignment_labels)

        # compute conciseness percentage
        _pred_sentence_line_numbers = []
        for idx in range(len(gt_sentence_line_numbers)):
            if (idx +1) in pred_sentence_line_numbers:
                _pred_sentence_line_numbers.append(1.0)
            else:
                _pred_sentence_line_numbers.append(0.0)
        pred_sentence_line_numbers = _pred_sentence_line_numbers

      
        assert len(gt_sentence_line_numbers) == len(pred_sentence_line_numbers)
        gt_conciseness_score = sum(gt_sentence_line_numbers) / len(gt_sentence_line_numbers)
        pred_conciseness_score = sum(pred_sentence_line_numbers) / len(pred_sentence_line_numbers)

        full_results['gt_completeness_scores'].append(gt_completeness_score)
        full_results['pred_completeness_scores'].append(pred_completeness_score)
        full_results['gt_conciseness_scores'].append(gt_conciseness_score)
        full_results['pred_conciseness_scores'].append(pred_conciseness_score)

        model_wise_results[model]['gt_completeness_scores'].append(gt_completeness_score)
        model_wise_results[model]['pred_completeness_scores'].append(pred_completeness_score)
        model_wise_results[model]['gt_conciseness_scores'].append(gt_conciseness_score)
        model_wise_results[model]['pred_conciseness_scores'].append(pred_conciseness_score)

    print('\n[Completeness Evaluation]')

    pearson_corr = pearsonr(full_results['gt_completeness_scores'], full_results['pred_completeness_scores'])
    spearman_corr = spearmanr(full_results['gt_completeness_scores'], full_results['pred_completeness_scores'])

    print('* Summary-level:')
    print("\t-Pearson:", pearson_corr)
    print("\t-Spearman:", spearman_corr)

    print('* System-level:')
    # model-wise ranking 
    _rank_correlation = rank_correlation(model_wise_results, key="completeness_scores")
    print("\t-Rank Correlation:", _rank_correlation)

    print('\n[Conciseness Evaluation]')

    pearson_corr = pearsonr(full_results['gt_conciseness_scores'], full_results['pred_conciseness_scores'])
    spearman_corr = spearmanr(full_results['gt_conciseness_scores'], full_results['pred_conciseness_scores'])

    print('* Summary-level:')
    print("\t-Pearson:", pearson_corr)
    print("\t-Spearman:", spearman_corr)

    print('* System-level:')
    # model-wise ranking 
    _rank_correlation = rank_correlation(model_wise_results, key="conciseness_scores")
    print("\t-Rank Correlation:", _rank_correlation)


    success_rate = cnt_success_inference / len(conv_ids)
    print('\n* Success ratio', '{:.1%}'.format(success_rate))

    completeness_metrics = {
        "summary": {
            "pearson": _corr_to_dict(pearsonr(full_results['gt_completeness_scores'], full_results['pred_completeness_scores'])),
            "spearman": _corr_to_dict(spearmanr(full_results['gt_completeness_scores'], full_results['pred_completeness_scores'])),
        },
        "system": {
            "rank_correlation": _corr_to_dict(rank_correlation(model_wise_results, key="completeness_scores")),
        },
    }

    conciseness_metrics = {
        "summary": {
            "pearson": _corr_to_dict(pearsonr(full_results['gt_conciseness_scores'], full_results['pred_conciseness_scores'])),
            "spearman": _corr_to_dict(spearmanr(full_results['gt_conciseness_scores'], full_results['pred_conciseness_scores'])),
        },
        "system": {
            "rank_correlation": _corr_to_dict(rank_correlation(model_wise_results, key="conciseness_scores")),
        },
        "success_ratio": _safe_float(success_rate),
    }

    return completeness_metrics, conciseness_metrics
  

def get_aggregate_gt_labels(raw_annotations, key):
    '''
    A function to generate the aggregated human labels from three annotators
    Args:
        - raw_annotations: the raw annotations from three annotators
        - key: the annotation type ('xxx:' )
    Returns:
        - final_labels: the aggregated labels by majority voting
    '''

    # if there are four annotators, we remove the last
    if key == "sentence_labels" and "3" in raw_annotations:
        del raw_annotations["3"]

    merged_gt_labels = []
    for worker_id, annotation in raw_annotations.items():
        gt_labels = annotation[key]    
        merged_gt_labels.append(gt_labels)

    final_labels = []
    merged_gt_labels = np.array(merged_gt_labels)
    num_labels = len(merged_gt_labels[-1])

    for sent_idx in range(num_labels):
        _column = merged_gt_labels[:, sent_idx]
        _column = [float(item) for item in _column if item != 'None']

        if len(_column) <= 1:
            final_labels.append('None')
        else:
            final_label = max(set(_column), key = _column.count)
            final_labels.append(float(final_label))

    assert len(final_labels) == num_labels

    return final_labels


def get_aggregate_gt_type_labels(raw_annotations):
    """Aggregate annotator factuality type labels to one category per sentence."""
    merged_type_labels = []
    for _, annotation in raw_annotations.items():
        merged_type_labels.append(annotation.get("factuality_types", []))

    if len(merged_type_labels) == 0:
        return []

    num_labels = len(merged_type_labels[-1])
    final_types = []

    for sent_idx in range(num_labels):
        votes = {}
        for worker_types in merged_type_labels:
            if sent_idx >= len(worker_types):
                continue
            cell = worker_types[sent_idx]
            if isinstance(cell, list):
                labels = cell
            else:
                labels = [cell]

            for label in labels:
                if label is None:
                    continue
                label = str(label)
                if label == 'None':
                    continue
                category = CATEGORY_CODE_TO_NAME.get(label, 'other error')
                votes[category] = votes.get(category, 0) + 1

        if len(votes) == 0:
            final_types.append('None')
        else:
            final_types.append(max(votes.items(), key=lambda x: x[1])[0])

    return final_types


def normalize_pred_type(value):
    if value is None:
        return 'other error'
    text = str(value).strip().lower().replace('_', ' ')
    text = ' '.join(text.split())

    aliases = {
        'out of context error': 'out-of-context error',
        'out-of-context': 'out-of-context error',
        'factual': 'no error',
        'no factual error': 'no error',
        'entity mismatch': 'entity error',
        'predicate mismatch': 'predicate error',
        'coref error': 'coreference error',
    }

    normalized = aliases.get(text, text)
    if normalized in CATEGORY_NAME_TO_INDEX:
        return normalized
    return 'other error'


def extract_pred_types_from_llm_output(llm_output):
    """Extract predicted category sequence from raw llm_output text when explicit labels are missing."""
    if llm_output is None:
        return []

    if isinstance(llm_output, list):
        text = "\n".join([str(item) for item in llm_output])
    else:
        text = str(llm_output)

    # Remove markdown code fences if present.
    text = text.replace("```json", "").replace("```", "")

    # Support both JSON-style and python-literal-style keys/quotes.
    patterns = [
        r'"category"\s*:\s*"([^\"]+)"',
        r"'category'\s*:\s*'([^']+)'",
    ]

    extracted = []
    for pattern in patterns:
        extracted = re.findall(pattern, text, flags=re.IGNORECASE)
        if len(extracted) > 0:
            break

    return [normalize_pred_type(item) for item in extracted]


def per_category_stats(gt_types, pred_types):
    rows = []
    for category in CATEGORY_NAME_TO_INDEX.keys():
        tp = 0
        fp = 0
        fn = 0
        support = 0
        for gt, pred in zip(gt_types, pred_types):
            if gt == category:
                support += 1
            if gt == category and pred == category:
                tp += 1
            elif gt != category and pred == category:
                fp += 1
            elif gt == category and pred != category:
                fn += 1

        precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
        recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        rows.append({
            'category': category,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
        })

    return rows


def macro_f1(gt_types, pred_types):
    stats = per_category_stats(gt_types, pred_types)
    return float(np.mean([row['f1'] for row in stats])) if len(stats) > 0 else 0.0


def category_confusion_matrix(gt_types, pred_types):
    matrix = np.zeros((len(CATEGORY_NAME_TO_INDEX), len(CATEGORY_NAME_TO_INDEX)), dtype=np.int32)

    for gt, pred in zip(gt_types, pred_types):
        gt_idx = CATEGORY_NAME_TO_INDEX.get(gt, CATEGORY_NAME_TO_INDEX['other error'])
        pred_idx = CATEGORY_NAME_TO_INDEX.get(pred, CATEGORY_NAME_TO_INDEX['other error'])
        matrix[gt_idx, pred_idx] += 1

    rows = []
    for gt_idx in range(len(CATEGORY_NAME_TO_INDEX)):
        gt_name = INDEX_TO_CATEGORY_NAME[gt_idx]
        pred_counts = {}
        for pred_idx in range(len(CATEGORY_NAME_TO_INDEX)):
            value = int(matrix[gt_idx, pred_idx])
            if value > 0:
                pred_counts[INDEX_TO_CATEGORY_NAME[pred_idx]] = value
        rows.append({'gt': gt_name, 'pred_counts': pred_counts})

    return rows


def balancedAcc(gt, pred):
    '''
    A function to compute the balanced accuracy
    Args:
        - gt: ground truth labels
        - pred: predicted labels
    Return:
        - balanced accuracy
    '''
    ones, zeros = [], []
    for idx in range(len(gt)):
        if gt[idx] == 1.0:
            ones.append(pred[idx])
        elif gt[idx] == 0.0:
            zeros.append(pred[idx])

    error_acc = sum(ones) / len(ones)
    non_error_acc =  1.0 - sum(zeros) / len(zeros)

    return (error_acc + non_error_acc) / 2.0


def rank_correlation(model_wise_results, key, min_number=5):
    '''
    A function to compute the balanced accuracy
    Args:
        - model_wise_results: evaluation results per model in dict
        - key: evaluation dimension
        - min_number: the minimum number of examples to be included in the evaluation
    Return:
        - rank correlation with p value
    '''

    model_list =  model_wise_results.keys()

    models = []
    gt_errors = []
    pred_errors = []
    for model_name in model_list:
        models.append(model_name)
        gt_error, pred_error = np.mean(model_wise_results[model_name]['gt_' + key]), np.mean(model_wise_results[model_name]['pred_' + key])

        if len(model_wise_results[model_name]['gt_' + key]) >= min_number:
            gt_errors.append(gt_error)
            pred_errors.append(pred_error)

    pred_errors = np.array(pred_errors) 
    gt_errors = np.array(gt_errors) 

    estimated_rank = ss.rankdata(pred_errors)
    human_rank = ss.rankdata(gt_errors)
    #print("models:", models)
    #print('gt ' + key + ':', gt_errors)
    #print('pred ' + key + ':', pred_errors )
    #print('gt rank ' + key + ':', human_rank)
    #print('pred rank ' + key + ':', estimated_rank)
    spearman_corr = spearmanr(estimated_rank, human_rank)

    return spearman_corr



if __name__ == "__main__":

    '''
    Runnining Command:
        cd CodeRelease/reproduce
        python reproduce-main-results.py results/frank-result-by-gpt4-w-finesure.json results/realsumm-result-by-gpt4-w-finesure.json
    '''

    parser = argparse.ArgumentParser(description="Reproduce FineSurE results with optional report export")
    parser.add_argument("frank_result_path", type=str, help="Path to frank result jsonl")
    parser.add_argument("realsumm_result_path", type=str, help="Path to realsumm result jsonl")
    parser.add_argument("--report-dir", type=str, default=None, help="Optional folder to save JSON/HTML report")
    args = parser.parse_args()

    main(args.frank_result_path, args.realsumm_result_path, args.report_dir)



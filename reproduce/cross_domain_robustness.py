"""
Cross-Domain Robustness Evaluation for FineSurE (Enhanced Version)
Loads SAMSum (dialogue) and GovReport (long-form) datasets and runs enhanced fact-checking
to compare FineSurE's behavior across different summarization styles.

Uses stricter fact-checking prompts and enhanced error categorization for better accuracy.
"""

import json
import os
import sys
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

# Add project and module directories to path for imports.
# `utils_opensource.py` imports `model_config` as a top-level module,
# so `finesure/` must also be on sys.path when this script runs from `reproduce/`.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FINESURE_DIR = os.path.join(PROJECT_ROOT, 'finesure')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, FINESURE_DIR)

from finesure.utils_opensource import (
    get_response_opensource,
    parsing_llm_fact_checking_output,
    get_fact_checking_prompt,
    compute_faithfulness_percentage_score
)
from finesure.utils_categorization import (
    normalize_category,
    build_strict_factcheck_prompt,
    parse_factcheck_json,
)


def load_huggingface_dataset(dataset_name: str, split: str = "test", max_samples: int = 100):
    """Load dataset from HuggingFace Hub."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install datasets: pip install datasets")
        return []
    
    try:
        dataset = load_dataset(dataset_name, split=split)
    except Exception as e:
        print(f"Warning: Could not load {dataset_name} split {split}: {e}")
        try:
            dataset = load_dataset(dataset_name, split="train")
        except:
            return []
    
    if max_samples and len(dataset) > max_samples:
        indices = np.random.choice(len(dataset), max_samples, replace=False)
        dataset = dataset.select(indices)
    
    return dataset


def prepare_samsum_data(dataset, max_sentences=3):
    """Convert SAMSum dataset to FineSurE format."""
    prepared = []
    for idx, item in enumerate(dataset):
        # SAMSum has 'dialogue' and 'summary'
        dialogue = item.get('dialogue', '')
        summary = item.get('summary', '')
        
        if not dialogue or not summary:
            continue
        
        # Split summary into sentences
        sentences = [s.strip() for s in summary.split('.') if s.strip()]
        if len(sentences) > max_sentences:
            sentences = sentences[:max_sentences]
        
        prepared.append({
            'doc_id': f"samsum-{idx}",
            'source': 'samsum',
            'split': 'test',
            'model': 'ground-truth',
            'transcript': dialogue,
            'reference': summary,
            'sentences': sentences,
            'raw_annotations': {}  # No human labels for zero-shot
        })
    
    return prepared


def prepare_govreport_data(dataset, max_sentences=4):
    """Convert GovReport dataset to FineSurE format."""
    prepared = []
    for idx, item in enumerate(dataset):
        # GovReport has 'report' and 'summary'
        report = item.get('report', '')
        summary = item.get('summary', '')
        
        if not report or not summary:
            continue
        
        # Split summary into sentences
        sentences = [s.strip() for s in summary.split('.') if s.strip()]
        if len(sentences) > max_sentences:
            sentences = sentences[:max_sentences]
        
        prepared.append({
            'doc_id': f"govreport-{idx}",
            'source': 'govreport',
            'split': 'test',
            'model': 'ground-truth',
            'transcript': report,
            'reference': summary,
            'sentences': sentences,
            'raw_annotations': {}  # No human labels for zero-shot
        })
    
    return prepared


def prepare_frank_data(frank_path: str, max_samples: int = 100):
    """Load FRANK in-domain baseline."""
    prepared = []
    count = 0
    for line in open(frank_path, 'r'):
        if count >= max_samples:
            break
        data = json.loads(line)
        prepared.append(data)
        count += 1
    return prepared


def run_fact_checking(data_list, model_key: str = "qwen2.5-7b", output_dir: str = None, use_strict: bool = True):
    """Run enhanced FineSurE fact-checking on a list of prepared samples.
    
    Uses strict fact-checking prompt for better error categorization.
    """
    results = []
    success_count = 0
    
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, item in enumerate(data_list):
        print(f"  [{idx+1}/{len(data_list)}] {item['doc_id']}")
        
        sentences = item['sentences']
        transcript = item['transcript']
        
        try:
            # Generate prompt using stricter version if enabled
            if use_strict:
                prompt = build_strict_factcheck_prompt(transcript=transcript, sentences=sentences)
            else:
                prompt = get_fact_checking_prompt(input=transcript, sentences=sentences)
            
            # Get LLM response
            output = get_response_opensource(
                prompt=prompt,
                model_key=model_key,
                temperature=0.0,
                max_tokens=2048
            )
            
            # Parse response
            if use_strict:
                # Try strict JSON parsing first
                parsed = parse_factcheck_json(output, expected_num_sentences=len(sentences))
                if parsed.get('success', False):
                    pred_labels = parsed.get('labels', [])
                    pred_categories = parsed.get('categories', [])
                    # Normalize categories
                    pred_types = [normalize_category(cat) for cat in pred_categories]
                    success = True
                else:
                    # Fall back to baseline parsing
                    pred_labels, pred_types = parsing_llm_fact_checking_output(output)
                    success = len(pred_labels) == len(sentences) and len(pred_labels) > 0
            else:
                # Baseline parsing
                pred_labels, pred_types = parsing_llm_fact_checking_output(output)
                success = len(pred_labels) == len(sentences) and len(pred_labels) > 0
            
            # Validation
            success = success and len(pred_labels) == len(sentences) and len(pred_labels) > 0
            
            item['llm_output'] = output
            item['pred_faithfulness_labels'] = pred_labels
            item['pred_faithfulness_error_type'] = pred_types
            item['success'] = success
            
            if success:
                success_count += 1
            
            results.append(item)
            
        except Exception as e:
            print(f"    Error: {str(e)[:100]}")
            item['success'] = False
            item['error'] = str(e)
            results.append(item)
    
    print(f"  Success rate: {success_count}/{len(data_list)} ({100*success_count/len(data_list):.1f}%)")
    
    return results


def analyze_results(results_dict: dict):
    """Compute comparative statistics across domains with enhanced categorization."""
    
    # Enhanced category mappings
    CATEGORY_MAP = {
        'NoE': 'no error',
        'EntE': 'entity error',
        'RelE': 'predicate error',
        'OutE': 'out-of-context error',
        'CircE': 'circumstantial error',
        'GramE': 'grammatical error',
        'CorefE': 'coreference error',
        'LinkE': 'linking error',
        'OtherE': 'other error',
        # Also accept full names from enhanced version
        'no error': 'no error',
        'entity error': 'entity error',
        'predicate error': 'predicate error',
        'out-of-context error': 'out-of-context error',
        'circumstantial error': 'circumstantial error',
        'grammatical error': 'grammatical error',
        'coreference error': 'coreference error',
        'linking error': 'linking error',
        'other error': 'other error',
    }
    
    analysis = {}
    
    for domain_name, results in results_dict.items():
        print(f"\n{'='*80}")
        print(f"Domain: {domain_name}")
        print(f"{'='*80}")
        
        # Basic stats
        total = len(results)
        successful = sum(1 for r in results if r.get('success', False))
        success_rate = successful / total if total > 0 else 0
        
        print(f"Total samples: {total}")
        print(f"Successful evaluations: {successful} ({100*success_rate:.1f}%)")
        
        # Error type frequency
        error_types = []
        error_labels = []
        sentences_per_sample = []
        category_diversity = Counter()  # Track unique categories per sample
        
        for result in results:
            if result.get('success', False):
                pred_types = result.get('pred_faithfulness_error_type', [])
                pred_labels = result.get('pred_faithfulness_labels', [])
                
                # Normalize types using the mapping
                normalized_types = []
                for t in pred_types:
                    mapped = CATEGORY_MAP.get(t, t)
                    normalized_types.append(mapped)
                
                error_types.extend(normalized_types)
                error_labels.extend(pred_labels)
                sentences_per_sample.append(len(result.get('sentences', [])))
                
                # Track category diversity for this sample
                unique_categories = set(normalized_types)
                category_diversity.update(unique_categories)
        
        # Error type distribution
        error_type_counts = Counter(error_types)
        total_predictions = len(error_types)
        
        print(f"\nError type distribution (out of {total_predictions} predictions):")
        for error_type, count in error_type_counts.most_common():
            pct = 100 * count / total_predictions if total_predictions > 0 else 0
            print(f"  {error_type:25s}: {count:4d} ({pct:5.1f}%)")
        
        # No-error rate (faithfulness)
        no_error_count = error_type_counts.get('no error', 0)
        no_error_rate = no_error_count / total_predictions if total_predictions > 0 else 0
        
        print(f"\nFaithfulness (% no-error): {100*no_error_rate:.1f}%")
        
        # Error severity: count sentences with at least one error
        error_labels_array = np.array(error_labels)
        error_rate = (error_labels_array == 1).sum() / len(error_labels_array) if len(error_labels_array) > 0 else 0
        
        print(f"Error sentence ratio: {100*error_rate:.1f}%")
        
        # Average summary length
        avg_sentences = np.mean(sentences_per_sample) if sentences_per_sample else 0
        print(f"Avg sentences per summary: {avg_sentences:.2f}")
        
        # Category diversity (# of unique error types found)
        num_error_categories = len([t for t in error_type_counts if t != 'no error'])
        print(f"Distinct error categories found: {num_error_categories}/8")
        
        analysis[domain_name] = {
            'total': total,
            'successful': successful,
            'success_rate': success_rate,
            'error_type_counts': dict(error_type_counts),
            'total_predictions': total_predictions,
            'faithfulness': no_error_rate,
            'error_rate': error_rate,
            'avg_sentences': avg_sentences,
            'num_error_categories': num_error_categories,
        }
    
    return analysis


def save_summary(results_dict: dict, analysis: dict, output_dir: str):
    """Save detailed summary as JSON and CSV with enhanced metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # JSON summary
    summary = {
        'domains': list(results_dict.keys()),
        'analysis': analysis,
        'raw_sample_outputs': {
            domain: [
                {
                    'doc_id': r['doc_id'],
                    'sentences': r.get('sentences', [])[:2],  # First 2 sentences
                    'pred_error_types': r.get('pred_faithfulness_error_type', [])[:2],
                    'success': r.get('success', False)
                }
                for r in results_dict[domain][:5]  # First 5 samples
            ]
            for domain in results_dict.keys()
        }
    }
    
    with open(os.path.join(output_dir, 'cross_domain_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: cross_domain_summary.json")
    
    # CSV summary with enhanced metrics
    csv_data = []
    for domain, stats in analysis.items():
        csv_data.append({
            'Domain': domain,
            'Total Samples': stats['total'],
            'Successful': stats['successful'],
            'Success Rate (%)': f"{100*stats['success_rate']:.1f}",
            'Faithfulness (%)': f"{100*stats['faithfulness']:.1f}",
            'Error Rate (%)': f"{100*stats['error_rate']:.1f}",
            'Avg Sentences': f"{stats['avg_sentences']:.2f}",
            'Error Categories': f"{stats['num_error_categories']}/8",
        })
    
    df = pd.DataFrame(csv_data)
    df.to_csv(os.path.join(output_dir, 'cross_domain_summary.csv'), index=False)
    print(f"Saved: cross_domain_summary.csv")


def main(args):
    print("="*80)
    print("Cross-Domain Robustness Evaluation for FineSurE")
    print("="*80)
    
    output_base = args.output_dir
    os.makedirs(output_base, exist_ok=True)
    
    results_dict = {}
    
    # 1. Load FRANK in-domain baseline
    print("\n[1/4] Loading FRANK (in-domain baseline)...")
    frank_path = args.frank_path
    frank_data = prepare_frank_data(frank_path, max_samples=args.frank_samples)
    print(f"  Loaded {len(frank_data)} FRANK samples")
    
    print("\n  Running fact-checking on FRANK...")
    frank_output_dir = os.path.join(output_base, 'frank')
    frank_results = run_fact_checking(frank_data, model_key=args.model, output_dir=frank_output_dir, use_strict=True)
    results_dict['FRANK (in-domain)'] = frank_results
    
    # 2. Load and evaluate SAMSum
    print("\n[2/4] Loading SAMSum (dialogue, out-of-domain)...")
    samsum_dataset = load_huggingface_dataset('knkarthick/samsum', split='test', max_samples=args.samsum_samples)
    if samsum_dataset:
        samsum_data = prepare_samsum_data(samsum_dataset)
        print(f"  Loaded {len(samsum_data)} SAMSum samples")
        
        print("\n  Running fact-checking on SAMSum...")
        samsum_output_dir = os.path.join(output_base, 'samsum')
        samsum_results = run_fact_checking(samsum_data, model_key=args.model, output_dir=samsum_output_dir, use_strict=True)
        results_dict['SAMSum (dialogue)'] = samsum_results
    else:
        print("  Failed to load SAMSum")
    
    # 3. Load and evaluate GovReport
    print("\n[3/4] Loading GovReport (long-form, out-of-domain)...")
    govreport_dataset = load_huggingface_dataset('ccdv/govreport-summarization', split='test', max_samples=args.govreport_samples)
    if govreport_dataset:
        govreport_data = prepare_govreport_data(govreport_dataset)
        print(f"  Loaded {len(govreport_data)} GovReport samples")
        
        print("\n  Running fact-checking on GovReport...")
        govreport_output_dir = os.path.join(output_base, 'govreport')
        govreport_results = run_fact_checking(govreport_data, model_key=args.model, output_dir=govreport_output_dir, use_strict=True)
        results_dict['GovReport (long-form)'] = govreport_results
    else:
        print("  Failed to load GovReport")
    
    # 4. Analyze and compare results
    print("\n[4/4] Analyzing and comparing results...")
    analysis = analyze_results(results_dict)
    
    # Save outputs
    print("\nSaving summaries...")
    save_summary(results_dict, analysis, output_base)
    
    print("\n" + "="*80)
    print("Cross-domain robustness evaluation complete!")
    print(f"Results saved to: {output_base}")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cross-domain robustness evaluation for FineSurE')
    parser.add_argument('--frank-path', type=str, 
                       default='dataset/frank/frank-data.json',
                       help='Path to FRANK dataset')
    parser.add_argument('--output-dir', type=str,
                       default='reproduce/results/cross_domain_robustness',
                       help='Output directory for results')
    parser.add_argument('--model', type=str, default='qwen2.5-7b',
                       help='Model to use for fact-checking')
    parser.add_argument('--frank-samples', type=int, default=50,
                       help='Number of FRANK samples to evaluate')
    parser.add_argument('--samsum-samples', type=int, default=50,
                       help='Number of SAMSum samples to evaluate')
    parser.add_argument('--govreport-samples', type=int, default=50,
                       help='Number of GovReport samples to evaluate')
    
    args = parser.parse_args()
    main(args)

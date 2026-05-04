#!/usr/bin/env python3
"""
Efficiency & RQ4 Analysis
Aggregates runtime and efficiency metrics from all enhanced-summary.json outputs
Generates RQ4 efficiency comparison matrix
"""

import json
from pathlib import Path

def load_efficiency_data(report_dir):
    """Load runtime/efficiency data from enhanced-summary.json"""
    summary_path = Path(report_dir) / "enhanced-summary.json"
    if not summary_path.exists():
        return None
    
    try:
        with open(summary_path) as f:
            data = json.load(f)
        
        # Extract efficiency fields across known schema variants.
        efficiency = {
            'total_runtime_sec': None,
            'avg_sec_per_document': None,
            'total_samples': None,
        }

        # Newer format used in current runs: data['runtime']
        runtime = data.get('runtime', {}) if isinstance(data.get('runtime', {}), dict) else {}
        if runtime:
            efficiency['total_runtime_sec'] = runtime.get('total_runtime_sec', efficiency['total_runtime_sec'])
            efficiency['avg_sec_per_document'] = runtime.get('avg_sec_per_document', efficiency['avg_sec_per_document'])
            if efficiency['avg_sec_per_document'] is None:
                efficiency['avg_sec_per_document'] = runtime.get('avg_sec_per_new_document', efficiency['avg_sec_per_document'])
            efficiency['total_samples'] = runtime.get('new_sample_calls', efficiency['total_samples'])

        # Older format: data['runtime_stats']
        runtime_stats = data.get('runtime_stats', {}) if isinstance(data.get('runtime_stats', {}), dict) else {}
        if runtime_stats:
            efficiency['total_runtime_sec'] = runtime_stats.get('total_runtime_sec', efficiency['total_runtime_sec'])
            efficiency['avg_sec_per_document'] = runtime_stats.get('avg_sec_per_document', efficiency['avg_sec_per_document'])
            efficiency['total_samples'] = runtime_stats.get('samples_successful', efficiency['total_samples'])

        # Legacy fallback: top-level keys
        if efficiency['total_runtime_sec'] is None:
            efficiency['total_runtime_sec'] = data.get('new_documents_runtime_sec')
        if efficiency['avg_sec_per_document'] is None:
            efficiency['avg_sec_per_document'] = data.get('avg_sec_per_new_document')

        if efficiency['total_runtime_sec'] is None or efficiency['avg_sec_per_document'] is None:
            return None

        return efficiency
    except Exception as e:
        print(f"Error reading {summary_path}: {e}")
        return None

def main():
    base_path = Path("/common/home/vv382/FineSurE-Ext/reproduce/results/reports")
    repo_root = Path("/common/home/vv382/FineSurE-Ext")
    
    print("\n" + "="*100)
    print("RQ4: EFFICIENCY & COST ANALYSIS")
    print("="*100)
    
    # Define experiment groups for comparison
    groups = {
        "FRANK Faithfulness (Qwen)": [
            ("base", "Baseline"),
            ("enhanced", "Enhanced"),
            ("enhanced_sc_qwen_100", "Enhanced+SC"),
        ],
        "REALSumm Keyfact SC (Qwen)": [
            ("keyfact_two_stage_sc_qwen_100", "Two-Stage+SC"),
            ("keyfact_machine_sc_qwen_100", "Machine+SC"),
        ],
    }

    cross_domain_cc_runs = [
        (
            repo_root / "reproduce/results/cross_domain_reference_vs_model/bart/completeness_conciseness/SAMSum_reference",
            "SAMSum | reference",
        ),
        (
            repo_root / "reproduce/results/cross_domain_reference_vs_model/bart/completeness_conciseness/SAMSum_bart",
            "SAMSum | bart",
        ),
        (
            repo_root / "reproduce/results/cross_domain_reference_vs_model/bart/completeness_conciseness/GovReport_reference",
            "GovReport | reference",
        ),
        (
            repo_root / "reproduce/results/cross_domain_reference_vs_model/bart/completeness_conciseness/GovReport_bart",
            "GovReport | bart",
        ),
    ]
    
    print("\n## EFFICIENCY METRICS BY EXPERIMENT GROUP")
    print("(Runtime per document in seconds)")
    
    all_comparisons = []
    
    for group_name, experiments in groups.items():
        print(f"\n### {group_name}")
        print(f"{'Variant':<25} | {'Status':<10} | {'Avg Sec/Doc':<15} | {'Total Runtime':<15}")
        print("-" * 75)
        
        for exp_name, variant_name in experiments:
            report_dir = base_path / exp_name
            
            if report_dir.exists():
                eff = load_efficiency_data(report_dir)
                status = "✅"
                
                if eff:
                    avg_sec = eff.get('avg_sec_per_document', 0)
                    total_sec = eff.get('total_runtime_sec', 0)
                    total_hours = total_sec / 3600
                    
                    print(f"{variant_name:<25} | {status:<10} | {avg_sec:<15.4f} | {total_hours:<15.2f}h")
                    all_comparisons.append({
                        'group': group_name,
                        'variant': variant_name,
                        'avg_sec_per_doc': avg_sec,
                        'total_runtime_sec': total_sec,
                    })
                else:
                    print(f"{variant_name:<25} | {'⚠️ NO_RUNTIME':<10} | {'No data':<15} | {'No data':<15}")
            else:
                status = "❌"
                print(f"{variant_name:<25} | {status:<10} | {'MISSING':<15} | {'MISSING':<15}")

    print("\n### Cross-Domain Completeness/Conciseness (Qwen)")
    print(f"{'Variant':<25} | {'Status':<10} | {'Avg Sec/Doc':<15} | {'Total Runtime':<15}")
    print("-" * 75)

    for run_dir, variant_name in cross_domain_cc_runs:
        if run_dir.exists():
            eff = load_efficiency_data(run_dir)
            status = "✅"

            if eff:
                avg_sec = eff.get('avg_sec_per_document', 0)
                total_sec = eff.get('total_runtime_sec', 0)
                total_hours = total_sec / 3600

                print(f"{variant_name:<25} | {status:<10} | {avg_sec:<15.4f} | {total_hours:<15.2f}h")
                all_comparisons.append({
                    'group': 'Cross-Domain Completeness/Conciseness (Qwen)',
                    'variant': variant_name,
                    'avg_sec_per_doc': avg_sec,
                    'total_runtime_sec': total_sec,
                })
            else:
                print(f"{variant_name:<25} | {'⚠️ NO_RUNTIME':<10} | {'No data':<15} | {'No data':<15}")
        else:
            status = "❌"
            print(f"{variant_name:<25} | {status:<10} | {'MISSING':<15} | {'MISSING':<15}")
    
    # Summary statistics
    if all_comparisons:
        print("\n" + "="*100)
        print("## EFFICIENCY TRENDS")
        print("="*100)
        
        # Compare enhanced vs baseline
        frank_qwen = [c for c in all_comparisons if c['group'] == 'FRANK Faithfulness (Qwen)']
        baseline = None
        enhanced = None
        if len(frank_qwen) >= 2:
            baseline = next((c for c in frank_qwen if 'Baseline' in c['variant']), None)
            enhanced = next((c for c in frank_qwen if 'Enhanced' in c['variant'] and 'SC' not in c['variant']), None)
            
            if baseline and enhanced and baseline['avg_sec_per_doc'] > 0:
                slowdown = enhanced['avg_sec_per_doc'] / baseline['avg_sec_per_doc']
                print(f"\nFRANK (Qwen) Enhanced Slowdown: {slowdown:.2f}x")
        
        # Compare self-consistency overhead
        frank_qwen_sc = next((c for c in frank_qwen if 'Enhanced+SC' in c['variant']), None)
        if frank_qwen_sc and enhanced and enhanced['avg_sec_per_doc'] > 0:
            sc_slowdown = frank_qwen_sc['avg_sec_per_doc'] / enhanced['avg_sec_per_doc']
            print(f"FRANK (Qwen) SC Overhead: {sc_slowdown:.2f}x")

        realsumm_sc = [c for c in all_comparisons if c['group'] == 'REALSumm Keyfact SC (Qwen)']
        if realsumm_sc:
            print("\nREALSumm Qwen SC timings:")
            for row in realsumm_sc:
                print(f"  - {row['variant']}: {row['avg_sec_per_doc']:.4f} sec/doc")

        cross_domain_cc = [
            c for c in all_comparisons if c['group'] == 'Cross-Domain Completeness/Conciseness (Qwen)'
        ]
        if cross_domain_cc:
            print("\nCross-domain completeness/conciseness timings:")
            for row in cross_domain_cc:
                print(f"  - {row['variant']}: {row['avg_sec_per_doc']:.4f} sec/doc")
    
    print("\n" + "="*100)
    print("## HOW TO BACKFILL MISSING RUNTIME")
    print("="*100)
    print("""
1. Check current completion:
    python check_rq_completion.py
   
2. If a row shows ⚠️ NO_RUNTIME, regenerate that output folder with runtime tracking enabled:
    sbatch run_missing_rq2_enhanced_sc.sh
    sbatch run_missing_rq2_keyfact.sh
    sbatch run_missing_rq3_machine_keyfact.sh
   
3. Monitor jobs:
    squeue -u $USER
    tail -f slurm-JOBID.out
   
4. After jobs complete, aggregate results:
    python analyze_rq4_efficiency.py
    python check_rq_completion.py
""")

if __name__ == "__main__":
    main()

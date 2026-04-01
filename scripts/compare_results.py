#!/usr/bin/env python3
"""Statistical comparison of model performance.

Performs statistical tests to compare baseline models with attention-enhanced models.
Includes paired t-tests, ANOVA, and effect size calculations.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Statistical comparison of model performance"
    )
    
    parser.add_argument(
        "--metrics_file",
        type=str,
        default="results/metrics/all_metrics.json",
        help="Path to metrics JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/metrics",
        help="Directory to save statistical report",
    )
    
    return parser.parse_args()


def load_metrics(metrics_path: Path):
    """Load metrics from JSON file."""
    with open(metrics_path, "r") as f:
        return json.load(f)


def paired_t_test(baseline_metrics, attention_metrics, metric_name):
    """Perform paired t-test between baseline and attention models.
    
    Args:
        baseline_metrics: List of baseline model metrics
        attention_metrics: List of attention model metrics
        metric_name: Name of metric to compare
        
    Returns:
        t-statistic, p-value, and significance flag
    """
    baseline_values = [m.get(metric_name, 0) for m in baseline_metrics]
    attention_values = [m.get(metric_name, 0) for m in attention_metrics]
    
    if len(baseline_values) < 2 or len(attention_values) < 2:
        return None, None, None
    
    t_stat, p_value = stats.ttest_ind(baseline_values, attention_values)
    significant = p_value < 0.05
    
    return t_stat, p_value, significant


def calculate_effect_size(baseline_values, attention_values):
    """Calculate Cohen's d effect size.
    
    Args:
        baseline_values: List of baseline metric values
        attention_values: List of attention metric values
        
    Returns:
        Cohen's d value
    """
    n1, n2 = len(baseline_values), len(attention_values)
    if n1 < 2 or n2 < 2:
        return None
    
    mean1, mean2 = np.mean(baseline_values), np.mean(attention_values)
    std1, std2 = np.std(baseline_values, ddof=1), np.std(attention_values, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return None
    
    cohens_d = (mean2 - mean1) / pooled_std
    
    return cohens_d


def interpret_effect_size(d):
    """Interpret Cohen's d effect size."""
    if d is None:
        return "N/A"
    elif abs(d) < 0.2:
        return "negligible"
    elif abs(d) < 0.5:
        return "small"
    elif abs(d) < 0.8:
        return "medium"
    else:
        return "large"


def anova_test(*groups):
    """Perform one-way ANOVA test.
    
    Args:
        *groups: Variable number of arrays/lists of values
        
    Returns:
        F-statistic, p-value
    """
    valid_groups = [g for g in groups if len(g) >= 2]
    if len(valid_groups) < 2:
        return None, None
    
    f_stat, p_value = stats.f_oneway(*valid_groups)
    return f_stat, p_value


def main():
    """Main comparison function."""
    args = parse_args()
    
    # Load metrics
    metrics_path = Path(args.metrics_file)
    if not metrics_path.exists():
        print(f"Metrics file not found: {metrics_path}")
        print("Please run scripts/evaluate_models.py first")
        return
    
    metrics = load_metrics(metrics_path)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group models by backbone and attention type
    resnet_baseline = [m for m in metrics if "resnet" in m["model_name"] and "baseline" in m["model_name"]]
    resnet_se = [m for m in metrics if "resnet" in m["model_name"] and "se" in m["model_name"]]
    resnet_cbam = [m for m in metrics if "resnet" in m["model_name"] and "cbam" in m["model_name"]]
    resnet_eca = [m for m in metrics if "resnet" in m["model_name"] and "eca" in m["model_name"]]
    
    densenet_baseline = [m for m in metrics if "densenet" in m["model_name"] and "baseline" in m["model_name"]]
    densenet_se = [m for m in metrics if "densenet" in m["model_name"] and "se" in m["model_name"]]
    densenet_cbam = [m for m in metrics if "densenet" in m["model_name"] and "cbam" in m["model_name"]]
    
    # Metrics to compare
    key_metrics = [
        "accuracy",
        "tb_sensitivity",
        "tb_precision",
        "tb_f1",
        "tb_specificity",
        "f1_macro",
        "roc_auc_macro",
    ]
    
    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append("STATISTICAL COMPARISON REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Section 1: ResNet Comparisons
    report_lines.append("-" * 80)
    report_lines.append("1. RESNET50 COMPARISONS")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    comparisons = [
        ("Baseline", "SE", resnet_baseline, resnet_se),
        ("Baseline", "CBAM", resnet_baseline, resnet_cbam),
        ("Baseline", "ECA", resnet_baseline, resnet_eca),
    ]
    
    for base_name, attn_name, base_group, attn_group in comparisons:
        if not base_group or not attn_group:
            continue
            
        report_lines.append(f"ResNet50 {base_name} vs ResNet50 + {attn_name}")
        report_lines.append("-" * 40)
        
        for metric_name in key_metrics:
            base_values = [m.get(metric_name, 0) for m in base_group]
            attn_values = [m.get(metric_name, 0) for m in attn_group]
            
            if len(base_values) < 1 or len(attn_values) < 1:
                continue
            
            mean_base = np.mean(base_values)
            mean_attn = np.mean(attn_values)
            std_base = np.std(base_values) if len(base_values) > 1 else 0
            std_attn = np.std(attn_values) if len(attn_values) > 1 else 0
            
            # T-test
            t_stat, p_val, significant = paired_t_test(base_group, attn_group, metric_name)
            
            # Effect size
            cohens_d = calculate_effect_size(base_values, attn_values)
            effect_interp = interpret_effect_size(cohens_d)
            
            # Format output
            sig_marker = "*" if significant else ""
            report_lines.append(
                f"  {metric_name:20s}: {mean_base:.4f}±{std_base:.4f} → {mean_attn:.4f}±{std_attn:.4f} "
                f"(Δ={mean_attn-mean_base:+.4f}{sig_marker}, p={p_val:.4f if p_val else 'N/A'}, d={cohens_d:.2f} [{effect_interp}])"
            )
        
        report_lines.append("")
    
    # Section 2: DenseNet Comparisons
    report_lines.append("-" * 80)
    report_lines.append("2. DENSENET121 COMPARISONS")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    densenet_comparisons = [
        ("Baseline", "SE", densenet_baseline, densenet_se),
        ("Baseline", "CBAM", densenet_baseline, densenet_cbam),
    ]
    
    for base_name, attn_name, base_group, attn_group in densenet_comparisons:
        if not base_group or not attn_group:
            continue
            
        report_lines.append(f"DenseNet121 {base_name} vs DenseNet121 + {attn_name}")
        report_lines.append("-" * 40)
        
        for metric_name in key_metrics:
            base_values = [m.get(metric_name, 0) for m in base_group]
            attn_values = [m.get(metric_name, 0) for m in attn_group]
            
            if len(base_values) < 1 or len(attn_values) < 1:
                continue
            
            mean_base = np.mean(base_values)
            mean_attn = np.mean(attn_values)
            std_base = np.std(base_values) if len(base_values) > 1 else 0
            std_attn = np.std(attn_values) if len(attn_values) > 1 else 0
            
            # T-test
            t_stat, p_val, significant = paired_t_test(base_group, attn_group, metric_name)
            
            # Effect size
            cohens_d = calculate_effect_size(base_values, attn_values)
            effect_interp = interpret_effect_size(cohens_d)
            
            sig_marker = "*" if significant else ""
            report_lines.append(
                f"  {metric_name:20s}: {mean_base:.4f}±{std_base:.4f} → {mean_attn:.4f}±{std_attn:.4f} "
                f"(Δ={mean_attn-mean_base:+.4f}{sig_marker}, p={p_val:.4f if p_val else 'N/A'}, d={cohens_d:.2f} [{effect_interp}])"
            )
        
        report_lines.append("")
    
    # Section 3: ANOVA across all models
    report_lines.append("-" * 80)
    report_lines.append("3. ANOVA: ALL MODELS COMPARISON")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    all_resnet_models = resnet_baseline + resnet_se + resnet_cbam + resnet_eca
    all_densenet_models = densenet_baseline + densenet_se + densenet_cbam
    
    if len(all_resnet_models) >= 4:
        report_lines.append("ResNet Variants (ANOVA):")
        for metric_name in key_metrics[:3]:  # Top 3 metrics
            values = [
                [m.get(metric_name, 0) for m in resnet_baseline],
                [m.get(metric_name, 0) for m in resnet_se],
                [m.get(metric_name, 0) for m in resnet_cbam],
                [m.get(metric_name, 0) for m in resnet_eca],
            ]
            f_stat, p_val = anova_test(*values)
            if p_val is not None:
                sig_marker = "*" if p_val < 0.05 else ""
                report_lines.append(f"  {metric_name:20s}: F={f_stat:.2f}, p={p_val:.4f}{sig_marker}")
        report_lines.append("")
    
    if len(all_densenet_models) >= 3:
        report_lines.append("DenseNet Variants (ANOVA):")
        for metric_name in key_metrics[:3]:
            values = [
                [m.get(metric_name, 0) for m in densenet_baseline],
                [m.get(metric_name, 0) for m in densenet_se],
                [m.get(metric_name, 0) for m in densenet_cbam],
            ]
            f_stat, p_val = anova_test(*values)
            if p_val is not None:
                sig_marker = "*" if p_val < 0.05 else ""
                report_lines.append(f"  {metric_name:20s}: F={f_stat:.2f}, p={p_val:.4f}{sig_marker}")
        report_lines.append("")
    
    # Section 4: Key Findings
    report_lines.append("-" * 80)
    report_lines.append("4. KEY FINDINGS")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    # Find best TB sensitivity
    all_models = [m for m in metrics if "model_name" in m]
    if all_models:
        best_tb = max(all_models, key=lambda m: m.get("tb_sensitivity", 0))
        report_lines.append(f"Best TB Sensitivity: {best_tb['model_name']} ({best_tb.get('tb_sensitivity', 0):.4f})")
        
        # Calculate improvement over baseline
        resnet_base = next((m for m in metrics if "resnet_baseline" in m["model_name"]), None)
        if resnet_base:
            improvement = best_tb.get("tb_sensitivity", 0) - resnet_base.get("tb_sensitivity", 0)
            report_lines.append(f"Improvement over ResNet Baseline: +{improvement:.4f} ({improvement/resnet_base.get('tb_sensitivity', 1)*100:+.1f}%)")
    
    report_lines.append("")
    report_lines.append("Significance levels: * p < 0.05")
    report_lines.append("Effect size interpretation: negligible (<0.2), small (<0.5), medium (<0.8), large (≥0.8)")
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # Save report
    report_text = "\n".join(report_lines)
    
    report_path = output_dir / f"statistical_report_{timestamp}.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    
    latest_path = output_dir / "statistical_report.txt"
    with open(latest_path, "w") as f:
        f.write(report_text)
    
    # Print report
    print(report_text)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()

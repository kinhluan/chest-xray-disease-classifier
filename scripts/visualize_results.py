#!/usr/bin/env python3
"""Visualize model comparison results.

Creates publication-quality figures comparing model performance.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize model comparison results"
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
        default="results/figures",
        help="Directory to save figures",
    )
    
    return parser.parse_args()


def load_metrics(metrics_path: Path):
    """Load metrics from JSON file."""
    with open(metrics_path, "r") as f:
        return json.load(f)


def plot_tb_sensitivity_comparison(metrics, output_path: Path):
    """Plot TB sensitivity comparison across all models."""
    plt.style.use("seaborn-v0_8-whitegrid")
    
    df = pd.DataFrame(metrics)
    
    # Sort by TB sensitivity
    df = df.sort_values("tb_sensitivity", ascending=True)
    
    # Create color mapping
    colors = []
    for name in df["model_name"]:
        if "baseline" in name:
            colors.append("#95a5a6")  # Gray for baseline
        elif "cbam" in name:
            colors.append("#e74c3c")  # Red for CBAM (best)
        elif "se" in name:
            colors.append("#3498db")  # Blue for SE
        elif "eca" in name:
            colors.append("#2ecc71")  # Green for ECA
        else:
            colors.append("#95a5a6")
    
    plt.figure(figsize=(12, 8))
    
    # Horizontal bar chart
    bars = plt.barh(df["model_name"], df["tb_sensitivity"], color=colors, alpha=0.8)
    
    # Add value labels
    for i, (idx, row) in enumerate(df.iterrows()):
        plt.text(
            row["tb_sensitivity"] + 0.01,
            i,
            f"{row['tb_sensitivity']:.4f}",
            va="center",
            fontsize=10,
        )
    
    plt.xlabel("TB Sensitivity (Recall)", fontsize=12)
    plt.ylabel("Model", fontsize=12)
    plt.title("Tuberculosis Sensitivity Comparison Across Models", fontsize=14, fontweight="bold")
    plt.xlim(0, 1.0)
    plt.gca().invert_yaxis()
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#95a5a6", label="Baseline"),
        Patch(facecolor="#e74c3c", label="CBAM"),
        Patch(facecolor="#3498db", label="SE"),
        Patch(facecolor="#2ecc71", label="ECA"),
    ]
    plt.legend(handles=legend_elements, loc="lower right")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


def plot_metrics_radar_chart(metrics, output_path: Path):
    """Plot radar chart comparing key metrics for selected models."""
    plt.style.use("seaborn-v0_8-whitegrid")
    
    df = pd.DataFrame(metrics)
    
    # Select representative models
    selected_models = []
    for pattern in ["resnet50_baseline", "resnet50_cbam", "densenet121_baseline", "densenet121_cbam"]:
        match = df[df["model_name"].str.contains(pattern)]
        if not match.empty:
            selected_models.append(match.iloc[0])
    
    if not selected_models:
        return
    
    # Metrics to plot
    metric_names = ["accuracy", "tb_sensitivity", "f1_macro", "precision_macro", "recall_macro"]
    
    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    colors = ["#95a5a6", "#e74c3c", "#3498db", "#2ecc71"]
    
    for i, model in enumerate(selected_models):
        values = [model.get(m, 0) for m in metric_names]
        values += values[:1]
        
        ax.plot(angles, values, "o-", linewidth=2, label=model["model_name"], color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), metric_names)
    ax.set_rlabel_position(0)
    ax.set_ylim(0, 1)
    
    plt.title("Model Performance Comparison (Key Metrics)", fontsize=14, fontweight="bold", pad=20)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


def plot_confusion_matrices(metrics, output_path: Path):
    """Plot confusion matrices for all models."""
    plt.style.use("seaborn-v0_8-whitegrid")
    
    # Select models with confusion matrices
    models_with_cm = [m for m in metrics if "confusion_matrix" in m]
    
    if not models_with_cm:
        print("  ⚠ No confusion matrices found")
        return
    
    # Determine grid size
    n_models = len(models_with_cm)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    
    if n_models == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    class_names = ["Normal", "Pneumonia", "Tuberculosis", "COVID-19"]
    
    for idx, model in enumerate(models_with_cm):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        cm = np.array(model["confusion_matrix"])
        
        # Normalize
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar=False,
        )
        
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)
        ax.set_title(model["model_name"], fontsize=11, fontweight="bold")
    
    # Hide empty subplots
    for idx in range(n_models, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")
    
    plt.suptitle("Confusion Matrices Comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


def plot_params_vs_performance(metrics, output_path: Path):
    """Plot model parameters vs TB sensitivity."""
    plt.style.use("seaborn-v0_8-whitegrid")
    
    df = pd.DataFrame(metrics)
    
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    for idx, row in df.iterrows():
        if "cbam" in row["model_name"]:
            plt.scatter(row["params_millions"], row["tb_sensitivity"], s=150, c="#e74c3c", marker="o", label="CBAM" if "CBAM" not in plt.gca().get_legend_handles_labels()[1] else "")
        elif "se" in row["model_name"]:
            plt.scatter(row["params_millions"], row["tb_sensitivity"], s=150, c="#3498db", marker="o", label="SE" if "SE" not in plt.gca().get_legend_handles_labels()[1] else "")
        elif "eca" in row["model_name"]:
            plt.scatter(row["params_millions"], row["tb_sensitivity"], s=150, c="#2ecc71", marker="o", label="ECA" if "ECA" not in plt.gca().get_legend_handles_labels()[1] else "")
        else:
            plt.scatter(row["params_millions"], row["tb_sensitivity"], s=150, c="#95a5a6", marker="o", label="Baseline" if "Baseline" not in plt.gca().get_legend_handles_labels()[1] else "")
        
        # Add model name
        plt.annotate(
            row["model_name"],
            (row["params_millions"], row["tb_sensitivity"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )
    
    plt.xlabel("Parameters (Millions)", fontsize=12)
    plt.ylabel("TB Sensitivity", fontsize=12)
    plt.title("Model Complexity vs TB Detection Performance", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


def main():
    """Main visualization function."""
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
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\nGenerating visualizations...")
    print("-" * 50)
    
    # Generate plots
    plot_tb_sensitivity_comparison(
        metrics,
        output_dir / f"tb_sensitivity_comparison_{timestamp}.png",
    )
    plot_tb_sensitivity_comparison(metrics, output_dir / "tb_sensitivity_comparison.png")
    
    plot_metrics_radar_chart(
        metrics,
        output_dir / f"radar_chart_metrics_{timestamp}.png",
    )
    plot_metrics_radar_chart(metrics, output_dir / "radar_chart_metrics.png")
    
    plot_confusion_matrices(
        metrics,
        output_dir / f"confusion_matrices_{timestamp}.png",
    )
    plot_confusion_matrices(metrics, output_dir / "confusion_matrices.png")
    
    plot_params_vs_performance(
        metrics,
        output_dir / f"params_vs_performance_{timestamp}.png",
    )
    plot_params_vs_performance(metrics, output_dir / "params_vs_performance.png")
    
    print("-" * 50)
    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()

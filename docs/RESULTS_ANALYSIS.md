---
tags: [results, analysis, attention, chest-xray]
status: template
created: 2026-04-01
author: Luân B.
---

# Attention-Enhanced CNN Results Analysis

## 📊 Executive Summary

**Study Period:** [Start Date] - [End Date]

**Objective:** Evaluate the impact of attention mechanisms (SE, CBAM, ECA) on chest X-ray disease classification, with focus on improving Tuberculosis (TB) sensitivity.

**Key Finding:** [TODO: Add main finding after training completes]

---

## 🎯 Study Overview

### Motivation

Tuberculosis (TB) remains a critical global health challenge, with early detection being crucial for effective treatment. Standard CNN models may miss subtle features in X-ray images that are indicative of TB. This study investigates whether attention mechanisms can improve TB sensitivity without compromising overall accuracy.

### Hypotheses

| ID | Hypothesis | Status |
|----|------------|--------|
| H1 | Attention models > Baseline (+3-5% accuracy) | [ ] Confirmed [ ] Refuted |
| H2 | CBAM > SE > ECA for TB detection | [ ] Confirmed [ ] Refuted |
| H3 | Attention improves TB sensitivity (+5-10% recall) | [ ] Confirmed [ ] Refuted |
| H4 | DenseNet+Attention > ResNet+Attention | [ ] Confirmed [ ] Refuted |
| H5 | Computational overhead acceptable (<10% slower) | [ ] Confirmed [ ] Refuted |

---

## 📋 Methods

### Models Evaluated

| Model ID | Backbone | Attention | Parameters | Training Time |
|----------|----------|-----------|------------|---------------|
| M1 | ResNet50 | None (Baseline) | 25.6M | [TODO] |
| M2 | ResNet50 | SE-Block | 28.1M | [TODO] |
| M3 | ResNet50 | CBAM | 28.5M | [TODO] |
| M4 | ResNet50 | ECA-Net | 26.2M | [TODO] |
| M5 | DenseNet121 | None (Baseline) | 8.0M | [TODO] |
| M6 | DenseNet121 | SE-Block | 8.9M | [TODO] |
| M7 | DenseNet121 | CBAM | 9.2M | [TODO] |

### Dataset

- **Source:** Kaggle - Chest X-Ray (Pneumonia, Covid-19, Tuberculosis)
- **Classes:** Normal, Pneumonia, Tuberculosis, COVID-19
- **Train/Val/Test Split:** 65% / 20% / 15%
- **Total Images:** ~9,280

### Training Configuration

```yaml
epochs: 50
batch_size: 32
learning_rate: 1.0e-4
optimizer: AdamW
scheduler: CosineAnnealingLR
early_stopping: patience=10
```

### Evaluation Metrics

- **Primary:** TB Sensitivity (Recall for Tuberculosis class)
- **Secondary:** Overall Accuracy, Macro F1-Score, TB F1-Score
- **Efficiency:** Parameters (M), Inference Time (ms/img)

---

## 📈 Results

### Overall Performance

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro) | TB Sensitivity | Params (M) |
|-------|----------|-------------------|----------------|------------|----------------|------------|
| ResNet50 Baseline | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | 25.6 |
| ResNet50 + SE | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | 28.1 |
| **ResNet50 + CBAM** | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** | 28.5 |
| ResNet50 + ECA | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | 26.2 |
| DenseNet121 Baseline | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | 8.0 |
| DenseNet121 + SE | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | 8.9 |
| **DenseNet121 + CBAM** | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** | 9.2 |

**Best values in each category are bolded.**

### TB-Specific Metrics

| Model | TB Sensitivity | TB Precision | TB F1-Score | TB Specificity |
|-------|----------------|--------------|-------------|----------------|
| ResNet50 Baseline | [TODO] | [TODO] | [TODO] | [TODO] |
| ResNet50 + SE | [TODO] | [TODO] | [TODO] | [TODO] |
| **ResNet50 + CBAM** | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** |
| ResNet50 + ECA | [TODO] | [TODO] | [TODO] | [TODO] |
| DenseNet121 Baseline | [TODO] | [TODO] | [TODO] | [TODO] |
| DenseNet121 + SE | [TODO] | [TODO] | [TODO] | [TODO] |
| **DenseNet121 + CBAM** | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** |

---

## 📊 Statistical Analysis

### ResNet50: Baseline vs Attention

| Metric | Baseline | +SE | +CBAM | +ECA | p-value (ANOVA) |
|--------|----------|-----|-------|-----|-----------------|
| Accuracy | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| TB Sensitivity | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| F1-Macro | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |

**Paired t-test (Baseline vs CBAM):**
- TB Sensitivity: t = [TODO], p = [TODO], Cohen's d = [TODO]
- Accuracy: t = [TODO], p = [TODO], Cohen's d = [TODO]

### DenseNet121: Baseline vs Attention

| Metric | Baseline | +SE | +CBAM | p-value (ANOVA) |
|--------|----------|-----|-------|-----------------|
| Accuracy | [TODO] | [TODO] | [TODO] | [TODO] |
| TB Sensitivity | [TODO] | [TODO] | [TODO] | [TODO] |
| F1-Macro | [TODO] | [TODO] | [TODO] | [TODO] |

**Paired t-test (Baseline vs CBAM):**
- TB Sensitivity: t = [TODO], p = [TODO], Cohen's d = [TODO]

---

## 🔍 Key Findings

### 1. TB Sensitivity Improvement

**Finding:** [TODO: Describe TB sensitivity improvement]

- Best model: [TODO] with TB sensitivity of [TODO]
- Improvement over baseline: +[TODO]% ([TODO] → [TODO])
- Statistical significance: p < [TODO]

### 2. Attention Mechanism Comparison

**Ranking:** [TODO: CBAM > SE > ECA or other]

| Attention | Avg TB Sensitivity Gain | Overhead |
|-----------|------------------------|----------|
| CBAM | +[TODO]% | +[TODO]% params |
| SE | +[TODO]% | +[TODO]% params |
| ECA | +[TODO]% | +[TODO]% params |

### 3. Backbone Comparison

**ResNet vs DenseNet:**

- ResNet50 + CBAM: TB Sensitivity = [TODO]
- DenseNet121 + CBAM: TB Sensitivity = [TODO]
- Winner: [TODO] (difference: [TODO]%)

### 4. Efficiency Analysis

| Model | Params (M) | Inference (ms) | TB Sensitivity | Efficiency Score* |
|-------|------------|----------------|----------------|-------------------|
| ResNet50 + CBAM | 28.5 | [TODO] | [TODO] | [TODO] |
| DenseNet121 + CBAM | 9.2 | [TODO] | [TODO] | [TODO] |

*Efficiency Score = TB Sensitivity / Params

---

## 🎨 Attention Visualization Analysis

### Grad-CAM Results

**Qualitative Observations:**

1. **Baseline Models:** [TODO: Describe attention patterns]

2. **Attention-Enhanced Models:** [TODO: Describe improvements]

3. **TB Cases:** [TODO: Specific observations for TB detection]

### Case Study: False Negatives Reduced

| Model | TB False Negatives | Reduction |
|-------|-------------------|-----------|
| ResNet50 Baseline | [TODO] | - |
| ResNet50 + CBAM | [TODO] | -[TODO]% |

---

## ✅ Hypothesis Validation

| Hypothesis | Result | Evidence |
|------------|--------|----------|
| H1: Attention > Baseline | [✓/✗] | [TODO: Brief explanation] |
| H2: CBAM > SE > ECA | [✓/✗] | [TODO] |
| H3: TB Sensitivity +5-10% | [✓/✗] | [TODO] |
| H4: DenseNet+Attn > ResNet+Attn | [✓/✗] | [TODO] |
| H5: Overhead <10% | [✓/✗] | [TODO] |

---

## 🎯 Best Model Selection

### Selection Criteria

- **TB Sensitivity (weight: 0.6)** - Priority metric
- **Overall Accuracy (weight: 0.3)**
- **Inference Speed (weight: 0.1)**

### Winner: [TODO: Model Name]

**Final Performance:**
- TB Sensitivity: [TODO]
- Overall Accuracy: [TODO]
- F1-Macro: [TODO]
- Parameters: [TODO]M
- Inference Time: [TODO]ms

**Rationale:** [TODO: Explain why this model was selected]

---

## 📝 Discussion

### Strengths

1. [TODO: Key strength 1]
2. [TODO: Key strength 2]
3. [TODO: Key strength 3]

### Limitations

1. [TODO: Limitation 1]
2. [TODO: Limitation 2]
3. [TODO: Limitation 3]

### Clinical Implications

[TODO: Discuss how improved TB sensitivity could impact clinical practice]

---

## 📚 Conclusions

### Main Contributions

1. **Implemented** three attention mechanisms (SE, CBAM, ECA) for chest X-ray classification
2. **Demonstrated** [TODO]% improvement in TB sensitivity with CBAM
3. **Achieved** statistical significance (p < 0.05) for attention benefits
4. **Validated** findings with Grad-CAM visualizations

### Future Work

- [ ] Test on external datasets for generalization
- [ ] Explore transformer-based attention (ViT, Swin)
- [ ] Multi-label classification for co-occurring diseases
- [ ] Real-time deployment optimization

---

## 📁 Generated Artifacts

### Code
- `src/classifier/models/attention.py` - Attention modules
- `src/classifier/models/attention_models.py` - Model wrappers
- `scripts/train_all_models.sh` - Training script
- `scripts/evaluate_models.py` - Evaluation pipeline
- `scripts/compare_results.py` - Statistical analysis
- `scripts/visualize_results.py` - Visualization tools
- `scripts/visualize_attention.py` - Grad-CAM implementation

### Results
- `results/models/*/best_model.pth` - 7 trained checkpoints
- `results/metrics/all_metrics.csv` - Comprehensive metrics
- `results/metrics/statistical_report.txt` - Statistical tests
- `results/figures/*.png` - All visualizations

### Documentation
- `docs/ATTENTION_DESIGN.md` - Architecture design
- `docs/ATTENTION_TODO.md` - Implementation plan
- `docs/RESULTS_ANALYSIS.md` - This document

---

## 📊 Figures

### Figure 1: TB Sensitivity Comparison

![TB Sensitivity Comparison](../results/figures/tb_sensitivity_comparison.png)

### Figure 2: Radar Chart - Key Metrics

![Radar Chart](../results/figures/radar_chart_metrics.png)

### Figure 3: Confusion Matrices

![Confusion Matrices](../results/figures/confusion_matrices.png)

### Figure 4: Parameters vs Performance

![Params vs Performance](../results/figures/params_vs_performance.png)

### Figure 5: Grad-CAM Attention Maps

![Grad-CAM](../results/figures/attention_maps/gradcam_*.png)

---

## 📅 Timeline

| Phase | Start Date | End Date | Status |
|-------|------------|----------|--------|
| Implementation | 2026-04-01 | 2026-04-01 | ✅ Complete |
| Training | [TODO] | [TODO] | [ ] In Progress |
| Evaluation | [TODO] | [TODO] | [ ] Pending |
| Analysis | [TODO] | [TODO] | [ ] Pending |

---

**Last Updated:** [TODO]
**Next Action:** [TODO: Complete training and fill in results]

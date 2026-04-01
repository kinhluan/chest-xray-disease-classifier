---
tags: [experiment, protocol, training, evaluation]
status: planned
created: 2026-03-31
related: [[ATTENTION_DESIGN]], [[ATTENTION_TODO]]
---

# Experiment Protocol - Attention-Enhanced CNN

## 🎯 Objective

Evaluate the impact of attention mechanisms (SE, CBAM, ECA) on CNN performance for chest X-ray classification, with focus on Tuberculosis sensitivity improvement.

---

## 📋 Experimental Design

### Independent Variables

| Variable | Levels |
|----------|--------|
| Backbone Architecture | ResNet50, DenseNet121 |
| Attention Module | None, SE, CBAM, ECA |
| **Total Combinations** | **7 models** |

### Dependent Variables

| Metric | Priority | Target |
|--------|----------|--------|
| TB Sensitivity (Recall) | ⭐⭐⭐ | > 0.85 |
| Overall Accuracy | ⭐⭐ | > 0.90 |
| Macro F1-Score | ⭐⭐ | > 0.88 |
| TB F1-Score | ⭐⭐⭐ | > 0.85 |
| Inference Time | ⭐ | < 50ms |

### Controlled Variables

```yaml
Dataset: Same train/val/test split for all models
Image Size: 224x224
Batch Size: 32
Learning Rate: 1e-4
Optimizer: AdamW
Epochs: 50 (with early stopping)
Data Augmentation: Same transforms
Class Weights: Applied to all
Random Seed: 42 (reproducible)
```

---

## 🔧 Setup Instructions

### 1. Environment Setup

```bash
# Navigate to project
cd chest-xray-disease-classifier

# Install dependencies
uv sync

# Verify GPU (optional)
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 2. Dataset Download

```bash
# Set Kaggle credentials
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

# Download and prepare dataset
./download_and_train.sh

# Verify structure
tree data/raw/ -L 2
# Expected:
# data/raw/
# ├── Normal/
# ├── Pneumonia/
# ├── Tuberculosis/
# └── COVID-19/
```

### 3. Verify Attention Modules

```bash
# Test attention module imports
uv run python -c "
from classifier.models.attention import SEBlock, CBAM, ECA
import torch

x = torch.randn(2, 64, 56, 56)

# Test SE
se = SEBlock(64)
y = se(x)
assert y.shape == x.shape, 'SE shape mismatch'

# Test CBAM
cbam = CBAM(64)
y = cbam(x)
assert y.shape == x.shape, 'CBAM shape mismatch'

# Test ECA
eca = ECA(64)
y = eca(x)
assert y.shape == x.shape, 'ECA shape mismatch'

print('✅ All attention modules working!')
"
```

---

## 📊 Training Protocol

### Step 1: Train Baseline Models

```bash
# ResNet50 Baseline
uv run python train.py \
  --data_dir data/raw \
  --model_type resnet \
  --model_name resnet50 \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-4 \
  --output_dir results/models/resnet50_baseline \
  --experiment_name resnet50_baseline

# DenseNet121 Baseline
uv run python train.py \
  --data_dir data/raw \
  --model_type densenet \
  --model_name densenet121 \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-4 \
  --output_dir results/models/densenet121_baseline \
  --experiment_name densenet121_baseline
```

### Step 2: Train Attention Models

```bash
# ResNet50 + SE
uv run python train.py \
  --data_dir data/raw \
  --model_type resnet \
  --model_name resnet50 \
  --attention se \
  --epochs 50 \
  --output_dir results/models/resnet50_se

# ResNet50 + CBAM
uv run python train.py \
  --data_dir data/raw \
  --model_type resnet \
  --model_name resnet50 \
  --attention cbam \
  --epochs 50 \
  --output_dir results/models/resnet50_cbam

# ResNet50 + ECA
uv run python train.py \
  --data_dir data/raw \
  --model_type resnet \
  --model_name resnet50 \
  --attention eca \
  --epochs 50 \
  --output_dir results/models/resnet50_eca

# DenseNet121 + SE
uv run python train.py \
  --data_dir data/raw \
  --model_type densenet \
  --model_name densenet121 \
  --attention se \
  --epochs 50 \
  --output_dir results/models/densenet121_se

# DenseNet121 + CBAM
uv run python train.py \
  --data_dir data/raw \
  --model_type densenet \
  --model_name densenet121 \
  --attention cbam \
  --epochs 50 \
  --output_dir results/models/densenet121_cbam
```

### Step 3: Monitor Training

```bash
# Watch training in real-time
tail -f results/models/*/training.log

# Or use TensorBoard (if configured)
tensorboard --logdir results/models/
```

---

## 📈 Evaluation Protocol

### Step 1: Evaluate All Models

```bash
uv run python scripts/evaluate_models.py \
  --models_dir results/models \
  --data_dir data/raw \
  --output_dir results/metrics
```

### Step 2: Generate Comparison Report

```bash
uv run python scripts/compare_results.py \
  --metrics_file results/metrics/all_metrics.csv \
  --output_dir results/metrics
```

### Step 3: Generate Visualizations

```bash
uv run python scripts/visualize_results.py \
  --metrics_file results/metrics/all_metrics.csv \
  --output_dir results/figures
```

### Step 4: Attention Visualization

```bash
uv run python scripts/visualize_attention.py \
  --model_path results/models/resnet50_cbam/best_model.pth \
  --data_dir data/raw \
  --output_dir results/figures/attention_maps
```

---

## 📋 Expected Outputs

### Directory Structure

```
results/
├── models/
│   ├── resnet50_baseline/
│   │   ├── best_model.pth
│   │   ├── latest_model.pth
│   │   ├── config.json
│   │   └── training_history.json
│   ├── resnet50_se/
│   ├── resnet50_cbam/
│   ├── resnet50_eca/
│   ├── densenet121_baseline/
│   ├── densenet121_se/
│   └── densenet121_cbam/
│
├── metrics/
│   ├── all_metrics.csv
│   ├── statistical_report.txt
│   └── per_class_metrics.json
│
└── figures/
│   ├── training_curves.png
│   ├── tb_sensitivity_comparison.png
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   ├── pr_curves.png
│   └── attention_maps/
│       ├── resnet50_cbam_gradcam.png
│       └── densenet121_cbam_gradcam.png
```

### Metrics CSV Format

```csv
model,accuracy,precision_macro,recall_macro,f1_macro,tb_sensitivity,tb_precision,tb_f1,params_m,model_size_mb,inference_ms
resnet50_baseline,0.89,0.87,0.85,0.86,0.78,0.82,0.80,25.6,98.5,32
resnet50_se,0.91,0.89,0.88,0.88,0.83,0.85,0.84,28.1,105.2,35
resnet50_cbam,0.92,0.90,0.89,0.89,0.86,0.87,0.86,28.5,108.1,38
...
```

---

## 🧪 Statistical Analysis Plan

### Primary Comparison

**Test:** Paired t-test (Baseline vs Attention)

```python
from scipy import stats

# Compare TB sensitivity
baseline_tb_sens = [0.78, 0.76, 0.79]  # From cross-validation
cbam_tb_sens = [0.86, 0.85, 0.87]

t_stat, p_value = stats.ttest_ind(cbam_tb_sens, baseline_tb_sens)
print(f"TB Sensitivity improvement: p = {p_value:.4f}")
# Expected: p < 0.05 (significant)
```

### Effect Size

```python
# Cohen's d
def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    s_pooled = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / s_pooled

d = cohens_d(cbam_tb_sens, baseline_tb_sens)
# Interpretation:
# d < 0.2: negligible
# 0.2-0.5: small
# 0.5-0.8: medium
# > 0.8: large
```

### Multiple Comparison Correction

```python
# Bonferroni correction for 7 models
alpha_corrected = 0.05 / 7  # = 0.0071
```

---

## ✅ Quality Checks

### During Training

- [ ] Loss decreasing (both train and val)
- [ ] No NaN/Inf in loss
- [ ] Validation accuracy improving
- [ ] No overfitting (train-val gap < 5%)
- [ ] GPU memory usage normal

### After Training

- [ ] Model checkpoint saved
- [ ] Metrics file complete
- [ ] Confusion matrix shows all classes
- [ ] No class has 0 predictions

### Before Deployment

- [ ] Best model selected based on TB sensitivity
- [ ] Inference time < 100ms
- [ ] Model size < 500MB
- [ ] HF Space tested with sample images

---

## 📌 Troubleshooting

### Issue: Out of Memory

```bash
# Reduce batch size
--batch_size 16

# Or use gradient accumulation
--accumulate_grad_batches 2
```

### Issue: TB Sensitivity Low

```bash
# Increase class weight for TB
--class_weights manual:1.0,1.0,2.0,1.0

# Or use focal loss
--loss focal --focal_gamma 2.0
```

### Issue: Training Too Slow

```bash
# Use mixed precision training
--use_amp true

# Reduce number of workers if CPU bottleneck
--num_workers 2
```

---

## 🔗 Related Documents

- [[ATTENTION_DESIGN]] - Architecture design and hypotheses
- [[ATTENTION_TODO]] - Detailed task list
- [[RESULTS_ANALYSIS]] - Results and conclusions (after completion)

---

**Protocol Version:** 1.0
**Last Updated:** 2026-03-31
**Approved By:** Luân B.

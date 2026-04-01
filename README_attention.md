# Attention-Enhanced CNN for Chest X-Ray Classification

## 🎯 Project Overview

This project implements and evaluates **attention mechanisms** (SE, CBAM, ECA) for chest X-ray disease classification, with a focus on improving **Tuberculosis (TB) sensitivity**.

---

## 📋 Quick Start

### 1. Setup

```bash
# Install dependencies
uv sync

# Download dataset
./download_and_train.sh
# OR
uv run python download_dataset.py
```

### 2. Train All Models

```bash
# Train all 7 models (ResNet50/DenseNet121 with SE/CBAM/ECA attention)
./scripts/train_all_models.sh
```

### 3. Evaluate Models

```bash
# Evaluate all trained models
uv run python scripts/evaluate_models.py

# Statistical comparison
uv run python scripts/compare_results.py

# Generate visualizations
uv run python scripts/visualize_results.py

# Grad-CAM attention maps
uv run python scripts/visualize_attention.py --checkpoint results/models/resnet50_cbam/best_model.pth
```

### 4. Explore Results

Open the Jupyter notebooks:
- `notebooks/01_data_exploration.ipynb` - Dataset analysis
- `notebooks/02_attention_comparison.ipynb` - Model comparison
- `notebooks/03_tb_sensitivity_analysis.ipynb` - TB detection deep dive

---

## 🏗️ Architecture

### Model Variants

| Model | Backbone | Attention | Parameters | Expected TB Sensitivity |
|-------|----------|-----------|------------|------------------------|
| ResNet50 Baseline | ResNet50 | None | 25.6M | 0.75-0.80 |
| ResNet50 + SE | ResNet50 | SE-Block | 28.1M | 0.80-0.85 |
| **ResNet50 + CBAM** | ResNet50 | **CBAM** | 28.5M | **0.82-0.87** |
| ResNet50 + ECA | ResNet50 | ECA-Net | 26.2M | 0.78-0.83 |
| DenseNet121 Baseline | DenseNet121 | None | 8.0M | 0.73-0.78 |
| DenseNet121 + SE | DenseNet121 | SE-Block | 8.9M | 0.78-0.83 |
| **DenseNet121 + CBAM** | DenseNet121 | **CBAM** | 9.2M | **0.80-0.85** |

### Attention Modules

1. **SE-Block (Squeeze-and-Excitation)**
   - Channel attention via global pooling + FC layers
   - Lightweight, effective for feature recalibration

2. **CBAM (Convolutional Block Attention Module)**
   - **Channel + Spatial attention** (recommended for TB detection)
   - Better feature localization

3. **ECA-Net (Efficient Channel Attention)**
   - 1D convolution for channel attention
   - Minimal computational overhead

---

## 📁 Project Structure

```
chest-xray-disease-classifier/
├── src/classifier/models/
│   ├── model.py              # Base ResNet/DenseNet models
│   ├── attention.py          # ⭐ SE, CBAM, ECA modules
│   └── attention_models.py   # ⭐ Models with attention
├── configs/
│   ├── attention_config.yaml    # Attention configurations
│   └── experiment_config.yaml   # Training configurations
├── scripts/
│   ├── train_all_models.sh      # Train all 7 models
│   ├── evaluate_models.py       # Evaluate on test set
│   ├── compare_results.py       # Statistical analysis
│   ├── visualize_results.py     # Performance visualizations
│   └── visualize_attention.py   # Grad-CAM maps
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_attention_comparison.ipynb
│   └── 03_tb_sensitivity_analysis.ipynb
├── docs/
│   ├── ATTENTION_DESIGN.md      # Architecture design
│   ├── ATTENTION_TODO.md        # Implementation plan
│   └── RESULTS_ANALYSIS.md      # Results template
├── train.py                  # Updated with attention support
└── README_attention.md       # This file
```

---

## 🧪 Training Configuration

### Default Hyperparameters

```yaml
epochs: 50
batch_size: 32
learning_rate: 1.0e-4
optimizer: AdamW
scheduler: CosineAnnealingLR
img_size: 224
class_weights: true  # Handle imbalanced data
```

### Single Model Training

```bash
# Train ResNet50 with CBAM attention
uv run python train.py \
  --data_dir data/raw \
  --model_type attention \
  --model_name resnet50 \
  --attention_type cbam \
  --epochs 50 \
  --output_dir results/models/resnet50_cbam
```

---

## 📊 Evaluation Metrics

### Primary Metrics

| Metric | Priority | Target |
|--------|----------|--------|
| **TB Sensitivity** | ⭐⭐⭐ | > 0.85 |
| Overall Accuracy | ⭐⭐ | > 0.90 |
| Macro F1-Score | ⭐⭐ | > 0.88 |
| TB F1-Score | ⭐⭐⭐ | > 0.85 |

### Secondary Metrics

- Precision per class
- Specificity per class
- ROC-AUC (One-vs-Rest)
- Model parameters (M)
- Inference time (ms/img)

---

## 📈 Expected Results

### Hypotheses

| ID | Hypothesis | Expected Outcome |
|----|------------|------------------|
| H1 | Attention models > Baseline | +3-5% accuracy |
| H2 | CBAM > SE > ECA | CBAM best for TB |
| H3 | Attention improves TB sensitivity | +5-10% recall for TB |
| H4 | DenseNet+Attention > ResNet+Attention | Better feature reuse |
| H5 | Overhead acceptable | <10% slower inference |

---

## 🔬 Analysis Tools

### 1. Statistical Comparison

```bash
uv run python scripts/compare_results.py
```

Generates:
- Paired t-tests (Baseline vs Attention)
- ANOVA (multiple models)
- Effect size (Cohen's d)
- Statistical report

### 2. Visualizations

```bash
uv run python scripts/visualize_results.py
```

Generates:
- TB sensitivity comparison bar chart
- Radar chart (key metrics)
- Confusion matrices heatmap
- Parameters vs performance scatter

### 3. Attention Maps (Grad-CAM)

```bash
uv run python scripts/visualize_attention.py \
  --checkpoint results/models/resnet50_cbam/best_model.pth \
  --class_name Tuberculosis
```

Shows where the model focuses attention for TB detection.

---

## 📝 Deliverables

### Code
- ✅ `src/classifier/models/attention.py`
- ✅ `src/classifier/models/attention_models.py`
- ✅ `scripts/train_all_models.sh`
- ✅ `scripts/evaluate_models.py`
- ✅ `scripts/compare_results.py`
- ✅ `scripts/visualize_results.py`
- ✅ `scripts/visualize_attention.py`

### Documentation
- ✅ `docs/ATTENTION_DESIGN.md`
- ✅ `docs/ATTENTION_TODO.md`
- ✅ `docs/RESULTS_ANALYSIS.md` (template)
- ✅ `README_attention.md` (this file)

### Notebooks
- ✅ `notebooks/01_data_exploration.ipynb`
- ✅ `notebooks/02_attention_comparison.ipynb`
- ✅ `notebooks/03_tb_sensitivity_analysis.ipynb`

### Results (After Training)
- ⏳ `results/models/*.pth` (7 checkpoints)
- ⏳ `results/metrics/all_metrics.csv`
- ⏳ `results/metrics/statistical_report.txt`
- ⏳ `results/figures/*.png`

---

## 🚀 Next Steps

1. **Train Models**
   ```bash
   ./scripts/train_all_models.sh
   ```

2. **Evaluate & Analyze**
   ```bash
   uv run python scripts/evaluate_models.py
   uv run python scripts/compare_results.py
   uv run python scripts/visualize_results.py
   ```

3. **Fill Results Analysis**
   - Open `docs/RESULTS_ANALYSIS.md`
   - Replace `[TODO]` placeholders with actual results

4. **Deploy Best Model**
   - Select best model based on TB sensitivity
   - Update Hugging Face Space
   - Commit to GitHub

---

## 📚 References

1. **SE-Net:** Hu et al. "Squeeze-and-Excitation Networks" (CVPR 2018)
2. **CBAM:** Woo et al. "CBAM: Convolutional Block Attention Module" (ECCV 2018)
3. **ECA-Net:** Wang et al. "ECA-Net: Efficient Channel Attention" (CVPR 2020)
4. **ResNet:** He et al. "Deep Residual Learning" (CVPR 2016)
5. **DenseNet:** Huang et al. "Densely Connected Convolutional Networks" (CVPR 2017)

---

## 📊 Dataset

**Source:** Kaggle - Chest X-Ray (Pneumonia, Covid-19, Tuberculosis)
**URL:** https://www.kaggle.com/datasets/jtiptj/chest-xray-pneumoniacovid19tuberculosis

| Class | Train | Validation | Test | Total |
|-------|-------|------------|------|-------|
| Normal | ~1,200 | ~350 | ~150 | ~1,700 |
| Pneumonia | ~2,000 | ~600 | ~250 | ~2,850 |
| Tuberculosis | ~1,500 | ~450 | ~200 | ~2,150 |
| COVID-19 | ~1,800 | ~550 | ~230 | ~2,580 |
| **Total** | **~6,500** | **~1,950** | **~830** | **~9,280** |

---

## 🎯 Success Criteria

- [ ] All 7 models trained successfully
- [ ] TB sensitivity > 0.85 for best attention model
- [ ] TB sensitivity improvement > 5% vs baseline
- [ ] Overall accuracy > 0.90
- [ ] Statistical significance p < 0.05
- [ ] Grad-CAM shows attention on lung regions
- [ ] Complete documentation

---

**Created:** 2026-04-01
**Author:** Luân B.
**Status:** Implementation Complete ✅ | Training Pending ⏳

---
tags: [todo, attention, implementation, training, analysis]
status: in_progress
created: 2026-03-31
priority: high
related: [[ATTENTION_DESIGN]]
---

# Attention-Enhanced CNN - TODO List

## 📋 Overview

**Objective:** Implement and evaluate attention mechanisms for chest X-ray classification with focus on TB sensitivity improvement.

**Status:** 🟡 In Progress

---

## 🎯 Phase 1: Implementation

### 1.1 Attention Modules

- [ ] **Create `src/classifier/models/attention.py`**
  - [ ] Implement `SEBlock` class
  - [ ] Implement `CBAM` class
  - [ ] Implement `ECA` class
  - [ ] Add unit tests for each module
  - [ ] Verify forward pass shapes

**Expected output:**
```python
# src/classifier/models/attention.py
class SEBlock(nn.Module): ...
class CBAM(nn.Module): ...
class ECA(nn.Module): ...
```

**Time estimate:** 2 hours

---

### 1.2 Model Integration

- [ ] **Create `src/classifier/models/attention_models.py`**
  - [ ] `ResNetWithAttention` wrapper class
  - [ ] `DenseNetWithAttention` wrapper class
  - [ ] Factory function `create_attention_model()`
  - [ ] Support for SE, CBAM, ECA
  - [ ] Pretrained weights loading

**Expected output:**
```python
# Model variants
model = create_attention_model(
    backbone='resnet50',
    attention='cbam',
    num_classes=4,
    pretrained=True
)
```

**Time estimate:** 2 hours

---

### 1.3 Training Pipeline Update

- [ ] **Update `train.py`**
  - [ ] Add `--attention` argument
  - [ ] Add `--attention_type` argument (se/cbam/eca)
  - [ ] Support attention model creation
  - [ ] Save attention config in checkpoint
  - [ ] Update config logging

**Time estimate:** 1 hour

---

### 1.4 Configuration Files

- [ ] **Create `configs/attention_config.yaml`**
  ```yaml
  attention:
    se:
      reduction: 16
    cbam:
      reduction: 16
      kernel_size: 7
    eca:
      gamma: 2
      b: 1
  
  models:
    - name: resnet50_baseline
      backbone: resnet50
      attention: null
      
    - name: resnet50_se
      backbone: resnet50
      attention: se
      
    - name: resnet50_cbam
      backbone: resnet50
      attention: cbam
      
    - name: resnet50_eca
      backbone: resnet50
      attention: eca
      
    - name: densenet121_baseline
      backbone: densenet121
      attention: null
      
    - name: densenet121_se
      backbone: densenet121
      attention: se
      
    - name: densenet121_cbam
      backbone: densenet121
      attention: cbam
  ```

- [ ] **Create `configs/experiment_config.yaml`**
  - [ ] Training hyperparameters
  - [ ] Data augmentation settings
  - [ ] Early stopping config
  - [ ] Random seeds

**Time estimate:** 30 minutes

---

## 📊 Phase 2: Training

### 2.1 Dataset Preparation

- [ ] **Setup Kaggle credentials**
  ```bash
  export KAGGLE_USERNAME=your_username
  export KAGGLE_KEY=your_api_key
  ```

- [ ] **Download dataset**
  ```bash
  ./download_and_train.sh
  ```

- [ ] **Verify dataset structure**
  ```bash
  ls -la data/raw/
  # Should show: Normal/, Pneumonia/, Tuberculosis/, COVID-19/
  ```

- [ ] **Create data split report**
  - [ ] Count images per class
  - [ ] Calculate class weights
  - [ ] Check for corrupted images

**Time estimate:** 1 hour

---

### 2.2 Model Training

- [ ] **Create `scripts/train_all_models.sh`**
  ```bash
  #!/bin/bash
  MODELS=(
    "resnet50_baseline"
    "resnet50_se"
    "resnet50_cbam"
    "resnet50_eca"
    "densenet121_baseline"
    "densenet121_se"
    "densenet121_cbam"
  )
  
  for model in "${MODELS[@]}"; do
    echo "Training $model..."
    uv run python train.py \
      --data_dir data/raw \
      --model_type attention \
      --attention_type ${model#resnet50_} \
      --epochs 50 \
      --output_dir results/models/$model
  done
  ```

- [ ] **Train Model 1: ResNet50 Baseline**
  - [ ] Run training
  - [ ] Monitor loss curves
  - [ ] Save checkpoint
  - [ ] Log training time

- [ ] **Train Model 2: ResNet50 + SE**
  - [ ] Run training
  - [ ] Monitor loss curves
  - [ ] Save checkpoint

- [ ] **Train Model 3: ResNet50 + CBAM**
  - [ ] Run training
  - [ ] Monitor loss curves
  - [ ] Save checkpoint

- [ ] **Train Model 4: ResNet50 + ECA**
  - [ ] Run training
  - [ ] Monitor loss curves
  - [ ] Save checkpoint

- [ ] **Train Model 5: DenseNet121 Baseline**
  - [ ] Run training
  - [ ] Monitor loss curves
  - [ ] Save checkpoint

- [ ] **Train Model 6: DenseNet121 + SE**
  - [ ] Run training
  - [ ] Monitor loss curves
  - [ ] Save checkpoint

- [ ] **Train Model 7: DenseNet121 + CBAM**
  - [ ] Run training
  - [ ] Monitor loss curves
  - [ ] Save checkpoint

**Time estimate:** 6-8 hours (GPU), 20+ hours (CPU)

---

## 📈 Phase 3: Evaluation

### 3.1 Metrics Calculation

- [ ] **Create `scripts/evaluate_models.py`**
  - [ ] Load all trained models
  - [ ] Evaluate on test set
  - [ ] Calculate all metrics per model
  - [ ] Save to `results/metrics/all_metrics.csv`

**Metrics to calculate:**
```python
metrics = {
    'accuracy': ...,
    'precision_macro': ...,
    'recall_macro': ...,
    'f1_macro': ...,
    'tb_sensitivity': ...,  # Priority!
    'tb_precision': ...,
    'tb_f1': ...,
    'tb_specificity': ...,
    'params_millions': ...,
    'model_size_mb': ...,
    'inference_ms': ...
}
```

**Time estimate:** 1 hour

---

### 3.2 Statistical Analysis

- [ ] **Create `scripts/compare_results.py`**
  - [ ] Load all metrics
  - [ ] Paired t-test (Baseline vs Attention)
  - [ ] ANOVA (multiple models)
  - [ ] Post-hoc Tukey HSD
  - [ ] Effect size (Cohen's d)
  - [ ] Save statistical report

**Expected output:**
```
results/metrics/statistical_report.txt

=== Statistical Comparison ===

ResNet50 Baseline vs ResNet50-CBAM:
- Accuracy: p = 0.023* (significant)
- TB Sensitivity: p = 0.008** (significant)
- Effect size (d): 0.65 (medium-large)

...
```

**Time estimate:** 1 hour

---

### 3.3 Visualization

- [ ] **Create `scripts/visualize_results.py`**
  - [ ] Training curves comparison
  - [ ] TB sensitivity bar chart
  - [ ] Confusion matrices heatmap
  - [ ] ROC curves overlay
  - [ ] PR curves overlay
  - [ ] Save to `results/figures/`

- [ ] **Create `scripts/visualize_attention.py`**
  - [ ] Grad-CAM implementation
  - [ ] Attention maps for each model
  - [ ] Overlay on X-ray images
  - [ ] Compare attention focus (TB cases)
  - [ ] Save visualizations

**Time estimate:** 3 hours

---

## 📝 Phase 4: Documentation

### 4.1 Results Documentation

- [ ] **Create `docs/RESULTS_ANALYSIS.md`**
  - [ ] Executive summary
  - [ ] Performance tables
  - [ ] Statistical test results
  - [ ] TB sensitivity analysis
  - [ ] Attention visualization analysis
  - [ ] Conclusions

**Template:**
```markdown
## Results Summary

### Overall Performance
| Model | Accuracy | TB Sensitivity | F1-Macro | Params |
|-------|----------|----------------|----------|--------|
| ... | ... | ... | ... | ... |

### Key Findings
1. CBAM improved TB sensitivity by X%
2. ...

### Statistical Significance
- ResNet-CBAM vs ResNet: p < 0.05
- ...
```

**Time estimate:** 2 hours

---

### 4.2 README Update

- [ ] **Create `README_attention.md`**
  - [ ] Study overview
  - [ ] Methods description
  - [ ] Results summary
  - [ ] How to reproduce
  - [ ] Citation info

**Time estimate:** 1 hour

---

### 4.3 Notebook Creation

- [ ] **Create `notebooks/01_data_exploration.ipynb`**
  - [ ] Dataset statistics
  - [ ] Class distribution
  - [ ] Sample images visualization

- [ ] **Create `notebooks/02_attention_comparison.ipynb`**
  - [ ] Load all results
  - [ ] Interactive comparisons
  - [ ] Statistical tests

- [ ] **Create `notebooks/03_tb_sensitivity_analysis.ipynb`**
  - [ ] Deep dive into TB class
  - [ ] Confusion matrix analysis
  - [ ] False negative analysis
  - [ ] Attention map visualization

**Time estimate:** 3 hours

---

## 🚀 Phase 5: Deployment

### 5.1 Best Model Selection

- [ ] **Select best model based on:**
  - [ ] TB sensitivity (priority)
  - [ ] Overall accuracy
  - [ ] Inference speed
  - [ ] Model size

**Selection criteria:**
```python
best_model = max(
    models, 
    key=lambda m: 0.6*m['tb_sensitivity'] + 0.3*m['accuracy'] + 0.1*m['speed_score']
)
```

**Time estimate:** 30 minutes

---

### 5.2 Hugging Face Space Update

- [ ] **Upload best model to HF Space**
  ```bash
  hf upload space kinhluan/chest-xray-disease-classifier \
    results/models/best_model.pth
  ```

- [ ] **Update `streamlit_app.py`**
  - [ ] Load new attention model
  - [ ] Display model info (with attention type)
  - [ ] Show confidence scores
  - [ ] Add attention visualization (Grad-CAM)

- [ ] **Update Space README**
  - [ ] Model architecture info
  - [ ] Performance metrics
  - [ ] TB sensitivity highlight

**Time estimate:** 2 hours

---

### 5.3 Final Push

- [ ] **Commit all code to GitHub**
  ```bash
  git add -A
  git commit -m "Add attention-enhanced CNN models for chest X-ray classification
  
  - Implemented SE, CBAM, ECA attention modules
  - Trained 7 model variants (ResNet/DenseNet backbones)
  - CBAM improved TB sensitivity by X%
  - Best model: DenseNet121-CBAM (TB sens: 0.XX)
  "
  git push origin master
  ```

- [ ] **Verify GitHub sync**
- [ ] **Verify HF Space running**

**Time estimate:** 30 minutes

---

## 📊 Deliverables Checklist

### Code
- [ ] `src/classifier/models/attention.py`
- [ ] `src/classifier/models/attention_models.py`
- [ ] `scripts/train_all_models.sh`
- [ ] `scripts/evaluate_models.py`
- [ ] `scripts/compare_results.py`
- [ ] `scripts/visualize_results.py`
- [ ] `scripts/visualize_attention.py`

### Documentation
- [ ] `docs/ATTENTION_DESIGN.md` ✅ (this file)
- [ ] `docs/ATTENTION_TODO.md` ✅ (current file)
- [ ] `docs/RESULTS_ANALYSIS.md`
- [ ] `README_attention.md`
- [ ] `notebooks/01_data_exploration.ipynb`
- [ ] `notebooks/02_attention_comparison.ipynb`
- [ ] `notebooks/03_tb_sensitivity_analysis.ipynb`

### Results
- [ ] `results/models/*.pth` (7 checkpoints)
- [ ] `results/metrics/all_metrics.csv`
- [ ] `results/metrics/statistical_report.txt`
- [ ] `results/figures/*.png` (all visualizations)

### Deployment
- [ ] HF Space updated with best model
- [ ] GitHub repository synced
- [ ] Demo working with attention model

---

## ⏱️ Total Time Estimate

| Phase | Tasks | Time |
|-------|-------|------|
| 1. Implementation | 4 tasks | 5.5 hours |
| 2. Training | 8 tasks | 8-10 hours |
| 3. Evaluation | 3 tasks | 5 hours |
| 4. Documentation | 3 tasks | 6 hours |
| 5. Deployment | 3 tasks | 3 hours |
| **TOTAL** | **21 tasks** | **27-29 hours** |

---

## 🎯 Success Metrics

- [ ] All 7 models trained successfully
- [ ] TB sensitivity > 0.85 for best attention model
- [ ] TB sensitivity improvement > 5% vs baseline
- [ ] Overall accuracy > 0.90
- [ ] Statistical significance p < 0.05
- [ ] Grad-CAM shows attention on lung regions
- [ ] HF Space demo working
- [ ] Complete documentation

---

## 📌 Progress Tracking

```
Phase 1: Implementation    [          ] 0%
Phase 2: Training          [          ] 0%
Phase 3: Evaluation        [          ] 0%
Phase 4: Documentation     [          ] 0%
Phase 5: Deployment        [          ] 0%

Overall:                   [          ] 0%
```

---

**Last Updated:** 2026-03-31
**Next Action:** Start Phase 1.1 - Implement attention modules

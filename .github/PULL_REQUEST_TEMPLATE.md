---
name: Attention Mechanism Implementation
about: Template for attention mechanism PR
title: 'Feat: Attention-Enhanced CNN for Chest X-Ray Classification'
labels: 'enhancement, deep-learning'
assignees: ''
---

## 🎯 Objective

Implement attention mechanisms (SE, CBAM, ECA) to improve Tuberculosis (TB) sensitivity in chest X-ray classification.

## 📊 Changes Summary

### New Files (27 files, +6,021 lines)

#### Models
- `src/classifier/models/attention.py` - SE-Block, CBAM, ECA modules
- `src/classifier/models/attention_models.py` - ResNet/DenseNet wrappers with attention

#### Configuration
- `configs/attention_config.yaml` - Attention module configurations
- `configs/experiment_config.yaml` - Training hyperparameters

#### Scripts
- `scripts/train_all_models.sh` - Train all 7 models
- `scripts/evaluate_models.py` - Evaluate on test set
- `scripts/compare_results.py` - Statistical analysis (t-test, ANOVA)
- `scripts/visualize_results.py` - Performance visualizations
- `scripts/visualize_attention.py` - Grad-CAM attention maps
- `scripts/push_to_kaggle.py` - Push to Kaggle
- `scripts/push_to_kaggle.sh` - Quick push script

#### Notebooks
- `notebooks/kaggle_train_all_models.ipynb` - Train on Kaggle GPU
- `notebooks/01_data_exploration.ipynb` - Dataset analysis
- `notebooks/02_attention_comparison.ipynb` - Model comparison
- `notebooks/03_tb_sensitivity_analysis.ipynb` - TB deep dive

#### Documentation
- `docs/ATTENTION_TODO.md` - Implementation checklist (updated)
- `docs/RESULTS_ANALYSIS.md` - Results template
- `docs/KAGGLE_TRAINING_GUIDE.md` - Kaggle training guide
- `README_attention.md` - Quick start guide

#### Modified
- `train.py` - Added `--attention` and `--attention_type` arguments
- `.gitignore` - Added Kaggle credentials

## 🏗️ Architecture

### Model Variants (7 total)

| Model | Backbone | Attention | Params | Expected TB Sensitivity |
|-------|----------|-----------|--------|------------------------|
| M1 | ResNet50 | None (Baseline) | 25.6M | 0.75-0.80 |
| M2 | ResNet50 | SE-Block | 28.1M | 0.80-0.85 |
| **M3** | **ResNet50** | **CBAM** ⭐ | 28.5M | **0.82-0.87** |
| M4 | ResNet50 | ECA-Net | 26.2M | 0.78-0.83 |
| M5 | DenseNet121 | None (Baseline) | 8.0M | 0.73-0.78 |
| M6 | DenseNet121 | SE-Block | 8.9M | 0.78-0.83 |
| **M7** | **DenseNet121** | **CBAM** ⭐ | 9.2M | **0.80-0.85** |

## ✅ Implementation Status

| Phase | Status | Progress |
|-------|--------|----------|
| Phase 1: Implementation | ✅ Complete | 100% |
| Phase 2: Training | ⏳ Ready | 0% (pending Kaggle) |
| Phase 3: Evaluation | ✅ Ready | 100% (scripts ready) |
| Phase 4: Documentation | ✅ Complete | 100% |
| Phase 5: Deployment | ⏳ Pending | 0% |

**Overall:** 50% complete

## 🚀 How to Train

### On Kaggle (Recommended - Free GPU P100)

```bash
# Push notebook to Kaggle
./scripts/push_to_kaggle.sh

# Or manually
uv run python scripts/push_to_kaggle.py --open
```

**Notebook URL:** https://www.kaggle.com/code/luanbhk/chest-x-ray-attention-training-all-7-models

**Steps:**
1. Open notebook on Kaggle
2. Enable GPU: Settings → Accelerator → GPU
3. Click "Run All"
4. Wait ~3-4 hours for all 7 models
5. Download results

### Local Training (CPU)

```bash
./scripts/train_all_models.sh
```

**Estimated time:** 12-24 hours on CPU

## 📈 Evaluation (After Training)

```bash
# Evaluate all models
uv run python scripts/evaluate_models.py

# Statistical comparison
uv run python scripts/compare_results.py

# Generate visualizations
uv run python scripts/visualize_results.py

# Grad-CAM attention maps
uv run python scripts/visualize_attention.py --checkpoint results/models/resnet50_cbam/best_model.pth
```

## 🎯 Success Metrics

- [ ] TB sensitivity > 0.85 for best attention model
- [ ] TB sensitivity improvement > 5% vs baseline
- [ ] Overall accuracy > 0.90
- [ ] Statistical significance p < 0.05
- [ ] Grad-CAM shows attention on lung regions

## 📝 Hypotheses

| ID | Hypothesis | Expected |
|----|------------|----------|
| H1 | Attention models > Baseline | +3-5% accuracy |
| H2 | CBAM > SE > ECA | CBAM best for TB |
| H3 | Attention improves TB sensitivity | +5-10% recall |
| H4 | DenseNet+Attn > ResNet+Attn | Better feature reuse |
| H5 | Overhead acceptable | <10% slower |

## 🔗 Related Links

- **Design Doc:** `docs/ATTENTION_DESIGN.md`
- **TODO List:** `docs/ATTENTION_TODO.md`
- **Training Guide:** `docs/KAGGLE_TRAINING_GUIDE.md`
- **Kaggle Notebook:** https://www.kaggle.com/code/luanbhk/chest-x-ray-attention-training-all-7-models

## 📚 References

1. SE-Net: Hu et al. "Squeeze-and-Excitation Networks" (CVPR 2018)
2. CBAM: Woo et al. "CBAM: Convolutional Block Attention Module" (ECCV 2018)
3. ECA-Net: Wang et al. "ECA-Net: Efficient Channel Attention" (CVPR 2020)
4. ResNet: He et al. "Deep Residual Learning" (CVPR 2016)
5. DenseNet: Huang et al. "Densely Connected Convolutional Networks" (CVPR 2017)

## ⚠️ Notes

- Training is currently running on Kaggle GPU
- Results will be filled in `docs/RESULTS_ANALYSIS.md` after training completes
- Best model will be deployed to Hugging Face Space after evaluation

---

**Branch:** `feature/attention-mechanism`
**Commits:** 5
**Author:** Luân B.

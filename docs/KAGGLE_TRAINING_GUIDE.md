# 🚀 Quick Start - Kaggle Training

## Cách Train 7 Models trên Kaggle GPU (Free P100)

### Option 1: Tự động (Recommended) ⭐

```bash
# 1. Push notebook lên Kaggle
uv run python scripts/push_to_kaggle.py --open

# 2. Mở Kaggle và click "Run All"
# URL: https://www.kaggle.com/code/luanbhk/chest-xray-attention-training
```

### Option 2: Thủ công

1. **Mở Kaggle**: https://www.kaggle.com/code/create

2. **Upload notebook**: 
   - File: `notebooks/kaggle_train_all_models.ipynb`

3. **Cấu hình**:
   - Settings → Accelerator → **GPU** ✅
   - Settings → Internet → **On** ✅
   - Add Dataset → Search "chest-xray-pneumoniacovid19tuberculosis"

4. **Run**: Click "Run All" button

---

## 📊 Training Progress

### Timeline (Estimated)

| Step | Time | Description |
|------|------|-------------|
| Setup | 5 min | Install deps, clone repo |
| Download Data | 5 min | Download & unzip dataset |
| Train Model 1-4 | 60-90 min | ResNet50 variants |
| Train Model 5-7 | 45-60 min | DenseNet121 variants |
| **TOTAL** | **~2-3 hours** | All 7 models |

### Monitoring

Trong Kaggle notebook, bạn sẽ thấy:
```
🚀 [1/7] Training: resnet50_baseline
  Backbone: resnet50
  Attention: none
  Start: 10:30:00
  ...
✅ resnet50_baseline completed successfully!
```

---

## 📥 Download Results

Sau khi training xong:

### Cách 1: Download qua Kaggle UI
1. Vào notebook của bạn trên Kaggle
2. Click "Output" tab
3. Download `results/models/` folder

### Cách 2: Download qua API
```bash
kaggle kernels output luanbhk/chest-xray-attention-training -p ./downloaded_results
```

### Cách 3: Download từ HF Space (nếu upload)
```bash
huggingface-cli download kinhluan/chest-xray-disease-classifier results/models
```

---

## 🔧 Troubleshooting

### Lỗi: GPU not available
```
❌ GPU Available: False
```

**Fix:**
- Settings → Accelerator → Chọn **GPU**
- Refresh page
- Re-run từ cell "Verify GPU"

### Lỗi: Dataset not found
```
❌ Dataset not found at data/raw
```

**Fix:**
- Re-run cell "Download Dataset"
- Hoặc add dataset manually:
  - Click "+ Add data" → Search "jtiptj/chest-xray-pneumoniacovid19tuberculosis"

### Lỗi: Out of Memory
```
RuntimeError: CUDA out of memory
```

**Fix:**
- Giảm batch size: `BATCH_SIZE = 16` (trong notebook)
- Re-run cell "Training Configuration"
- Continue từ cell "Train All Models"

---

## 📈 Next Steps

Sau khi có models:

### 1. Evaluate (trên Kaggle)
```python
!python scripts/evaluate_models.py
!python scripts/compare_results.py
!python scripts/visualize_results.py
```

### 2. Download & Analyze Local
```bash
# Download results
kaggle kernels output luanbhk/chest-xray-attention-training -p ./results

# Run notebooks local
jupyter notebook notebooks/02_attention_comparison.ipynb
```

### 3. Deploy Best Model
```bash
# Upload best model to HF Space
hf upload space kinhluan/chest-xray-disease-classifier \
  results/models/resnet50_cbam/best_model.pth
```

---

## 💡 Tips

1. **Save Version**: Trước khi run, click "Save Version" để tạo checkpoint
2. **Run Detached**: Click "Run All" rồi có thể tắt browser, Kaggle sẽ continue
3. **Check Quota**: GPU free có giới hạn ~30 hours/tuần
4. **Save Output**: Notebook tự động save output, nhưng nên download results

---

## 📞 Support

- Kaggle Docs: https://www.kaggle.com/docs/notebooks
- GPU Quota: https://www.kaggle.com/settings → GPU
- Discussion: https://www.kaggle.com/discussion

---

**Happy Training! 🎉**

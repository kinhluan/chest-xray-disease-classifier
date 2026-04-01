#!/bin/bash
# Train all attention-enhanced models for chest X-ray classification
# Reference: docs/ATTENTION_TODO.md - Phase 2.2

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DATA_DIR="data/raw"
OUTPUT_DIR="results/models"
EPOCHS=50
BATCH_SIZE=32
LEARNING_RATE=1e-4

# Model configurations
# Format: "model_name:backbone:attention_type"
MODELS=(
    "resnet50_baseline:resnet50:none"
    "resnet50_se:resnet50:se"
    "resnet50_cbam:resnet50:cbam"
    "resnet50_eca:resnet50:eca"
    "densenet121_baseline:densenet121:none"
    "densenet121_se:densenet121:se"
    "densenet121_cbam:densenet121:cbam"
)

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Chest X-Ray Attention Model Training${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if dataset exists
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}❌ Dataset not found at $DATA_DIR${NC}"
    echo ""
    echo "Please download the dataset first:"
    echo "  ./download_and_train.sh"
    echo "OR"
    echo "  uv run python download_dataset.py"
    exit 1
fi

echo -e "${GREEN}✓ Dataset found at $DATA_DIR${NC}"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo -e "${GREEN}✓ Output directory: $OUTPUT_DIR${NC}"
echo ""

# Count models
TOTAL_MODELS=${#MODELS[@]}
CURRENT=0

# Training loop
for model_config in "${MODELS[@]}"; do
    CURRENT=$((CURRENT + 1))
    
    # Parse config
    IFS=':' read -r model_name backbone attention_type <<< "$model_config"
    
    echo -e "${BLUE}----------------------------------------${NC}"
    echo -e "${BLUE}[$CURRENT/$TOTAL_MODELS] Training: $model_name${NC}"
    echo -e "${BLUE}----------------------------------------${NC}"
    echo "  Backbone: $backbone"
    echo "  Attention: $attention_type"
    echo "  Epochs: $EPOCHS"
    echo "  Batch Size: $BATCH_SIZE"
    echo "  Learning Rate: $LEARNING_RATE"
    echo ""
    
    # Determine model type
    if [ "$attention_type" = "none" ]; then
        MODEL_TYPE="resnet"
        ATTENTION_FLAG=""
    else
        MODEL_TYPE="attention"
        ATTENTION_FLAG="--attention_type $attention_type"
    fi
    
    # Create experiment directory
    EXP_DIR="$OUTPUT_DIR/$model_name"
    mkdir -p "$EXP_DIR"
    
    # Train model
    echo -e "${YELLOW}Starting training...${NC}"
    echo ""
    
    uv run python train.py \
        --data_dir "$DATA_DIR" \
        --model_type "$MODEL_TYPE" \
        --model_name "$backbone" \
        $ATTENTION_FLAG \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LEARNING_RATE \
        --output_dir "$OUTPUT_DIR" \
        --experiment_name "$model_name" \
        --scheduler cosine \
        --use_amp \
        --seed 42
    
    # Check if training was successful
    if [ -f "$EXP_DIR/best_model.pth" ]; then
        echo ""
        echo -e "${GREEN}✓ Training completed successfully!${NC}"
        echo "  Checkpoint saved: $EXP_DIR/best_model.pth"
    else
        echo ""
        echo -e "${RED}✗ Training failed! No checkpoint found.${NC}"
        exit 1
    fi
    
    echo ""
done

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✓ All models trained successfully!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Evaluate models: uv run python scripts/evaluate_models.py"
echo "  2. Compare results: uv run python scripts/compare_results.py"
echo "  3. Visualize: uv run python scripts/visualize_results.py"
echo ""
echo "Results directory: $OUTPUT_DIR"

#!/bin/bash
# Deploy script - Push to both GitHub and Hugging Face Spaces

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}==================================${NC}"
echo -e "${GREEN}  Deploy to GitHub & Hugging Face${NC}"
echo -e "${GREEN}==================================${NC}"
echo ""

# Configuration
HF_SPACE_ID="kinhluan/chest-xray-disease-classifier"
GIT_REMOTE=$(git remote get-url origin 2>/dev/null || echo "")

# Check if huggingface-cli is logged in
echo -e "${YELLOW}Checking Hugging Face login...${NC}"
if ! hf whoami &>/dev/null; then
    echo -e "${RED}Not logged in to Hugging Face.${NC}"
    echo "Please run: hf login"
    exit 1
fi
echo -e "${GREEN}✓ Logged in to Hugging Face${NC}"
echo ""

# Step 1: Push to GitHub
echo -e "${YELLOW}Step 1: Pushing to GitHub...${NC}"
if [ -z "$GIT_REMOTE" ]; then
    echo -e "${RED}No git remote found. Please add your GitHub remote:${NC}"
    echo "  git remote add origin https://github.com/kinhluan/chest-xray-disease-classifier.git"
    exit 1
fi

git add -A
git commit -m "Deploy: $(date '+%Y-%m-%d %H:%M:%S')" || echo "No changes to commit"
git push origin master
echo -e "${GREEN}✓ Pushed to GitHub${NC}"
echo ""

# Step 2: Push to Hugging Face Spaces
echo -e "${YELLOW}Step 2: Pushing to Hugging Face Spaces...${NC}"
echo "Space ID: $HF_SPACE_ID"

# Create HF deployment files
echo "Creating deployment files..."

# Copy app.py if not exists in space
cp app.py app_hf.py

# Deploy using huggingface-cli
hf upload "$HF_SPACE_ID" . "." \
    --include="*.py" \
    --include="*.txt" \
    --include="*.toml" \
    --include="*.md" \
    --include="*.json" \
    --exclude=".git/*" \
    --exclude="*.pth" \
    --exclude="*.pt" \
    --exclude="data/*" \
    --exclude=".venv/*" \
    --exclude="__pycache__/*"

echo -e "${GREEN}✓ Pushed to Hugging Face Spaces${NC}"
echo ""

# Cleanup
rm -f app_hf.py

echo -e "${GREEN}==================================${NC}"
echo -e "${GREEN}  Deployment Complete!${NC}"
echo -e "${GREEN}==================================${NC}"
echo ""
echo "GitHub: https://github.com/kinhluan/chest-xray-disease-classifier"
echo "Hugging Face Space: https://huggingface.co/spaces/$HF_SPACE_ID"
echo ""
echo -e "${YELLOW}Note: Hugging Face Space will rebuild automatically.${NC}"

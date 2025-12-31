#!/bin/bash
# ============================================================================
# Setup Script for NLP II Final Project
# ============================================================================
# Automated setup including venv creation and dependency installation
#
# Usage:
#   bash setup.sh
#
# This will:
# 1. Create a virtual environment
# 2. Install PyTorch with CUDA support
# 3. Install all project dependencies (including Unsloth)
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_step() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}[OK] $1${NC}"
}

print_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

# Check Python version
print_step "Checking Python"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

if ! command -v python3 &> /dev/null; then
    print_error "Python 3 not found"
    exit 1
fi

# Create virtual environment
print_step "Creating Virtual Environment"
if [ -d "venv_pipeline" ]; then
    print_warning "Virtual environment already exists. Removing..."
    rm -rf venv_pipeline
fi

python3 -m venv venv_pipeline
print_success "Virtual environment created at venv_pipeline/"

# Activate venv
print_step "Activating Virtual Environment"
source venv_pipeline/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
print_step "Upgrading pip, setuptools, wheel"
pip install --upgrade pip setuptools wheel -q
print_success "pip upgraded"

# Install PyTorch with CUDA support
print_step "Installing PyTorch with CUDA 12.1"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
print_success "PyTorch installed"

# Verify PyTorch
echo "Verifying PyTorch installation..."
python3 -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"
if [ $? -eq 0 ]; then
    print_success "PyTorch verification passed"
else
    print_warning "PyTorch verification failed"
fi

# Install project dependencies
print_step "Installing Project Dependencies"
pip install -r requirements.txt
print_success "Project dependencies installed"

# Verify critical installations
print_step "Verifying Installations"
python3 -c "import transformers; print(f'  Transformers: {transformers.__version__}')" && print_success "Transformers OK"
python3 -c "import peft; print(f'  PEFT: {peft.__version__}')" && print_success "PEFT OK"
python3 -c "import datasets; print(f'  Datasets: {datasets.__version__}')" && print_success "Datasets OK"
python3 -c "import unsloth; print('  Unsloth loaded')" && print_success "Unsloth OK" || print_warning "Unsloth not available"
python3 -c "import vllm; print(f'  vLLM: {vllm.__version__}')" && print_success "vLLM OK" || print_warning "vLLM not available"

# Final message
print_step "Setup Complete!"
echo -e "${GREEN}"
echo "Setup completed successfully!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv_pipeline/bin/activate"
echo ""
echo "To run the complete pipeline, use:"
echo "  python main.py --mode full"
echo ""
echo "Or use the bash script:"
echo "  ./run_pipeline.sh full"
echo -e "${NC}"

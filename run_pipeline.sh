#!/bin/bash
# ============================================================================
# NLP II Final Project - Full Pipeline Execution Script
# ============================================================================
# Fine-tune Once, Serve Anywhere
#
# This script runs the complete project pipeline:
# 1. Data preparation
# 2. Fine-tuning with both frameworks
# 3. Model evaluation
# 4. Inference benchmarking
# 5. Visualization generation
#
# Usage:
#   ./run_pipeline.sh              # Run everything
#   ./run_pipeline.sh data         # Only data preparation
#   ./run_pipeline.sh train        # Only training (both frameworks)
#   ./run_pipeline.sh evaluate     # Only evaluation
#   ./run_pipeline.sh benchmark    # Only benchmarking
#   ./run_pipeline.sh visualize    # Only visualizations
# ============================================================================

# Don't exit on error - we want to continue even if some steps fail
set +e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print header
print_header() {
    echo -e "\n${BLUE}======================================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}======================================================================${NC}\n"
}

# Print step
print_step() {
    echo -e "\n${GREEN}[STEP $1] $2${NC}"
    echo "----------------------------------------------------------------------"
}

# Print error
print_error() {
    echo -e "${RED}ERROR: $1${NC}"
}

# Print warning
print_warning() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_step "0" "Checking prerequisites"
    
    # Check Python
    if ! command -v python &> /dev/null; then
        print_error "Python not found"
        exit 1
    fi
    echo "Python: $(python --version)"
    
    # Check CUDA
    if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        echo "CUDA: Available"
        python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
    else
        print_warning "CUDA not available. Training will not work."
    fi
    
    # Check required directories
    mkdir -p data/processed results/benchmarks results/evaluation results/plots outputs
    echo "Directories: OK"
}

# Step 1: Data Preparation
run_data() {
    print_step "1" "Data Preparation"
    python main.py --mode data
    if [ $? -ne 0 ]; then
        print_error "Data preparation failed"
        return 1
    fi
}

# Clear GPU memory between steps
clear_gpu() {
    python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true
}

# Step 2: Training
run_train_transformers() {
    print_step "2a" "Training with Transformers + PEFT"
    clear_gpu
    python main.py --mode train --trainer transformers
    if [ $? -ne 0 ]; then
        print_error "Transformers training failed"
        return 1
    fi
    clear_gpu
}

run_train_unsloth() {
    print_step "2b" "Training with Unsloth"
    clear_gpu
    python main.py --mode train --trainer unsloth
    if [ $? -ne 0 ]; then
        print_warning "Unsloth training failed (optional)"
        return 1
    fi
    clear_gpu
}

# Step 3: Evaluation
run_evaluate() {
    print_step "3" "Model Evaluation"
    clear_gpu
    
    # Evaluate Transformers model
    if [ -d "outputs/transformers" ]; then
        echo "Evaluating Transformers model..."
        python main.py --mode evaluate --trainer transformers --eval-samples 50
    fi
    
    clear_gpu
    
    # Evaluate Unsloth model
    if [ -d "outputs/unsloth" ]; then
        echo "Evaluating Unsloth model..."
        python main.py --mode evaluate --trainer unsloth --eval-samples 50
    fi
    
    clear_gpu
}

# Step 4: Benchmarking
run_benchmark() {
    print_step "4" "Inference Benchmarking"
    clear_gpu
    
    # Benchmark each framework separately to avoid memory issues
    echo "Benchmarking Transformers..."
    python main.py --mode benchmark --frameworks transformers || print_warning "Transformers benchmark failed"
    clear_gpu
    
    echo "Benchmarking Unsloth..."
    python main.py --mode benchmark --frameworks unsloth || print_warning "Unsloth benchmark failed"
    clear_gpu
    
    echo "Benchmarking vLLM..."
    python main.py --mode benchmark --frameworks vllm || print_warning "vLLM benchmark failed"
    clear_gpu
    
    echo "Benchmarking Ollama..."
    python main.py --mode benchmark --frameworks ollama || print_warning "Ollama benchmark failed"
    clear_gpu
}

# Step 5: Visualizations
run_visualize() {
    print_step "5" "Generating Visualizations"
    python main.py --mode visualize
}

# Full pipeline
run_full() {
    print_header "NLP II Final Project - Full Pipeline"
    
    START_TIME=$(date +%s)
    
    check_prerequisites
    
    # Track which steps succeeded
    STEPS_OK=""
    STEPS_FAILED=""
    
    run_data && STEPS_OK="$STEPS_OK data" || STEPS_FAILED="$STEPS_FAILED data"
    run_train_transformers && STEPS_OK="$STEPS_OK train-transformers" || STEPS_FAILED="$STEPS_FAILED train-transformers"
    run_train_unsloth && STEPS_OK="$STEPS_OK train-unsloth" || STEPS_FAILED="$STEPS_FAILED train-unsloth"
    run_evaluate && STEPS_OK="$STEPS_OK evaluate" || STEPS_FAILED="$STEPS_FAILED evaluate"
    run_benchmark && STEPS_OK="$STEPS_OK benchmark" || STEPS_FAILED="$STEPS_FAILED benchmark"
    run_visualize && STEPS_OK="$STEPS_OK visualize" || STEPS_FAILED="$STEPS_FAILED visualize"
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    MINUTES=$((DURATION / 60))
    
    print_header "Pipeline Complete!"
    echo "Total time: ${MINUTES} minutes (${DURATION} seconds)"
    echo -e "${GREEN}Completed:${NC}$STEPS_OK"
    if [ -n "$STEPS_FAILED" ]; then
        echo -e "${YELLOW}Failed:${NC}$STEPS_FAILED"
    fi
    echo "Results saved in: results/"
    echo "Plots saved in: results/plots/"
}

# Main
main() {
    MODE=${1:-full}
    
    case $MODE in
        full)
            run_full
            ;;
        data)
            check_prerequisites
            run_data
            ;;
        train)
            check_prerequisites
            run_train_transformers
            run_train_unsloth
            ;;
        train-transformers)
            check_prerequisites
            run_train_transformers
            ;;
        train-unsloth)
            check_prerequisites
            run_train_unsloth
            ;;
        evaluate)
            check_prerequisites
            run_evaluate
            ;;
        benchmark)
            check_prerequisites
            run_benchmark
            ;;
        visualize)
            check_prerequisites
            run_visualize
            ;;
        *)
            echo "Usage: $0 {full|data|train|train-transformers|train-unsloth|evaluate|benchmark|visualize}"
            exit 1
            ;;
    esac
}

main "$@"

#!/bin/bash

# Navigate to the script's directory
cd "$(dirname "$0")"

LOGFILE="../Log.txt"

# Logging function
log() {
    local msg="$1"
    if [ "$msg" = "_BLANK_" ]; then
        echo ""
        echo "" >> "$LOGFILE"
    else
        echo "$msg"
        echo "$msg" >> "$LOGFILE"
    fi
}

log "========================================================"
log "           SPARK LLM MASSIVE-SCALE PIPELINE (LINUX)"
log "========================================================"
log "_BLANK_"

log "[1/8] Streaming Extrapolated Dataset Sample for Vocabulary [Stage 12]..."
python3 src/data_pipeline/hf_collector.py
if [ $? -ne 0 ]; then
    log "Error during Data Collection."
    exit 1
fi
log "_BLANK_"

log "[2/8] Training BPE Tokenizer..."
python3 src/tokenizer/bpe_tokenizer.py
if [ $? -ne 0 ]; then
    log "Error during Tokenizer Training."
    exit 1
fi
log "_BLANK_"

log "[3/8] Compiling High-Velocity Binary Dataset Memmap [Stage 12, Part 2]..."
python3 src/data_pipeline/hf_build_bin.py
if [ $? -ne 0 ]; then
    log "Error during Binary Data Compilation."
    exit 1
fi
log "_BLANK_"

log "[4/8] Running Deep Neural Pretraining [with Checkpointing]..."
python3 src/training/pretrain.py
if [ $? -ne 0 ]; then
    log "Error during Base Pretraining."
    exit 1
fi
log "_BLANK_"

log "[5/8] Running Instruction Fine-Tuning..."
python3 src/training/finetune.py
if [ $? -ne 0 ]; then
    log "Error during Instruction Fine-Tuning."
    exit 1
fi
log "_BLANK_"

log "[6/8] Running Evaluation and Benchmarking..."
python3 src/evaluation/evaluate.py
if [ $? -ne 0 ]; then
    log "Error during Evaluation."
    exit 1
fi
log "_BLANK_"

log "[7/8] Running Reinforcement Learning [RLHF]..."
python3 src/training/rlhf_tune.py
if [ $? -ne 0 ]; then
    log "Error during Reinforcement Tuning."
    exit 1
fi
log "_BLANK_"

log "[8/8] Securing Output to Final_output/model..."
mkdir -p Final_output/model
cp -r src/model/*.pt Final_output/model/ 2>/dev/null
if [ $? -ne 0 ]; then
    log "Error saving the model. (Or no .pt files generated)"
    # We won't exit here just in case, because python inherently auto-saved them already
fi
log "_BLANK_"

log "========================================================"
log "      SPARK LLM AUTO-TRAINING COMPLETELY FINISHED"
log "========================================================"
log "_BLANK_"
log "The massive-scale model is now fully trained and ready to talk!"
echo "To chat with your custom model, download the .pt files to your Windows machine,"
echo "put them in the Final_output folder, and start the Streamlit app!"
echo ""

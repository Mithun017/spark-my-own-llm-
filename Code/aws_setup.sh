#!/bin/bash

# SPARK LLM - AWS Mass-Scale Training Setup Script
# Run this on your AWS EC2 Ubuntu instance (Deep Learning AMI recommended)

echo "=========================================="
echo "    SPARK LLM CLOUD SUPERCOMPUTING"
echo "=========================================="
echo "1. Updating System Packages..."
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv unzip curl

echo "2. Setting up Python Virtual Environment..."
python3 -m venv spark_env
source spark_env/bin/activate

echo "3. Installing PyTorch & GPU Dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "4. Installing AI Ecosystem Libraries..."
pip install numpy tokenizers datasets streamlit transformers

echo "=========================================="
echo "    AWS ENVIRONMENT READY!"
echo "=========================================="
echo ""
echo "To start the massive-scale training, make sure your 'Code' folder is uploaded to this server, then run:"
echo ""
echo "source spark_env/bin/activate"
echo "cd Code"
echo "chmod +x auto_train.bat  # (Or run the python scripts manually: python src/training/pretrain.py)"
echo "python src/data_pipeline/hf_collector.py"
echo "python src/tokenizer/bpe_tokenizer.py"
echo "python src/data_pipeline/hf_build_bin.py"
echo "python src/training/pretrain.py"
echo "python src/training/finetune.py"
echo ""
echo "Note: It's highly recommended to run training inside 'tmux' or 'screen' so it keeps running if your SSH disconnects."

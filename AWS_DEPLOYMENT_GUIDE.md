# ⚡ SPARK LLM: AWS Supercomputing Deployment Guide

Since you are restricted by your laptop's 6GB VRAM, we have upgraded your architecture inside `config.py` to **1024 Dimensions, 12 Layers, and 16 Attention Heads**. This makes your model absolutely massive (comparable to GPT-2 Large) and highly capable of deep logic.

Follow these strict instructions to train your model on the Cloud natively.

## 1. Rent the AWS GPU Instance
1. Log into your **AWS Management Console**.
2. Go to **EC2** -> **Launch Instance**.
3. **Name**: `Spark-GPU-Trainer`
4. **AMI (OS)**: Search for and select the **"Deep Learning AMI GPU PyTorch"** (Ubuntu). This comes with Nvidia drivers pre-installed.
5. **Instance Type**: 
   - **`g5.2xlarge`** (1 x A10G GPU, 24GB VRAM) -> ~ $1.20 / hr (Recommended)
   - **`p3.2xlarge`** (1 x V100 GPU, 16GB VRAM) -> ~ $3.00 / hr
   - **`g4dn.xlarge`** (1 x T4 GPU, 16GB VRAM) -> ~ $0.50 / hr (Cheapest, but 2x slower)
6. **Storage**: Increase the Root EBS Volume to **100 GB** minimum (because the OpenWebText dataset will download 38GB of data).
7. Create a Key Pair, download the `.pem` file, and **Launch**.

## 2. Upload your Code to AWS
To move this entire folder to your AWS server, you can Zip it and upload it, or push it to GitHub.
If using GitHub:
```bash
# On your local laptop:
git init
git add .
git commit -m "Spark AWS Deployment"
git remote add origin https://github.com/YOUR_USERNAME/spark-llm.git
git push -u origin main
```

## 3. SSH and Run the Setup Script
Open your terminal and SSH into your new AWS Server:
```bash
ssh -i "your-key.pem" ubuntu@YOUR_AWS_IP
```

Clone your repository (or unzip your files):
```bash
git clone https://github.com/YOUR_USERNAME/spark-llm.git
cd spark-llm/Code
```

Make the bash setup script executable and run it:
```bash
chmod +x aws_setup.sh
./aws_setup.sh
```

## 4. Run the Training (Detached)
Deep learning takes many hours, and if your Wi-Fi disconnects, the SSH session dies. You must run the training in a detached background terminal using `tmux`.

```bash
# Activate your environment
source spark_env/bin/activate

# Create a background terminal session
tmux new -s spark_training

# Run the training stages!
python src/data_pipeline/hf_collector.py
python src/tokenizer/bpe_tokenizer.py
python src/data_pipeline/hf_build_bin.py
python src/training/pretrain.py
python src/training/finetune.py
```
*(To exit the background terminal without stopping it, press `Ctrl+B`, let go, then press `D`)*

## 5. Download the Brain back to your Laptop
Once `finetune.py` reaches 3,000 epochs and finishes, your highly intelligent weights will be stored in `src/model/spark_llm_instruct.pt`.

You DO NOT pay AWS to run the Streamlit UI! Shut down your AWS instance to stop paying per hour. 
Simply download the `spark_llm_instruct.pt` and `spark_llm_weights.pt` files back to your Windows laptop. Place them inside `Code/Final_output/model/`.

Open your local Windows PowerShell and boot the app:
```powershell
cd Code/Final_output
streamlit run app.py
```
Because *Inference* purely runs on VRAM sequentially, your 6GB laptop GPU can effortlessly run the 1.5GB generated model. It only struggles with training!

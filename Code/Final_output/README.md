# ⚡ Final SPARK LLM Deliverable

This directory contains the final web interface for your custom Small Language Model.

The SPARK neural network (trained in `src/training/`) is exposed here via a clean **Streamlit** Chat interface.

## How to Run the App

1. Ensure you have activated your virtual environment (if using one) and installed dependencies:
   ```bash
   pip install -r ../requirements.txt
   pip install streamlit
   ```

2. Because the model architecture lives in `src/`, run this Streamlit app from **inside** the `Final_output` directory:
   ```bash
   cd "C:\Users\MITHUN\Desktop\STUDIES\PROJECT\42.SPARK - My own llm\Code\Final_output"
   streamlit run app.py
   ```

3. Your browser will automatically open to `http://localhost:8501`, displaying the SPARK chat window!

## Important Notes:
- The app automatically tries to load the weights from `src/model/spark_llm_instruct.pt`.
- If you have not trained the model using `pretrain.py` and `finetune.py`, the weights are completely random, and the model will output gibberish!
- Use the sidebar on the left side of the Streamlit window to adjust **Temperature**, **Top-K**, and **Max Tokens** in real-time.

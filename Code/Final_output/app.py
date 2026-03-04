import streamlit as st
import torch
import os
import sys
import json
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.inference.generator import SparkGenerator

@st.cache_resource
def load_model():
    return SparkGenerator(use_instruct=True)

st.set_page_config(page_title="SPARK LLM", page_icon="⚡", layout="centered")

st.markdown("""
<div style="text-align: center;">
    <h1>⚡ SPARK LLM ⚡</h1>
    <p>Custom Small Language Model Interface</p>
</div>
""", unsafe_allow_html=True)

try:
    generator = load_model()
    st.success("✅ Neural Network Loaded and Ready!", icon="🤖")
except Exception as e:
    st.error(f"Failed to load model architecture or weights: {e}")
    st.stop()

st.sidebar.header("Hyperparameters")
max_tokens = st.sidebar.slider("Max New Tokens", min_value=10, max_value=200, value=50)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
top_k = st.sidebar.slider("Top-K Sampling", min_value=1, max_value=50, value=10)
rep_penalty = st.sidebar.slider("Repetition Penalty", min_value=1.0, max_value=2.0, value=1.2, step=0.05)

if "messages" not in st.session_state:
    st.session_state.messages = []

def log_rlhf(prompt: str, response: str, score: int):
    log_file = os.path.join(BASE_DIR, 'data', 'rlhf_logs.json')
    try:
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        else:
            logs = []
            
        logs.append({
            "prompt": prompt,
            "response": response,
            "score": score,
            "timestamp": time.time()
        })
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=4)
        st.toast(f"Feedback Logged for RLHF Pipeline! Score: {'+1' if score > 0 else '-1'}")
    except Exception as e:
        st.error(f"Failed to log RLHF feedback: {e}")

for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message["role"] == "assistant":
            col1, col2, col3 = st.columns([1,1,10])
            with col1:
                if st.button("👍", key=f"up_{idx}"):
                    prior_prompt = st.session_state.messages[idx-1]["content"] if idx > 0 else "Unknown"
                    log_rlhf(prior_prompt, message["content"], 1)
            with col2:
                if st.button("👎", key=f"down_{idx}"):
                    prior_prompt = st.session_state.messages[idx-1]["content"] if idx > 0 else "Unknown"
                    log_rlhf(prior_prompt, message["content"], -1)

if prompt := st.chat_input("What is your instruction for SPARK?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("SPARK is thinking..."):
            try:
                # The generator cleanly executes and returns ONLY the new tokens now
                answer = generator.generate(
                    prompt=prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    repetition_penalty=rep_penalty
                )
                
                # Cleanup edge-case trailing EOS strings if tokenizer slips
                answer = answer.replace("[EOS]", "").strip()
                    
                message_placeholder.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"LLM Engine Crash: {e}")

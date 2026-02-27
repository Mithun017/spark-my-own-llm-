import streamlit as st
import torch
import os
import sys

# Ensure we can import the architecture from the src folder below
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.inference.generator import SparkGenerator

@st.cache_resource
def load_model():
    # Cache the model to avoid reloading on every Streamlit interaction
    return SparkGenerator(use_instruct=True)

st.set_page_config(page_title="SPARK SLM", page_icon="⚡", layout="centered")

st.markdown("""
<div style="text-align: center;">
    <h1>⚡ SPARK SLM ⚡</h1>
    <p>Custom Small Language Model Interface</p>
</div>
""", unsafe_allow_html=True)

# Load generator
try:
    generator = load_model()
    st.success("✅ Neural Network Loaded and Ready!", icon="🤖")
except Exception as e:
    st.error(f"Failed to load model architecture or weights: {e}")
    st.stop()

# Sidebar controls
st.sidebar.header("Hyperparameters")
max_tokens = st.sidebar.slider("Max New Tokens", min_value=10, max_value=200, value=50)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
top_k = st.sidebar.slider("Top-K Sampling", min_value=1, max_value=50, value=10)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is your instruction for SPARK?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Actually run the Model inference here
        with st.spinner("SPARK is thinking..."):
            try:
                # The generator.py logic handles internal tokenizing and softmax autoregression
                raw_output = generator.generate(
                    prompt=prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k
                )
                
                # The generator returns the full text (including prompt), so we clean it up for the chat UI
                # Usually we just want the newly generated part after the prompt.
                # In finetuning, it's structured as "Response: "
                if "Response:" in raw_output:
                    answer = raw_output.split("Response:")[-1].replace("[EOS]", "").strip()
                else:
                    answer = raw_output.replace(prompt, "").replace("[EOS]", "").strip()
                    
                message_placeholder.markdown(answer)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Inference Engine Crash: {e}")

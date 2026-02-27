from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os
import uvicorn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.agent.agent_core import SparkAgent
from src.deployment.safety import is_safe_prompt, format_refusal

# Initialize FastAPI App
app = FastAPI(title="SPARK SLM Deployment API", version="1.0.0")

# Preload the agent (which loads the PyTorch models into VRAM)
print("Booting SPARK SLM Backend...")
# Wrapped in try/except in case weights are missing for immediate run
try:
    spark_agent = SparkAgent()
except Exception as e:
    print(f"Warning: Could not load SPARK model weights natively: {e}")
    spark_agent = None


class ChatRequest(BaseModel):
    prompt: str
    temperature: float = 0.7
    max_tokens: int = 50

class ChatResponse(BaseModel):
    response: str
    tool_triggered: bool = False


@app.post("/v1/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    # 1. Safety Filter Pre-computation
    if not is_safe_prompt(req.prompt):
        return ChatResponse(response=format_refusal(req.prompt))
        
    if spark_agent is None:
        raise HTTPException(status_code=500, detail="SPARK Model weights not found or failed to load. Run training first.")
        
    try:
        # 2. Run the Neural Agent Reasoning Loop
        # Note: Temperature and max_tokens are hardcoded in agent_core atm, but conceptually link here
        answer = spark_agent.execute_reasoning_loop(req.prompt)
        
        return ChatResponse(
            response=answer,
            tool_triggered="<CALL:" in answer
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": spark_agent is not None}

if __name__ == "__main__":
    print("\n--- SPARK SLM Server Online ---")
    print("Access the API docs at: http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.inference.generator import SparkGenerator
from src.agent.tools import TOOLS_REGISTRY

class SparkAgent:
    def __init__(self):
        print("Initializing SPARK Reasoning Core...")
        # Load the instruction-tuned model for the agent
        self.generator = SparkGenerator(use_instruct=True)
        self.system_prompt = "System: You are SPARK, a highly capable analytical SLM. If you need to do math, output '<CALL:calc:math_expression>'. If asked for the time, output '<CALL:time>'. "

    def execute_reasoning_loop(self, user_msg: str):
        # Build prompt
        full_context = f"{self.system_prompt}\nUser: {user_msg}\nResponse: "
        
        # 1. Ask the SLM what it wants to do
        response = self.generator.generate(full_context, max_new_tokens=40, temperature=0.7)
        
        # 2. Check if the SLM generated a Tool Call trigger keyword
        # E.g. <CALL:calc:5*5>
        if "<CALL:" in response:
            try:
                # Naive parsing for educational demo
                call_body = response.split("<CALL:")[1].split(">")[0]
                parts = call_body.split(":")
                tool_name = parts[0]
                tool_args = parts[1] if len(parts) > 1 else ""
                
                if tool_name in TOOLS_REGISTRY:
                    print(f"\n[Agent] Intercepted Tool Call: Executing '{tool_name}'...")
                    if tool_args:
                        tool_result = TOOLS_REGISTRY[tool_name](tool_args)
                    else:
                        tool_result = TOOLS_REGISTRY[tool_name]()
                        
                    print(f"[Agent] Observation from Tool: {tool_result}")
                    
                    # Inject observation back into context and prompt the model again
                    second_pass = full_context + response.split("<CALL:")[0] + f"\n[System Observation: Tool {tool_name} returned '{tool_result}']\nResponse:"
                    print(f"\n[Agent] Resolving final response based on observation...")
                    final_res = self.generator.generate(second_pass, max_new_tokens=30, temperature=0.7)
                    return final_res
                else:
                    return "Error: Model attempted to call an unknown tool."
            except Exception as e:
                return f"Agent parsing error: {e}"
                
        # If no tool was needed, just return the raw SLM prediction
        return response

if __name__ == "__main__":
    agent = SparkAgent()
    # Mocking interaction
    print("\n=== Agent Test ===")
    user_q = "What is 550 * 3?"
    print(f"User: {user_q}")
    # In a real loaded model, it would output <CALL:calc:550*3>
    # Since our weights are currently tiny/mocked, it will just generate text
    agent.execute_reasoning_loop(user_q)

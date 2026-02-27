import re

# Simple hardcoded safety checks for educational SLM blueprint
OFFENSIVE_WORDS = ["kill", "destroy", "hack", "illegal", "murder"]

def is_safe_prompt(prompt: str) -> bool:
    """
    Checks if the user's input prompt violates safety policy.
    """
    prompt_lower = prompt.lower()
    for word in OFFENSIVE_WORDS:
        # Regex to match exact words, preventing sub-word triggers
        if re.search(rf'\b{word}\b', prompt_lower):
            return False
    return True

def format_refusal(prompt: str) -> str:
    return "SPARK Safety Policy Violation: I cannot assist with this request."

def calculator_tool(expression: str) -> str:
    """
    Solves math expressions natively in Python. The SLM will trigger this when it detects a math problem.
    """
    try:
        # Extremely dangerous in prod, but fine for SLM local tool demonstration
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error computing {expression}: {str(e)}"

def get_time_tool() -> str:
    """
    Returns the current local time.
    """
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# A dictionary of accessible tools that the Agent can call
TOOLS_REGISTRY = {
    "calc": calculator_tool,
    "time": get_time_tool
}

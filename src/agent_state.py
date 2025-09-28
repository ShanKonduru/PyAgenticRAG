from typing import TypedDict, Dict

# --- A. Agent State (Inherits from TypedDict for LangGraph compatibility) ---
class AgentState(TypedDict):
    """Defines the state passed between all agents/nodes."""
    query: str
    plan: str # Output from Planning Agent (The Reasoning)
    search_results: Dict[str, str] # Results from sub-agents (The Retrieval)
    final_response: str # The final synthesized output

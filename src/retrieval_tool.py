from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, Any

# --- B. Retrieval Tool Class (Encapsulates Data Access) ---

class RetrievalTool:
    """Encapsulates the logic for accessing a specific data source (e.g., Vector DB)."""
    
    def __init__(self, data_source_name: str):
        self.data_source_name = data_source_name
        # In a real system, initialization of a Vector DB client would go here.
        print(f"Tool initialized for: {data_source_name}")

    def retrieve(self, search_term: str) -> str:
        """Simulates a search/retrieval operation on the specific data source."""
        # Simple conditional logic replacing actual vector store lookup
        if "policy" in search_term.lower() and self.data_source_name == "Local Data":
            return "Local Data: Policy 2.1 requires all sensitive data to be encrypted at rest."
        elif "cloud" in search_term.lower() and self.data_source_name == "Cloud Servers":
            return "Cloud Servers: AWS S3 logs show the last successful backup was 2 hours ago."
        else:
            return f"No specific document found for '{search_term}' in {self.data_source_name}."

# Initialize the LLM once
LLM = Ollama(model="llama3")
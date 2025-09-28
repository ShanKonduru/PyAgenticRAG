
import os
from dotenv import load_dotenv

from langchain_ollama import OllamaLLM
from src.agent_state import AgentState
from src.planning_agent import AggregatorAgent, LocalRAGAgent, PlanningAgent
from src.retrieval_tool import RetrievalTool
from langgraph.graph import StateGraph, END

load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize the LLM once
LLM = OllamaLLM(model="llama3")

def main():
    # --- F. Build and Run the Agentic System ---

    # 1. Initialize Tools
    local_data_tool = RetrievalTool(data_source_name="Local Data")

    # 2. Initialize Agents (Objects)
    planning_agent_obj = PlanningAgent(llm=LLM)
    local_rag_agent_obj = LocalRAGAgent(tool=local_data_tool)
    aggregator_agent_obj = AggregatorAgent(llm=LLM)

    # 3. Build the LangGraph Workflow
    workflow = StateGraph(AgentState)

    # Add nodes, using the .run method of each object
    workflow.add_node("planning", planning_agent_obj.run)
    workflow.add_node("local_rag", local_rag_agent_obj.run)
    workflow.add_node("aggregation", aggregator_agent_obj.run)

    # Set the entry point
    workflow.set_entry_point("planning")

    # Define the edges (transitions)
    workflow.add_edge("planning", "local_rag")
    workflow.add_edge("local_rag", "aggregation")
    workflow.add_edge("aggregation", END)

    # Compile the graph
    app = workflow.compile()

    # 4. Run the Agentic System
    initial_query = "What is the policy regarding sensitive data, and how does that relate to our cloud backup logs?"
    print(f"--- Running Agentic RAG for Query: {initial_query} ---\n")

    final_state = app.invoke({
        "query": initial_query, 
        "plan": "", 
        "search_results": {}, 
        "final_response": ""
    })

    print("\n\n--- FINAL SYNTHESIZED ANSWER ---")
    print(final_state['final_response'])

if __name__ == "__main__":
    main()

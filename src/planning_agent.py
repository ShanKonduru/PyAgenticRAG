from langchain_core.prompts import ChatPromptTemplate
from typing import Dict

from src.agent_state import AgentState
from src.retrieval_tool import RetrievalTool

# --- C. Planning Agent (Handles ReAct/CoT Reasoning) ---

class PlanningAgent:
    """Corresponds to the Planning/ReAct/CoT component."""

    def __init__(self, llm):
        self.llm = llm
        self.tool_names = "Local_RAG_Agent, Cloud_API_Agent"
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"You are a master planner. Analyze the query and use ReAct format to decide which tool/agent to call. Available tools: {self.tool_names}. Your plan should conclude with: Action: [Agent Name] | Action Input: [Query for Agent]",
                ),
                ("human", "User Query: {query}"),
            ]
        )

    def run(self, state: AgentState) -> Dict:
        """Generates the action plan based on the user query."""
        chain = self.prompt | self.llm.bind(stop=["\nObservation"])
        plan_output = chain.invoke({"query": state["query"]})

        print(f"✅ Planning Agent Output (Plan):\n{plan_output}\n")

        # In a real LangGraph setup, the plan_output would be parsed here
        # to determine the next transition in the graph.
        return {"plan": plan_output, "search_results": state.get("search_results", {})}


# --- D. Local RAG Agent (Agent 1 - Handles Local Data Sources) ---


class LocalRAGAgent:
    """Corresponds to Agent 1, interacting with Local Data Sources."""

    def __init__(self, tool: RetrievalTool):
        self.tool = tool

    def run(self, state: AgentState) -> Dict:
        """Executes the retrieval task defined in the plan."""
        # Simplified: extracting a fixed search term for demonstration
        search_term = "internal policy on data"

        retrieved_info = self.tool.retrieve(search_term)

        print(
            f"🔎 Local RAG Agent Output:\nSearch: '{search_term}'\nRetrieved: {retrieved_info[:80]}...\n"
        )

        # Update the shared state with the result
        results = state.get("search_results", {})
        results["local_rag_result"] = retrieved_info

        return {"search_results": results}


# --- E. Aggregator Agent (Synthesizes Final Output) ---


class AggregatorAgent:
    """Corresponds to the Aggregator Agent, synthesizing results from sub-agents."""

    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are the final synthesis engine. Use the gathered context from ALL sources to generate a comprehensive, coherent, and final answer to the user's query.",
                ),
                (
                    "human",
                    "Original Query: {query}\n\nGathered Context:\n{full_context}",
                ),
            ]
        )

    def run(self, state: AgentState) -> Dict:
        """Generates the final, augmented response."""
        # Compile the context from all sub-agents
        full_context = "\n".join(
            [f"Source ({k}): {v}" for k, v in state["search_results"].items()]
        )

        final_response = (self.prompt | self.llm).invoke(
            {"query": state["query"], "full_context": full_context}
        )

        print(f"✨ Aggregator Agent Output (Final):\n{final_response}\n")

        return {"final_response": final_response}

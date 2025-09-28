"""
Enhanced Planning Agent with sophisticated reasoning and error handling
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import re
import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from src.enhanced_agent_state import (
    EnhancedAgentState, 
    AgentExecution, 
    ErrorInfo,
    add_error_to_state,
    update_state_with_execution
)

logger = logging.getLogger(__name__)

class AgentAction(BaseModel):
    """Structured representation of an agent action"""
    agent_name: str = Field(description="Name of the agent to execute")
    action_input: str = Field(description="Input/query for the agent")
    reasoning: str = Field(description="Reasoning behind choosing this agent")
    confidence: float = Field(description="Confidence in this decision (0.0-1.0)")

class ExecutionPlan(BaseModel):
    """Structured execution plan"""
    strategy: str = Field(description="Overall execution strategy")
    steps: List[AgentAction] = Field(description="Sequential steps to execute")
    parallel_capable: bool = Field(description="Whether steps can be executed in parallel")
    estimated_time: int = Field(description="Estimated execution time in seconds")

class EnhancedPlanningAgent:
    """
    Enhanced Planning Agent with sophisticated ReAct reasoning,
    error handling, and adaptive planning capabilities
    """
    
    def __init__(self, llm, available_agents: Optional[List[str]] = None):
        self.llm = llm
        self.available_agents = available_agents or [
            "LocalRAGAgent", 
            "WebSearchAgent", 
            "DatabaseAgent", 
            "APIAgent",
            "DocumentAnalysisAgent"
        ]
        
        # Set up structured output parser
        self.plan_parser = PydanticOutputParser(pydantic_object=ExecutionPlan)
        
        # Create the planning prompt
        self.planning_prompt = self._create_planning_prompt()
        
        # ReAct reasoning prompt for step-by-step analysis
        self.react_prompt = self._create_react_prompt()
    
    def _create_planning_prompt(self) -> ChatPromptTemplate:
        """Create the main planning prompt"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a master planning agent responsible for orchestrating a multi-agent RAG system.
            
Available Agents:
{available_agents}

Your task is to analyze the user query and create an optimal execution plan using ReAct reasoning.

Consider:
1. Query complexity and type
2. Required data sources
3. Agent capabilities and limitations  
4. Potential error scenarios
5. Performance optimization
6. Parallel execution opportunities

Output your plan as structured JSON following this format:
{format_instructions}

Be thorough in your reasoning and provide confidence scores for each decision."""),
            
            ("human", """Query: {query}

Previous Context: {context}
Previous Errors: {errors}
Retry Count: {retry_count}

Please analyze this query and create an optimal execution plan.""")
        ])
    
    def _create_react_prompt(self) -> ChatPromptTemplate:
        """Create ReAct reasoning prompt for step-by-step analysis"""
        return ChatPromptTemplate.from_messages([
            ("system", """Use ReAct (Reasoning and Acting) methodology to analyze the query step by step.

Format your response as:
Thought: [Your reasoning about the current situation]
Action: [The action you want to take]
Action Input: [The specific input for the action]
Observation: [Expected outcome or what to look for]

Continue this pattern until you have a complete understanding of what needs to be done.

Available agents: {available_agents}"""),
            
            ("human", "Query: {query}\n\nContext: {context}")
        ])
    
    def run(self, state: EnhancedAgentState) -> Dict[str, Any]:
        """
        Main planning execution with enhanced error handling and reasoning
        """
        execution = AgentExecution(
            agent_name="PlanningAgent",
            start_time=datetime.now()
        )
        state["agent_executions"].append(execution)
        state["current_agent"] = "PlanningAgent"
        
        try:
            logger.info(f"Planning Agent starting for query: {state['query'][:100]}...")
            
            # Step 1: ReAct reasoning for query analysis
            react_analysis = self._perform_react_analysis(state)
            
            # Step 2: Create structured execution plan
            execution_plan = self._create_execution_plan(state, react_analysis)
            
            # Step 3: Validate and optimize the plan
            validated_plan = self._validate_plan(execution_plan, state)
            
            # Update state with plan
            plan_output = {
                "react_analysis": react_analysis,
                "execution_plan": validated_plan.dict(),
                "planning_metadata": {
                    "total_steps": len(validated_plan.steps),
                    "strategy": validated_plan.strategy,
                    "parallel_capable": validated_plan.parallel_capable,
                    "estimated_time": validated_plan.estimated_time
                }
            }
            
            # Update state
            state["plan"] = json.dumps(plan_output, indent=2)
            state["execution_strategy"] = validated_plan.strategy
            state["total_steps"] = len(validated_plan.steps)
            
            execution.complete(plan_output)
            
            logger.info(f"✅ Planning Agent completed with {len(validated_plan.steps)} steps")
            logger.info(f"Strategy: {validated_plan.strategy}")
            
            return {
                "plan": state["plan"],
                "execution_strategy": state["execution_strategy"],
                "total_steps": state["total_steps"],
                "search_results": state.get("search_results", {})
            }
            
        except Exception as e:
            error = ErrorInfo(
                error_type="PlanningError",
                message=f"Planning failed: {str(e)}",
                agent="PlanningAgent",
                recoverable=True
            )
            
            execution.fail(error)
            add_error_to_state(state, error.error_type, error.message, error.agent)
            
            logger.error(f"❌ Planning Agent failed: {e}")
            
            # Fallback to simple plan
            fallback_plan = self._create_fallback_plan(state)
            state["plan"] = fallback_plan
            
            return {
                "plan": state["plan"], 
                "execution_strategy": "fallback",
                "search_results": state.get("search_results", {})
            }
    
    def _perform_react_analysis(self, state: EnhancedAgentState) -> str:
        """Perform ReAct reasoning to analyze the query"""
        try:
            context = self._build_context(state)
            
            chain = self.react_prompt | self.llm
            
            analysis = chain.invoke({
                "query": state["query"],
                "context": context,
                "available_agents": ", ".join(self.available_agents)
            })
            
            logger.info("ReAct analysis completed")
            return str(analysis)
            
        except Exception as e:
            logger.warning(f"ReAct analysis failed, using simplified analysis: {e}")
            return f"Simple analysis: Query requires information retrieval for '{state['query']}'"
    
    def _create_execution_plan(self, state: EnhancedAgentState, react_analysis: str) -> ExecutionPlan:
        """Create a structured execution plan based on ReAct analysis"""
        try:
            context = self._build_context(state)
            errors_context = self._build_errors_context(state)
            
            chain = self.planning_prompt | self.llm | self.plan_parser
            
            plan = chain.invoke({
                "query": state["query"],
                "context": context,
                "errors": errors_context,
                "retry_count": state["retry_count"],
                "available_agents": ", ".join(self.available_agents),
                "format_instructions": self.plan_parser.get_format_instructions()
            })
            
            return plan
            
        except Exception as e:
            logger.warning(f"Structured planning failed, creating simple plan: {e}")
            return self._create_simple_plan(state)
    
    def _validate_plan(self, plan: ExecutionPlan, state: EnhancedAgentState) -> ExecutionPlan:
        """Validate and optimize the execution plan"""
        # Ensure all agent names are valid
        valid_steps = []
        for step in plan.steps:
            if step.agent_name in self.available_agents:
                valid_steps.append(step)
            else:
                logger.warning(f"Invalid agent name in plan: {step.agent_name}")
                # Try to find a similar valid agent
                fallback_agent = self._find_fallback_agent(step.agent_name)
                if fallback_agent:
                    step.agent_name = fallback_agent
                    valid_steps.append(step)
        
        plan.steps = valid_steps
        
        # Ensure we have at least one step
        if not plan.steps:
            plan.steps = [AgentAction(
                agent_name="LocalRAGAgent",
                action_input=state["query"],
                reasoning="Fallback to local RAG search",
                confidence=0.5
            )]
        
        return plan
    
    def _create_simple_plan(self, state: EnhancedAgentState) -> ExecutionPlan:
        """Create a simple fallback execution plan"""
        return ExecutionPlan(
            strategy="simple_retrieval",
            steps=[
                AgentAction(
                    agent_name="LocalRAGAgent",
                    action_input=state["query"],
                    reasoning="Simple local RAG search for relevant information",
                    confidence=0.7
                )
            ],
            parallel_capable=False,
            estimated_time=30
        )
    
    def _create_fallback_plan(self, state: EnhancedAgentState) -> str:
        """Create a simple text-based fallback plan"""
        return f"""
Fallback Plan for Query: {state['query']}

Strategy: Simple local search
Steps:
1. Search local knowledge base for relevant information
2. Synthesize response from retrieved documents

This is a simplified plan due to planning errors.
        """.strip()
    
    def _find_fallback_agent(self, invalid_agent: str) -> Optional[str]:
        """Find a fallback agent for an invalid agent name"""
        agent_mapping = {
            "local": "LocalRAGAgent",
            "web": "WebSearchAgent", 
            "search": "WebSearchAgent",
            "database": "DatabaseAgent",
            "db": "DatabaseAgent",
            "api": "APIAgent",
            "document": "DocumentAnalysisAgent",
            "doc": "DocumentAnalysisAgent"
        }
        
        invalid_lower = invalid_agent.lower()
        for key, agent in agent_mapping.items():
            if key in invalid_lower and agent in self.available_agents:
                return agent
        
        # Default fallback
        return "LocalRAGAgent" if "LocalRAGAgent" in self.available_agents else None
    
    def _build_context(self, state: EnhancedAgentState) -> str:
        """Build context string from state"""
        context_parts = []
        
        # Add conversation history
        if state["memory"].conversation_history:
            recent_history = state["memory"].conversation_history[-3:]
            context_parts.append("Recent Conversation:")
            for entry in recent_history:
                context_parts.append(f"- Q: {entry['query'][:100]}...")
        
        # Add successful queries
        if state["memory"].successful_queries:
            recent_successful = state["memory"].successful_queries[-5:]
            context_parts.append(f"Recent Successful Queries: {', '.join(recent_successful)}")
        
        # Add current context window
        if state["context_window"]:
            context_parts.append(f"Current Context: {' '.join(state['context_window'])}")
        
        return "\n".join(context_parts) if context_parts else "No prior context available."
    
    def _build_errors_context(self, state: EnhancedAgentState) -> str:
        """Build error context for learning from failures"""
        if not state["errors"]:
            return "No previous errors."
        
        recent_errors = state["errors"][-3:]
        error_context = []
        
        for error in recent_errors:
            error_context.append(f"- {error.agent}: {error.message}")
        
        return "\n".join(error_context)
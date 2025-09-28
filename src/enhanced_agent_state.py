"""
Enhanced Agent State with memory, error handling, and context management
"""
from typing import TypedDict, Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import uuid

@dataclass
class AgentMemory:
    """Memory structure for agents to maintain context across interactions"""
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_documents: List[Dict[str, Any]] = field(default_factory=list)
    successful_queries: List[str] = field(default_factory=list)
    failed_queries: List[Dict[str, Any]] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    
    def add_interaction(self, query: str, response: str, metadata: Optional[Dict] = None):
        """Add an interaction to conversation history"""
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "metadata": metadata or {}
        })
        
        # Keep only last 50 interactions to prevent memory bloat
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]
    
    def add_successful_query(self, query: str):
        """Track successful queries for learning"""
        self.successful_queries.append(query)
        if len(self.successful_queries) > 100:
            self.successful_queries = self.successful_queries[-100:]
    
    def add_failed_query(self, query: str, error: str, agent: str):
        """Track failed queries for debugging and improvement"""
        self.failed_queries.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "error": error,
            "agent": agent
        })
        if len(self.failed_queries) > 50:
            self.failed_queries = self.failed_queries[-50:]

@dataclass
class ErrorInfo:
    """Error information structure"""
    error_type: str
    message: str
    agent: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    traceback: Optional[str] = None
    recoverable: bool = True

@dataclass
class AgentExecution:
    """Information about agent execution"""
    agent_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"  # running, completed, failed, timeout
    output: Optional[Dict[str, Any]] = None
    error: Optional[ErrorInfo] = None
    
    def complete(self, output: Dict[str, Any]):
        """Mark execution as completed"""
        self.end_time = datetime.now()
        self.status = "completed"
        self.output = output
    
    def fail(self, error: ErrorInfo):
        """Mark execution as failed"""
        self.end_time = datetime.now()
        self.status = "failed"
        self.error = error
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Get execution duration in seconds"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

class EnhancedAgentState(TypedDict):
    """Enhanced state passed between all agents/nodes with comprehensive tracking"""
    # Core query and response
    query: str
    final_response: str
    
    # Planning and execution
    plan: str
    execution_strategy: str
    current_step: int
    total_steps: int
    
    # Search and retrieval
    search_results: Dict[str, str]
    retrieved_documents: List[Dict[str, Any]]
    search_queries_used: List[str]
    
    # Agent execution tracking
    agent_executions: List[AgentExecution]
    current_agent: str
    
    # Memory and context
    memory: AgentMemory
    context_window: List[str]  # Recent relevant context
    
    # Error handling
    errors: List[ErrorInfo]
    retry_count: int
    max_retries: int
    
    # Metadata and configuration
    session_id: str
    user_id: Optional[str]
    timestamp: str
    config: Dict[str, Any]
    
    # Quality and confidence
    confidence_score: float  # 0.0 to 1.0
    quality_indicators: Dict[str, Any]
    
    # Performance metrics
    total_tokens_used: int
    api_calls_made: int
    processing_time: float

def create_initial_state(
    query: str,
    user_id: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> EnhancedAgentState:
    """Create an initial agent state for a new query"""
    session_id = str(uuid.uuid4())
    
    return EnhancedAgentState(
        # Core
        query=query,
        final_response="",
        
        # Planning
        plan="",
        execution_strategy="",
        current_step=0,
        total_steps=0,
        
        # Search
        search_results={},
        retrieved_documents=[],
        search_queries_used=[],
        
        # Execution
        agent_executions=[],
        current_agent="",
        
        # Memory
        memory=AgentMemory(),
        context_window=[],
        
        # Errors
        errors=[],
        retry_count=0,
        max_retries=3,
        
        # Metadata
        session_id=session_id,
        user_id=user_id,
        timestamp=datetime.now().isoformat(),
        config=config or {},
        
        # Quality
        confidence_score=0.0,
        quality_indicators={},
        
        # Performance
        total_tokens_used=0,
        api_calls_made=0,
        processing_time=0.0
    )

def update_state_with_execution(
    state: EnhancedAgentState,
    agent_name: str,
    output: Dict[str, Any],
    error: Optional[ErrorInfo] = None
) -> EnhancedAgentState:
    """Update state with agent execution results"""
    # Find the current execution
    current_execution = None
    for execution in state["agent_executions"]:
        if execution.agent_name == agent_name and execution.status == "running":
            current_execution = execution
            break
    
    if current_execution:
        if error:
            current_execution.fail(error)
            state["errors"].append(error)
        else:
            current_execution.complete(output)
    
    return state

def add_error_to_state(
    state: EnhancedAgentState,
    error_type: str,
    message: str,
    agent: str,
    traceback: Optional[str] = None,
    recoverable: bool = True
) -> EnhancedAgentState:
    """Add an error to the state"""
    error = ErrorInfo(
        error_type=error_type,
        message=message,
        agent=agent,
        traceback=traceback,
        recoverable=recoverable
    )
    
    state["errors"].append(error)
    
    # Also add to memory for learning
    state["memory"].add_failed_query(
        query=state["query"],
        error=message,
        agent=agent
    )
    
    return state

def should_retry(state: EnhancedAgentState) -> bool:
    """Determine if we should retry based on current state"""
    if state["retry_count"] >= state["max_retries"]:
        return False
    
    # Check if we have recoverable errors
    recent_errors = [e for e in state["errors"][-3:] if e.recoverable]
    if not recent_errors:
        return False
    
    return True

def calculate_confidence_score(state: EnhancedAgentState) -> float:
    """Calculate confidence score based on various factors"""
    score = 1.0
    
    # Reduce score for errors
    error_penalty = len(state["errors"]) * 0.1
    score -= error_penalty
    
    # Reduce score for retries
    retry_penalty = state["retry_count"] * 0.05
    score -= retry_penalty
    
    # Increase score for successful retrievals
    if state["retrieved_documents"]:
        retrieval_bonus = min(len(state["retrieved_documents"]) * 0.05, 0.2)
        score += retrieval_bonus
    
    # Ensure score is between 0 and 1
    return max(0.0, min(1.0, score))

def get_state_summary(state: EnhancedAgentState) -> Dict[str, Any]:
    """Get a summary of the current state for logging/debugging"""
    return {
        "session_id": state["session_id"],
        "query": state["query"][:100] + "..." if len(state["query"]) > 100 else state["query"],
        "current_step": state["current_step"],
        "total_steps": state["total_steps"],
        "current_agent": state["current_agent"],
        "errors_count": len(state["errors"]),
        "retry_count": state["retry_count"],
        "confidence_score": state["confidence_score"],
        "documents_retrieved": len(state["retrieved_documents"]),
        "processing_time": state["processing_time"],
        "has_final_response": bool(state["final_response"])
    }
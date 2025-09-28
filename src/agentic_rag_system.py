"""
Enhanced Agentic RAG System - Main orchestrator class
Integrates all components into a cohesive system
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from langchain_ollama import OllamaLLM

from config.config import get_config
from utils.logging_setup import get_logger
from src.enhanced_agent_state import (
    EnhancedAgentState, 
    create_initial_state,
    calculate_confidence_score,
    get_state_summary
)
from src.enhanced_planning_agent import EnhancedPlanningAgent
from src.vector_store import VectorStoreFactory, VectorStore, Document
from src.data_sources import DataSourceManager
from src.retrieval_tool import EnhancedRetrievalTool
from src.enhanced_agents import (
    EnhancedLocalRAGAgent,
    EnhancedAggregatorAgent,
    WebSearchAgent,
    DatabaseAgent
)

logger = get_logger(__name__)

class AgenticRAGSystem:
    """
    Main orchestrator for the Enhanced Agentic RAG System
    Manages all agents, data sources, and system lifecycle
    """
    
    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        self.config = get_config()
        if config_override:
            # Apply configuration overrides
            for key, value in config_override.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        # Core components
        self.llm: Optional[OllamaLLM] = None
        self.vector_store: Optional[VectorStore] = None
        self.data_source_manager: Optional[DataSourceManager] = None
        
        # Agents
        self.planning_agent: Optional[EnhancedPlanningAgent] = None
        self.local_rag_agent: Optional[EnhancedLocalRAGAgent] = None
        self.aggregator_agent: Optional[EnhancedAggregatorAgent] = None
        self.web_search_agent: Optional[WebSearchAgent] = None
        self.database_agent: Optional[DatabaseAgent] = None
        
        # System state
        self.initialized = False
        self.startup_time = datetime.now()
        self.total_queries = 0
        self.active_sessions: Dict[str, EnhancedAgentState] = {}
        
        # Performance tracking
        self.performance_metrics = {
            "total_processing_time": 0.0,
            "average_response_time": 0.0,
            "successful_queries": 0,
            "failed_queries": 0,
            "agent_usage": {}
        }
        
        logger.info("AgenticRAGSystem initialized")
    
    def initialize(self) -> None:
        """Initialize all system components"""
        try:
            logger.info("Initializing Agentic RAG System...")
            
            # Initialize LLM
            self._initialize_llm()
            
            # Initialize Vector Store
            self._initialize_vector_store()
            
            # Initialize Data Sources
            self._initialize_data_sources()
            
            # Initialize Agents
            self._initialize_agents()
            
            # Load initial data
            self._load_initial_data()
            
            self.initialized = True
            logger.info("✅ Agentic RAG System initialization complete")
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            raise
    
    def _initialize_llm(self) -> None:
        """Initialize the Language Model"""
        try:
            if self.config.llm.provider == "ollama":
                self.llm = OllamaLLM(
                    model=self.config.llm.model,
                    temperature=self.config.llm.temperature,
                    num_ctx=self.config.llm.max_tokens
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {self.config.llm.provider}")
            
            logger.info(f"LLM initialized: {self.config.llm.provider}/{self.config.llm.model}")
            
        except Exception as e:
            logger.error(f"LLM initialization failed: {e}")
            raise
    
    def _initialize_vector_store(self) -> None:
        """Initialize the Vector Store"""
        try:
            self.vector_store = VectorStoreFactory.create_vector_store(
                provider=self.config.vector_db.provider,
                collection_name=self.config.vector_db.collection_name,
                persist_directory="./vector_db"
            )
            
            logger.info(f"Vector store initialized: {self.config.vector_db.provider}")
            
        except Exception as e:
            logger.error(f"Vector store initialization failed: {e}")
            raise
    
    def _initialize_data_sources(self) -> None:
        """Initialize data source manager and connectors"""
        try:
            self.data_source_manager = DataSourceManager()
            
            # Add configured data sources
            if "local_files" in self.config.data_sources.enabled_sources:
                self.data_source_manager.add_file_source(
                    data_path=self.config.data_sources.local_data_path
                )
            
            if "web_search" in self.config.data_sources.enabled_sources:
                # Web search would be configured here
                pass
            
            logger.info("Data source manager initialized")
            
        except Exception as e:
            logger.error(f"Data source initialization failed: {e}")
            raise
    
    def _initialize_agents(self) -> None:
        """Initialize all agents"""
        try:
            # Enhanced Planning Agent
            self.planning_agent = EnhancedPlanningAgent(
                llm=self.llm,
                available_agents=[
                    "LocalRAGAgent", 
                    "WebSearchAgent", 
                    "DatabaseAgent",
                    "AggregatorAgent"
                ]
            )
            
            # Enhanced Local RAG Agent
            retrieval_tool = EnhancedRetrievalTool(
                vector_store=self.vector_store,
                data_source_name="LocalKnowledgeBase"
            )
            self.local_rag_agent = EnhancedLocalRAGAgent(tool=retrieval_tool)
            
            # Enhanced Aggregator Agent
            self.aggregator_agent = EnhancedAggregatorAgent(llm=self.llm)
            
            # Additional agents (would be implemented)
            # self.web_search_agent = WebSearchAgent(...)
            # self.database_agent = DatabaseAgent(...)
            
            logger.info("All agents initialized")
            
        except Exception as e:
            logger.error(f"Agent initialization failed: {e}")
            raise
    
    def _load_initial_data(self) -> None:
        """Load initial data into the system"""
        try:
            if not self.data_source_manager:
                logger.warning("No data source manager available")
                return
            
            # Load documents from all data sources
            documents = self.data_source_manager.load_all_documents()
            
            if documents and self.vector_store:
                # Add to vector store
                self.vector_store.add_documents(documents)
                logger.info(f"Loaded {len(documents)} documents into vector store")
            else:
                logger.warning("No documents loaded or no vector store available")
                
        except Exception as e:
            logger.error(f"Initial data loading failed: {e}")
            # Don't raise - system can still function without initial data
    
    def process_query(self, initial_state: EnhancedAgentState) -> EnhancedAgentState:
        """
        Main query processing pipeline
        """
        if not self.initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        start_time = datetime.now()
        session_id = initial_state["session_id"]
        
        try:
            logger.info(f"Processing query [{session_id}]: {initial_state['query'][:100]}...")
            
            # Track session
            self.active_sessions[session_id] = initial_state
            self.total_queries += 1
            
            # Step 1: Planning
            state = self._execute_planning(initial_state)
            
            # Step 2: Execute retrieval agents based on plan
            state = self._execute_retrieval_agents(state)
            
            # Step 3: Aggregate results
            state = self._execute_aggregation(state)
            
            # Step 4: Post-processing and quality assurance
            state = self._post_process_response(state)
            
            # Update performance metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_metrics(True, processing_time)
            
            state["processing_time"] = processing_time
            state["confidence_score"] = calculate_confidence_score(state)
            
            logger.info(f"✅ Query processed successfully [{session_id}] in {processing_time:.2f}s")
            
            return state
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_metrics(False, processing_time)
            
            logger.error(f"❌ Query processing failed [{session_id}]: {e}")
            
            # Create error response
            initial_state["final_response"] = f"I apologize, but I encountered an error processing your query: {str(e)}"
            initial_state["confidence_score"] = 0.0
            initial_state["processing_time"] = processing_time
            
            return initial_state
            
        finally:
            # Clean up session
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
    
    def _execute_planning(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Execute the planning phase"""
        if not self.planning_agent:
            raise RuntimeError("Planning agent not initialized")
        
        result = self.planning_agent.run(state)
        
        # Merge result into state
        for key, value in result.items():
            state[key] = value
        
        return state
    
    def _execute_retrieval_agents(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Execute retrieval agents based on the plan"""
        try:
            # For now, execute Local RAG Agent
            if self.local_rag_agent:
                result = self.local_rag_agent.run(state)
                
                # Merge results
                for key, value in result.items():
                    if key in state:
                        if isinstance(state[key], dict) and isinstance(value, dict):
                            state[key].update(value)
                        else:
                            state[key] = value
                    else:
                        state[key] = value
            
            return state
            
        except Exception as e:
            logger.error(f"Retrieval execution failed: {e}")
            # Add fallback behavior
            state["search_results"]["fallback"] = f"Retrieval failed: {str(e)}"
            return state
    
    def _execute_aggregation(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Execute the aggregation phase"""
        if not self.aggregator_agent:
            raise RuntimeError("Aggregator agent not initialized")
        
        result = self.aggregator_agent.run(state)
        
        # Merge result into state
        for key, value in result.items():
            state[key] = value
        
        return state
    
    def _post_process_response(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Post-process the final response"""
        try:
            # Add metadata to response
            if not state["final_response"]:
                state["final_response"] = "I apologize, but I couldn't generate a response to your query."
            
            # Add confidence and quality indicators
            state["quality_indicators"] = {
                "has_sources": bool(state.get("retrieved_documents")),
                "response_length": len(state["final_response"]),
                "processing_steps": state.get("current_step", 0)
            }
            
            # Add session to memory
            state["memory"].add_interaction(
                query=state["query"],
                response=state["final_response"],
                metadata=get_state_summary(state)
            )
            
            return state
            
        except Exception as e:
            logger.error(f"Post-processing failed: {e}")
            return state
    
    def _update_performance_metrics(self, success: bool, processing_time: float):
        """Update system performance metrics"""
        try:
            if success:
                self.performance_metrics["successful_queries"] += 1
            else:
                self.performance_metrics["failed_queries"] += 1
            
            self.performance_metrics["total_processing_time"] += processing_time
            
            total_queries = (
                self.performance_metrics["successful_queries"] + 
                self.performance_metrics["failed_queries"]
            )
            
            if total_queries > 0:
                self.performance_metrics["average_response_time"] = (
                    self.performance_metrics["total_processing_time"] / total_queries
                )
                
        except Exception as e:
            logger.error(f"Performance metrics update failed: {e}")
    
    def refresh_data_sources(self) -> None:
        """Refresh all data sources"""
        try:
            logger.info("Refreshing data sources...")
            
            if self.data_source_manager and self.vector_store:
                # Reload documents
                documents = self.data_source_manager.load_all_documents()
                
                # Clear and reload vector store (in production, you'd want incremental updates)
                # This is a simplified approach
                if documents:
                    self.vector_store.add_documents(documents)
                    logger.info(f"Refreshed {len(documents)} documents")
                else:
                    logger.warning("No documents to refresh")
            
        except Exception as e:
            logger.error(f"Data source refresh failed: {e}")
            raise
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        return {
            "status": "running" if self.initialized else "initializing",
            "startup_time": self.startup_time.isoformat(),
            "uptime_seconds": (datetime.now() - self.startup_time).total_seconds(),
            "total_queries": self.total_queries,
            "active_sessions": len(self.active_sessions),
            "performance_metrics": self.performance_metrics,
            "configuration": {
                "llm_provider": self.config.llm.provider,
                "llm_model": self.config.llm.model,
                "vector_db_provider": self.config.vector_db.provider,
                "enabled_data_sources": self.config.data_sources.enabled_sources
            }
        }
    
    def cleanup(self) -> None:
        """Cleanup system resources"""
        try:
            logger.info("Cleaning up Agentic RAG System...")
            
            # Clear active sessions
            self.active_sessions.clear()
            
            # Additional cleanup would go here
            # (close database connections, save state, etc.)
            
            logger.info("System cleanup complete")
            
        except Exception as e:
            logger.error(f"System cleanup failed: {e}")

# Convenience function for simple usage
def create_rag_system(config_path: Optional[str] = None) -> AgenticRAGSystem:
    """Create and initialize a RAG system"""
    system = AgenticRAGSystem()
    system.initialize()
    return system
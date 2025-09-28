"""
FastAPI web interface for PyAgenticRAG
Provides REST API endpoints for the Agentic RAG system
"""
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from config.config import get_config
from utils.logging_setup import setup_logging, get_logger
from src.enhanced_agent_state import create_initial_state, get_state_summary
from src.agentic_rag_system import AgenticRAGSystem

# Initialize logging
config = get_config()
setup_logging(
    level=config.logging.level,
    file_path=config.logging.file_path,
    format_string=config.logging.format
)
logger = get_logger(__name__)

# Pydantic models for API
class QueryRequest(BaseModel):
    """Request model for RAG queries"""
    query: str = Field(..., description="The user query")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    session_id: Optional[str] = Field(None, description="Optional session identifier")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    streaming: bool = Field(False, description="Whether to stream the response")

class QueryResponse(BaseModel):
    """Response model for RAG queries"""
    session_id: str
    query: str
    response: str
    confidence_score: float
    processing_time: float
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class SystemStatus(BaseModel):
    """System status response model"""
    status: str
    version: str
    uptime: str
    total_queries: int
    active_sessions: int
    vector_db_status: str
    agent_status: Dict[str, str]

class DataSourceInfo(BaseModel):
    """Data source information model"""
    type: str
    status: str
    document_count: int
    last_updated: str
    metadata: Dict[str, Any]

# Global system instance
rag_system: Optional[AgenticRAGSystem] = None
start_time = datetime.now()
query_count = 0
active_sessions: Dict[str, Dict] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global rag_system
    
    # Startup
    logger.info("Starting PyAgenticRAG API server...")
    
    try:
        # Initialize the RAG system
        rag_system = AgenticRAGSystem()
        await asyncio.get_event_loop().run_in_executor(None, rag_system.initialize)
        logger.info("RAG system initialized successfully")
        
        yield
        
    finally:
        # Shutdown
        logger.info("Shutting down PyAgenticRAG API server...")
        if rag_system:
            await asyncio.get_event_loop().run_in_executor(None, rag_system.cleanup)

# Create FastAPI app
app = FastAPI(
    title="PyAgenticRAG API",
    description="Advanced Agentic RAG system with multi-agent orchestration",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_rag_system() -> AgenticRAGSystem:
    """Dependency to get RAG system instance"""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    return rag_system

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic info"""
    return {
        "name": "PyAgenticRAG API",
        "version": "1.0.0",
        "description": "Advanced Agentic RAG system",
        "docs_url": "/docs"
    }

@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint"""
    try:
        # Basic health checks
        system_healthy = rag_system is not None
        
        return {
            "status": "healthy" if system_healthy else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "uptime": str(datetime.now() - start_time)
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="System unhealthy")

@app.get("/status", response_model=SystemStatus)
async def get_system_status(system: AgenticRAGSystem = Depends(get_rag_system)):
    """Get detailed system status"""
    try:
        uptime = str(datetime.now() - start_time)
        
        # Get agent status
        agent_status = {}
        for agent_name in ["PlanningAgent", "LocalRAGAgent", "AggregatorAgent"]:
            agent_status[agent_name] = "active"  # In real implementation, check actual status
        
        # Get vector DB status
        vector_db_status = "active"  # In real implementation, ping vector DB
        
        return SystemStatus(
            status="running",
            version="1.0.0",
            uptime=uptime,
            total_queries=query_count,
            active_sessions=len(active_sessions),
            vector_db_status=vector_db_status,
            agent_status=agent_status
        )
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system status")

@app.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    system: AgenticRAGSystem = Depends(get_rag_system)
):
    """Process a RAG query"""
    global query_count
    
    try:
        query_count += 1
        start_time_query = datetime.now()
        
        logger.info(f"Processing query: {request.query[:100]}...")
        
        # Create initial state
        initial_state = create_initial_state(
            query=request.query,
            user_id=request.user_id,
            config=request.context
        )
        
        # Track session
        session_id = initial_state["session_id"]
        active_sessions[session_id] = {
            "user_id": request.user_id,
            "start_time": start_time_query,
            "query": request.query
        }
        
        # Process query
        final_state = await asyncio.get_event_loop().run_in_executor(
            None, system.process_query, initial_state
        )
        
        processing_time = (datetime.now() - start_time_query).total_seconds()
        
        # Extract sources from retrieved documents
        sources = []
        for doc in final_state.get("retrieved_documents", []):
            sources.append({
                "id": doc.get("id", ""),
                "source": doc.get("metadata", {}).get("source", "unknown"),
                "relevance_score": doc.get("score", 0.0)
            })
        
        # Create response
        response = QueryResponse(
            session_id=session_id,
            query=request.query,
            response=final_state["final_response"],
            confidence_score=final_state.get("confidence_score", 0.0),
            processing_time=processing_time,
            sources=sources,
            metadata={
                "total_steps": final_state.get("total_steps", 0),
                "errors_count": len(final_state.get("errors", [])),
                "retry_count": final_state.get("retry_count", 0),
                "execution_strategy": final_state.get("execution_strategy", "unknown")
            }
        )
        
        # Clean up session
        background_tasks.add_task(cleanup_session, session_id)
        
        logger.info(f"Query processed successfully in {processing_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/query/{session_id}/status")
async def get_query_status(session_id: str):
    """Get status of a specific query session"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    processing_time = (datetime.now() - session["start_time"]).total_seconds()
    
    return {
        "session_id": session_id,
        "status": "processing",
        "query": session["query"],
        "processing_time": processing_time,
        "user_id": session.get("user_id")
    }

@app.get("/data-sources", response_model=List[DataSourceInfo])
async def get_data_sources(system: AgenticRAGSystem = Depends(get_rag_system)):
    """Get information about available data sources"""
    try:
        # In real implementation, get from system
        sources = [
            DataSourceInfo(
                type="local_files",
                status="active",
                document_count=100,  # placeholder
                last_updated=datetime.now().isoformat(),
                metadata={"path": "./data", "extensions": [".txt", ".pdf", ".docx"]}
            )
        ]
        
        return sources
        
    except Exception as e:
        logger.error(f"Error getting data sources: {e}")
        raise HTTPException(status_code=500, detail="Failed to get data sources")

@app.post("/data-sources/refresh")
async def refresh_data_sources(
    background_tasks: BackgroundTasks,
    system: AgenticRAGSystem = Depends(get_rag_system)
):
    """Refresh/reload data sources"""
    try:
        background_tasks.add_task(system.refresh_data_sources)
        
        return {
            "message": "Data source refresh initiated",
            "status": "in_progress"
        }
        
    except Exception as e:
        logger.error(f"Error refreshing data sources: {e}")
        raise HTTPException(status_code=500, detail="Failed to refresh data sources")

@app.get("/metrics")
async def get_metrics():
    """Get system metrics for monitoring"""
    try:
        uptime_seconds = (datetime.now() - start_time).total_seconds()
        
        metrics = {
            "uptime_seconds": uptime_seconds,
            "total_queries": query_count,
            "active_sessions": len(active_sessions),
            "queries_per_minute": query_count / max(uptime_seconds / 60, 1),
            "memory_usage": "N/A",  # Would need psutil for real metrics
            "cpu_usage": "N/A"
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get metrics")

async def cleanup_session(session_id: str):
    """Clean up completed session"""
    await asyncio.sleep(300)  # Keep session info for 5 minutes
    if session_id in active_sessions:
        del active_sessions[session_id]
        logger.info(f"Cleaned up session: {session_id}")

def main():
    """Main function to run the API server"""
    config = get_config()
    
    uvicorn.run(
        "api.web_interface:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.debug,
        log_level=config.logging.level.lower()
    )

if __name__ == "__main__":
    main()
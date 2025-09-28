"""
Enhanced Agents with improved error handling, memory, and capabilities
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from langchain_core.prompts import ChatPromptTemplate

from src.enhanced_agent_state import (
    EnhancedAgentState, 
    AgentExecution, 
    ErrorInfo,
    add_error_to_state,
    update_state_with_execution
)
from src.retrieval_tool import EnhancedRetrievalTool

logger = logging.getLogger(__name__)

class EnhancedLocalRAGAgent:
    """
    Enhanced Local RAG Agent with improved retrieval capabilities,
    error handling, and context awareness
    """
    
    def __init__(self, tool: EnhancedRetrievalTool):
        self.tool = tool
        self.agent_name = "LocalRAGAgent"
        
        # Performance tracking
        self.query_history = []
        self.successful_queries = 0
        self.failed_queries = 0
        
        logger.info(f"Enhanced Local RAG Agent initialized")
    
    def run(self, state: EnhancedAgentState) -> Dict[str, Any]:
        """Execute the local RAG retrieval with enhanced capabilities"""
        execution = AgentExecution(
            agent_name=self.agent_name,
            start_time=datetime.now()
        )
        state["agent_executions"].append(execution)
        state["current_agent"] = self.agent_name
        
        try:
            logger.info(f"🔎 Local RAG Agent processing: {state['query'][:100]}...")
            
            # Extract search terms from query and plan
            search_terms = self._extract_search_terms(state)
            
            # Perform multiple searches if needed
            all_results = []
            search_results = {}
            
            for search_term in search_terms:
                try:
                    retrieved_info = self.tool.retrieve(
                        search_term=search_term,
                        top_k=5
                    )
                    
                    if retrieved_info and "Retrieval error" not in retrieved_info:
                        search_results[search_term] = retrieved_info
                        all_results.append({
                            "search_term": search_term,
                            "content": retrieved_info,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        # Track in search history
                        state["search_queries_used"].append(search_term)
                        
                        logger.info(f"✅ Retrieved content for: {search_term}")
                    else:
                        logger.warning(f"⚠️ No results for: {search_term}")
                        
                except Exception as e:
                    logger.error(f"❌ Search failed for '{search_term}': {e}")
                    search_results[search_term] = f"Search failed: {str(e)}"
            
            # Combine and enhance results
            if all_results:
                # Update state with successful retrieval
                current_results = state.get("search_results", {})
                current_results.update(search_results)
                
                # Add to retrieved documents with metadata
                for result in all_results:
                    state["retrieved_documents"].append({
                        "id": f"local_rag_{len(state['retrieved_documents'])}",
                        "content": result["content"],
                        "metadata": {
                            "agent": self.agent_name,
                            "search_term": result["search_term"],
                            "retrieved_at": result["timestamp"],
                            "source_type": "local_knowledge_base"
                        },
                        "score": 0.8  # Default score for local RAG
                    })
                
                self.successful_queries += 1
                
                output = {
                    "search_results": current_results,
                    "retrieved_documents": state["retrieved_documents"],
                    "agent_metadata": {
                        "searches_performed": len(search_terms),
                        "successful_searches": len(all_results),
                        "agent": self.agent_name
                    }
                }
                
                execution.complete(output)
                
                logger.info(f"🎯 Local RAG Agent completed: {len(all_results)} successful searches")
                
                return output
            
            else:
                # No results found
                error = ErrorInfo(
                    error_type="NoResultsError",
                    message="No relevant information found in local knowledge base",
                    agent=self.agent_name,
                    recoverable=True
                )
                
                execution.fail(error)
                add_error_to_state(state, error.error_type, error.message, error.agent)
                
                self.failed_queries += 1
                
                return {
                    "search_results": {"local_rag_result": "No relevant local information found."},
                    "retrieved_documents": state.get("retrieved_documents", [])
                }
                
        except Exception as e:
            error = ErrorInfo(
                error_type="LocalRAGError",
                message=f"Local RAG processing failed: {str(e)}",
                agent=self.agent_name,
                recoverable=True
            )
            
            execution.fail(error)
            add_error_to_state(state, error.error_type, error.message, error.agent)
            
            self.failed_queries += 1
            
            logger.error(f"❌ Local RAG Agent failed: {e}")
            
            return {
                "search_results": {"local_rag_result": f"Local search failed: {str(e)}"},
                "retrieved_documents": state.get("retrieved_documents", [])
            }
    
    def _extract_search_terms(self, state: EnhancedAgentState) -> List[str]:
        """Extract relevant search terms from the query and plan"""
        search_terms = []
        
        # Primary search term is the original query
        search_terms.append(state["query"])
        
        # Try to extract additional search terms from the plan
        plan = state.get("plan", "")
        if plan:
            try:
                # Look for Action Input in the plan
                if "Action Input:" in plan:
                    action_input = plan.split("Action Input:")[-1].split("\n")[0].strip()
                    if action_input and action_input != state["query"]:
                        search_terms.append(action_input)
                
                # Extract key phrases (simple keyword extraction)
                keywords = self._extract_keywords(state["query"])
                search_terms.extend(keywords[:2])  # Add top 2 keywords
                
            except Exception as e:
                logger.warning(f"Failed to extract search terms from plan: {e}")
        
        # Remove duplicates and empty terms
        unique_terms = []
        for term in search_terms:
            if term and term.strip() and term not in unique_terms:
                unique_terms.append(term.strip())
        
        return unique_terms[:3]  # Limit to 3 search terms
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Simple keyword extraction from query"""
        # Remove common stop words and extract meaningful terms
        stop_words = {'the', 'is', 'at', 'which', 'on', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by', 'about', 'what', 'how', 'when', 'where', 'why', 'who'}
        
        words = query.lower().split()
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        
        return keywords[:5]  # Return top 5 keywords
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        total_queries = self.successful_queries + self.failed_queries
        success_rate = (self.successful_queries / total_queries) if total_queries > 0 else 0
        
        return {
            "agent_name": self.agent_name,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "success_rate": success_rate,
            "total_queries": total_queries
        }

class EnhancedAggregatorAgent:
    """
    Enhanced Aggregator Agent with sophisticated synthesis capabilities,
    quality assessment, and error handling
    """
    
    def __init__(self, llm):
        self.llm = llm
        self.agent_name = "AggregatorAgent"
        
        # Create sophisticated prompts
        self.synthesis_prompt = self._create_synthesis_prompt()
        self.quality_assessment_prompt = self._create_quality_prompt()
        
        # Performance tracking
        self.successful_syntheses = 0
        self.failed_syntheses = 0
        
        logger.info("Enhanced Aggregator Agent initialized")
    
    def _create_synthesis_prompt(self) -> ChatPromptTemplate:
        """Create the main synthesis prompt"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are an advanced AI synthesis engine responsible for creating comprehensive, accurate, and helpful responses.

Your task is to:
1. Analyze all retrieved information from multiple sources
2. Synthesize a coherent, well-structured response
3. Ensure accuracy and relevance to the user's query
4. Cite sources when appropriate
5. Acknowledge limitations or gaps in available information

Guidelines:
- Be thorough but concise
- Maintain objectivity
- Use clear, professional language
- Structure your response logically
- If information is conflicting, acknowledge this
- If information is insufficient, state this clearly"""),
            
            ("human", """Original Query: {query}

Execution Plan Context: {plan}

Retrieved Information:
{full_context}

Previous Errors (if any): {errors}

Please synthesize a comprehensive response that directly addresses the user's query using the retrieved information.""")
        ])
    
    def _create_quality_prompt(self) -> ChatPromptTemplate:
        """Create quality assessment prompt"""
        return ChatPromptTemplate.from_messages([
            ("system", """Evaluate the quality of the following response. Provide a score from 0.0 to 1.0 and brief reasoning.

Criteria:
- Relevance to the original query
- Accuracy based on provided sources
- Completeness of information
- Clarity and structure
- Appropriate citations/acknowledgments

Respond with: SCORE: [0.0-1.0] | REASONING: [brief explanation]"""),
            
            ("human", "Query: {query}\n\nResponse: {response}\n\nSources: {sources}")
        ])
    
    def run(self, state: EnhancedAgentState) -> Dict[str, Any]:
        """Generate the final synthesized response"""
        execution = AgentExecution(
            agent_name=self.agent_name,
            start_time=datetime.now()
        )
        state["agent_executions"].append(execution)
        state["current_agent"] = self.agent_name
        
        try:
            logger.info(f"✨ Aggregator Agent synthesizing response...")
            
            # Prepare context from all sources
            full_context = self._prepare_context(state)
            errors_context = self._prepare_errors_context(state)
            
            # Generate the response
            final_response = self._generate_response(state, full_context, errors_context)
            
            # Assess quality
            quality_score = self._assess_quality(state, final_response, full_context)
            
            # Enhance the response if needed
            enhanced_response = self._enhance_response(final_response, quality_score, state)
            
            output = {
                "final_response": enhanced_response,
                "quality_score": quality_score,
                "synthesis_metadata": {
                    "sources_used": len(state.get("retrieved_documents", [])),
                    "context_length": len(full_context),
                    "has_errors": len(state.get("errors", [])) > 0,
                    "agent": self.agent_name
                }
            }
            
            execution.complete(output)
            self.successful_syntheses += 1
            
            logger.info(f"🎯 Aggregator Agent completed with quality score: {quality_score:.2f}")
            
            return output
            
        except Exception as e:
            error = ErrorInfo(
                error_type="AggregationError",
                message=f"Response synthesis failed: {str(e)}",
                agent=self.agent_name,
                recoverable=True
            )
            
            execution.fail(error)
            add_error_to_state(state, error.error_type, error.message, error.agent)
            
            self.failed_syntheses += 1
            
            logger.error(f"❌ Aggregator Agent failed: {e}")
            
            # Provide fallback response
            fallback_response = self._create_fallback_response(state)
            
            return {
                "final_response": fallback_response,
                "quality_score": 0.3,
                "synthesis_metadata": {
                    "fallback_used": True,
                    "error": str(e)
                }
            }
    
    def _prepare_context(self, state: EnhancedAgentState) -> str:
        """Prepare comprehensive context from all sources"""
        context_parts = []
        
        # Add retrieved documents
        retrieved_docs = state.get("retrieved_documents", [])
        if retrieved_docs:
            context_parts.append("=== RETRIEVED INFORMATION ===")
            for i, doc in enumerate(retrieved_docs, 1):
                source = doc.get("metadata", {}).get("source", "Unknown")
                content = doc.get("content", "")[:1000]  # Limit length
                score = doc.get("score", 0.0)
                
                context_parts.append(f"\n[Source {i}] (Relevance: {score:.2f}) - {source}:")
                context_parts.append(content)
        
        # Add search results (legacy format)
        search_results = state.get("search_results", {})
        if search_results:
            context_parts.append("\n=== ADDITIONAL SEARCH RESULTS ===")
            for source, content in search_results.items():
                if isinstance(content, str):
                    context_parts.append(f"\n{source}: {content[:500]}")
        
        return "\n".join(context_parts) if context_parts else "No relevant information available."
    
    def _prepare_errors_context(self, state: EnhancedAgentState) -> str:
        """Prepare error context for transparency"""
        errors = state.get("errors", [])
        if not errors:
            return "No errors encountered."
        
        error_summary = []
        for error in errors[-3:]:  # Last 3 errors
            error_summary.append(f"- {error.agent}: {error.message}")
        
        return "Errors encountered:\n" + "\n".join(error_summary)
    
    def _generate_response(self, state: EnhancedAgentState, full_context: str, errors_context: str) -> str:
        """Generate the main response"""
        try:
            chain = self.synthesis_prompt | self.llm
            
            response = chain.invoke({
                "query": state["query"],
                "plan": state.get("plan", "No plan available"),
                "full_context": full_context,
                "errors": errors_context
            })
            
            return str(response)
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return f"I apologize, but I encountered an error while generating a response: {str(e)}"
    
    def _assess_quality(self, state: EnhancedAgentState, response: str, context: str) -> float:
        """Assess the quality of the generated response"""
        try:
            # Simple quality assessment based on response characteristics
            quality_score = 0.5  # Base score
            
            # Length check
            if len(response) > 100:
                quality_score += 0.1
            if len(response) > 300:
                quality_score += 0.1
            
            # Source utilization
            if len(state.get("retrieved_documents", [])) > 0:
                quality_score += 0.2
            
            # Error penalty
            error_count = len(state.get("errors", []))
            quality_score -= (error_count * 0.1)
            
            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return 0.5
    
    def _enhance_response(self, response: str, quality_score: float, state: EnhancedAgentState) -> str:
        """Enhance the response based on quality assessment"""
        enhanced_response = response
        
        # Add disclaimers if quality is low
        if quality_score < 0.4:
            enhanced_response += "\n\nNote: This response may be incomplete due to limited available information."
        
        # Add source information if available
        sources_count = len(state.get("retrieved_documents", []))
        if sources_count > 0:
            enhanced_response += f"\n\n*Based on information from {sources_count} source(s) in the knowledge base.*"
        
        return enhanced_response
    
    def _create_fallback_response(self, state: EnhancedAgentState) -> str:
        """Create a fallback response when synthesis fails"""
        query = state["query"]
        
        return f"""I apologize, but I encountered difficulties processing your query: "{query}"

While I attempted to retrieve and analyze relevant information, technical issues prevented me from generating a complete response. 

To get better assistance, you might try:
- Rephrasing your question with more specific terms
- Breaking complex queries into smaller parts
- Checking if the information you're looking for might be available in our knowledge base

Please feel free to try again with a different approach to your question."""

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregator statistics"""
        total_syntheses = self.successful_syntheses + self.failed_syntheses
        success_rate = (self.successful_syntheses / total_syntheses) if total_syntheses > 0 else 0
        
        return {
            "agent_name": self.agent_name,
            "successful_syntheses": self.successful_syntheses,
            "failed_syntheses": self.failed_syntheses,
            "success_rate": success_rate,
            "total_syntheses": total_syntheses
        }

# Placeholder classes for additional agents
class WebSearchAgent:
    """Placeholder for web search agent"""
    def __init__(self):
        self.agent_name = "WebSearchAgent"
        logger.info("Web Search Agent placeholder initialized")
    
    def run(self, state: EnhancedAgentState) -> Dict[str, Any]:
        return {"search_results": {"web_search": "Web search not yet implemented"}}

class DatabaseAgent:
    """Placeholder for database agent"""
    def __init__(self):
        self.agent_name = "DatabaseAgent"
        logger.info("Database Agent placeholder initialized")
    
    def run(self, state: EnhancedAgentState) -> Dict[str, Any]:
        return {"search_results": {"database": "Database search not yet implemented"}}
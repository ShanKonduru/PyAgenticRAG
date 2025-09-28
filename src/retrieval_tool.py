"""
Enhanced Retrieval Tool with better search capabilities and error handling
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# --- B. Retrieval Tool Class (Encapsulates Data Access) ---

class RetrievalTool:
    """Legacy retrieval tool - kept for backward compatibility"""

    def __init__(self, data_source_name: str):
        self.data_source_name = data_source_name
        logger.info(f"Tool initialized for: {data_source_name}")

    def retrieve(self, search_term: str) -> str:
        """Simulates a search/retrieval operation on the specific data source."""
        if "policy" in search_term.lower() and self.data_source_name == "Local Data":
            return "Local Data: Policy 2.1 requires all sensitive data to be encrypted at rest."
        elif "cloud" in search_term.lower() and self.data_source_name == "Cloud Servers":
            return "Cloud Servers: AWS S3 logs show the last successful backup was 2 hours ago."
        else:
            return f"No specific document found for '{search_term}' in {self.data_source_name}."

class EnhancedRetrievalTool:
    """
    Enhanced retrieval tool with improved search capabilities,
    caching, and error handling
    """
    
    def __init__(
        self, 
        vector_store = None,
        data_source_name: str = "Enhanced Knowledge Base"
    ):
        self.vector_store = vector_store
        self.data_source_name = data_source_name
        
        # Query cache for performance
        self._query_cache: Dict[str, List] = {}
        self._cache_max_size = 100
        
        # Search configuration
        self.default_top_k = 5
        self.similarity_threshold = 0.3
        
        logger.info(f"Enhanced Retrieval Tool initialized for: {data_source_name}")
    
    def retrieve(
        self, 
        search_term: str, 
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> str:
        """Enhanced retrieval with caching and filtering"""
        try:
            # Fallback to simple search if no vector store
            if not self.vector_store:
                return self._simple_search(search_term)
            
            top_k = top_k or self.default_top_k
            
            # Check cache first
            cache_key = f"{search_term}_{top_k}_{str(filters)}"
            if use_cache and cache_key in self._query_cache:
                logger.info(f"Cache hit for query: {search_term[:50]}...")
                results = self._query_cache[cache_key]
            else:
                # Perform search
                results = self.vector_store.search(search_term, top_k)
                
                # Filter results by similarity threshold
                results = [r for r in results if r.score >= self.similarity_threshold]
                
                # Cache results
                if use_cache:
                    self._update_cache(cache_key, results)
            
            # Combine results into text
            combined_text = self._combine_results(results, search_term)
            
            logger.info(f"Retrieved {len(results)} documents for: {search_term[:50]}...")
            
            return combined_text
            
        except Exception as e:
            logger.error(f"Retrieval failed for '{search_term}': {e}")
            return f"Retrieval error: Unable to search for '{search_term}'. Error: {str(e)}"
    
    def _simple_search(self, search_term: str) -> str:
        """Simple search fallback when no vector store is available"""
        if "policy" in search_term.lower():
            return f"{self.data_source_name}: Policy 2.1 requires all sensitive data to be encrypted at rest."
        elif "cloud" in search_term.lower():
            return f"{self.data_source_name}: Cloud backup logs show successful operations within the last 2 hours."
        else:
            return f"{self.data_source_name}: Found general information related to '{search_term}'."
    
    def _combine_results(self, results: List, search_term: str) -> str:
        """Combine search results into formatted text"""
        if not results:
            return f"No relevant information found for '{search_term}' in {self.data_source_name}."
        
        combined_parts = []
        combined_parts.append(f"Retrieved information for '{search_term}' from {self.data_source_name}:")
        
        for i, result in enumerate(results[:self.default_top_k], 1):
            # Handle different result formats
            if hasattr(result, 'document') and hasattr(result, 'score'):
                content = result.document.content
                score = result.score
                source = getattr(result.document, 'metadata', {}).get("source", "Unknown")
            else:
                # Fallback for simple results
                content = str(result)
                score = 1.0
                source = "Unknown"
            
            # Truncate very long content
            if len(content) > 500:
                content = content[:500] + "..."
            
            combined_parts.append(
                f"\n[Source {i}] (Relevance: {score:.2f}) {source}:\n{content}"
            )
        
        return "\n".join(combined_parts)
    
    def _update_cache(self, cache_key: str, results: List):
        """Update the query cache with new results"""
        try:
            # Remove oldest entries if cache is full
            if len(self._query_cache) >= self._cache_max_size:
                oldest_key = next(iter(self._query_cache))
                del self._query_cache[oldest_key]
            
            self._query_cache[cache_key] = results
            
        except Exception as e:
            logger.error(f"Cache update failed: {e}")
    
    def clear_cache(self):
        """Clear the query cache"""
        self._query_cache.clear()
        logger.info("Query cache cleared")

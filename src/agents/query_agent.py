"""
Query agent for handling user queries and orchestrating retrieval.
"""
from typing import List, Dict, Any, Optional
import logging

from ..rag_utils import query_context

logger = logging.getLogger(__name__)


class QueryAgent:
    """Agent responsible for processing user queries and retrieving relevant context."""
    
    def __init__(
        self,
        embeddings_model: str,
        index,
        namespace: str = "default",
        top_k: int = 5
    ):
        """
        Initialize QueryAgent.
        
        Args:
            embeddings_model: Name of embeddings model
            index: Pinecone index object
            namespace: Namespace to query
            top_k: Number of results to return
        """
        self.embeddings_model = embeddings_model
        self.index = index
        self.namespace = namespace
        self.top_k = top_k
    
    def query(self, query_text: str) -> List[Dict[str, Any]]:
        """
        Query for relevant context.
        
        Args:
            query_text: User query text
            
        Returns:
            List of relevant context dictionaries
        """
        try:
            contexts = query_context(
                query_text=query_text,
                embeddings_model=self.embeddings_model,
                index=self.index,
                namespace=self.namespace,
                top_k=self.top_k
            )
            
            logger.info(f"Retrieved {len(contexts)} contexts for query")
            return contexts
            
        except Exception as e:
            logger.error(f"Error querying context: {e}")
            return []
    
    def format_contexts(self, contexts: List[Dict[str, Any]]) -> str:
        """
        Format retrieved contexts for prompt inclusion.
        
        Args:
            contexts: List of context dictionaries
            
        Returns:
            Formatted context string
        """
        if not contexts:
            return "No relevant context found."
        
        formatted_contexts = []
        for i, context in enumerate(contexts, 1):
            formatted_contexts.append(
                f"Context {i} (from {context['filename']}):\n{context['text']}"
            )
        
        return "\n\n".join(formatted_contexts)

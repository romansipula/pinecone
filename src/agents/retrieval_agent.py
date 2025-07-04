"""
Retrieval agent for managing document retrieval operations.
"""
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import logging

from ..rag_utils import query_context

if TYPE_CHECKING:
    from pinecone import Index

logger = logging.getLogger(__name__)


class RetrievalAgent:
    """Agent responsible for retrieving relevant documents from the vector database."""
    
    def __init__(
        self,
        embeddings_model: str,
        index: "Index",
        namespace: str = "default"
    ):
        """
        Initialize RetrievalAgent.
        
        Args:
            embeddings_model: Name of embeddings model
            index: Pinecone index object
            namespace: Namespace to query
        """
        self.embeddings_model = embeddings_model
        self.index = index
        self.namespace = namespace
    
    def retrieve(
        self,
        query_text: str,
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant contexts.
        
        Args:
            query_text: Query text
            top_k: Number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of relevant context dictionaries
        """
        try:
            contexts = query_context(
                query_text=query_text,
                embeddings_model=self.embeddings_model,
                index=self.index,
                namespace=self.namespace,
                top_k=top_k
            )
            
            # Filter by score threshold
            filtered_contexts = [
                ctx for ctx in contexts
                if ctx['score'] >= score_threshold
            ]
            
            logger.info(
                f"Retrieved {len(filtered_contexts)} contexts "
                f"(filtered from {len(contexts)} by score threshold {score_threshold})"
            )
            
            return filtered_contexts
            
        except Exception as e:
            logger.error(f"Error retrieving contexts: {e}")
            return []
    
    def get_context_summary(self, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary statistics of retrieved contexts.
        
        Args:
            contexts: List of context dictionaries
            
        Returns:
            Summary statistics
        """
        if not contexts:
            return {"total": 0, "avg_score": 0.0, "sources": []}
        
        total_contexts = len(contexts)
        avg_score = sum(ctx['score'] for ctx in contexts) / total_contexts
        sources = list(set(ctx['filename'] for ctx in contexts))
        
        return {
            "total": total_contexts,
            "avg_score": avg_score,
            "sources": sources,
            "score_range": {
                "min": min(ctx['score'] for ctx in contexts),
                "max": max(ctx['score'] for ctx in contexts)
            }
        }

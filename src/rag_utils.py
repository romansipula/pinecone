"""
RAG utilities for Pinecone vector database operations and document processing.
"""
import os
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import logging
from pathlib import Path

from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from pypdf import PdfReader

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    # Mock SentenceTransformer if not available
    class SentenceTransformer:
        def __init__(self, model_name):
            self.model_name = model_name
        
        def encode(self, texts):
            # Return dummy embeddings for testing
            import random
            if isinstance(texts, str):
                texts = [texts]
            return [[random.random() for _ in range(1536)] for _ in texts]

if TYPE_CHECKING:
    from pinecone import Index

logger = logging.getLogger(__name__)


def init_pinecone(
    api_key: str,
    environment: str = None,
    index_name: str = None,
    dimension: int = 1536,
    metric: str = "cosine"
) -> "Index":
    """
    Initialize Pinecone client and create/connect to index.
    
    Args:
        api_key: Pinecone API key
        environment: Pinecone environment (deprecated in new API)
        index_name: Name of the index
        dimension: Vector dimension
        metric: Distance metric for similarity search
        
    Returns:
        Pinecone index object
    """
    pc = Pinecone(api_key=api_key)
    
    if index_name and not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        logger.info(f"Created new index: {index_name}")
    
    return pc.Index(index_name) if index_name else None


def ingest_documents(
    directory: str,
    embeddings_model: str,
    index: "Index",
    namespace: str = "default",
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> None:
    """
    Ingest documents from directory into Pinecone index.
    
    Args:
        directory: Path to directory containing documents
        embeddings_model: Name of embeddings model to use
        index: Pinecone index object
        namespace: Namespace for the vectors
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
    """
    model = SentenceTransformer(embeddings_model)
    documents = []
    
    # Load documents from directory
    for file_path in Path(directory).glob("**/*"):
        if file_path.suffix.lower() == '.pdf':
            text = _extract_pdf_file(file_path)
        elif file_path.suffix.lower() == '.txt':
            text = _extract_text_file(file_path)
        else:
            continue
            
        # Split into chunks
        chunks = _split_text(text, chunk_size, chunk_overlap)
        
        for i, chunk in enumerate(chunks):
            documents.append({
                'id': f"{file_path.stem}_{i}",
                'text': chunk,
                'metadata': {
                    'filename': file_path.name,
                    'chunk_id': i,
                    'total_chunks': len(chunks)
                }
            })
    
    # Generate embeddings and upsert
    if documents:
        texts = [doc['text'] for doc in documents]
        embeddings = model.encode(texts)
        
        vectors = [
            {
                'id': doc['id'],
                'values': embedding.tolist(),
                'metadata': {**doc['metadata'], 'text': doc['text']}
            }
            for doc, embedding in zip(documents, embeddings)
        ]
        
        index.upsert(vectors=vectors, namespace=namespace)
        logger.info(f"Ingested {len(vectors)} document chunks")


def query_context(
    query_text: str,
    embeddings_model: str,
    index: "Index",
    namespace: str = "default",
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Query Pinecone index for relevant context.
    
    Args:
        query_text: Query text to search for
        embeddings_model: Name of embeddings model to use
        index: Pinecone index object to query
        namespace: Namespace to query within the index
        top_k: Number of most similar results to return
        
    Returns:
        List of relevant context dictionaries containing:
        - text: The text content
        - score: Similarity score
        - filename: Source filename
        - chunk_id: Chunk identifier
    """
    model = SentenceTransformer(embeddings_model)
    query_embedding = model.encode([query_text])
    
    results = index.query(
        vector=query_embedding[0].tolist(),
        top_k=top_k,
        namespace=namespace,
        include_metadata=True
    )
    
    contexts = []
    for match in results['matches']:
        contexts.append({
            'text': match['metadata']['text'],
            'score': match['score'],
            'filename': match['metadata']['filename'],
            'chunk_id': match['metadata']['chunk_id']
        })
    
    return contexts


def _extract_pdf_file(file_path: Path) -> str:
    """
    Extract text from PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text content from all pages
    """
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def _extract_text_file(file_path: Path) -> str:
    """
    Extract text from text file.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        File content as string
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to split
        chunk_size: Maximum size of each chunk
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        if end >= len(text):
            break
            
        start = end - chunk_overlap
    
    return chunks

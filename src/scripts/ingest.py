#!/usr/bin/env python3
"""
Simple document ingestion script for Pinecone RAG system.
Ingests all .txt files from src/data/ into Pinecone index.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI


def load_environment():
    """Load environment variables from .env file."""
    load_dotenv()
    return {
        'pinecone_api_key': os.getenv('PINECONE_API_KEY'),
        'pinecone_index_name': os.getenv('PINECONE_INDEX_NAME'),
        'openai_api_key': os.getenv('OPENAI_API_KEY')
    }


def initialize_pinecone(api_key: str, index_name: str):
    """Initialize Pinecone client and get index."""
    pc = Pinecone(api_key=api_key)
    
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=1536,  # OpenAI embedding dimension
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Created new index: {index_name}")
    else:
        # Get existing index info to check dimension
        index_info = pc.describe_index(index_name)
        print(f"Using existing index: {index_name} (dimension: {index_info.dimension})")
        
        # If dimensions don't match, we need to delete and recreate
        if index_info.dimension != 1536:
            print(f"Index dimension mismatch. Deleting and recreating index...")
            pc.delete_index(index_name)
            # Wait a moment for deletion to complete
            import time
            time.sleep(10)
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"Recreated index: {index_name} with dimension 1536")
    
    return pc.Index(index_name)


def split_text(text: str, chunk_size: int = 500) -> List[str]:
    """Split text into chunks of specified size."""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end
    
    return [chunk for chunk in chunks if chunk]


def get_openai_embeddings(texts: List[str], api_key: str) -> List[List[float]]:
    """Generate embeddings using OpenAI API."""
    client = OpenAI(api_key=api_key)
    
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=texts
    )
    
    return [embedding.embedding for embedding in response.data]


def ingest_documents():
    """Main ingestion function."""
    # Load environment
    env = load_environment()
    
    # Initialize Pinecone
    index = initialize_pinecone(env['pinecone_api_key'], env['pinecone_index_name'])
    
    # Get data directory
    data_dir = Path(__file__).parent.parent / "data"
    
    # Process all .txt files
    all_vectors = []
    file_count = 0
    total_chunks = 0
    
    for file_path in data_dir.glob("*.txt"):
        if file_path.name == '.gitkeep':
            continue
            
        print(f"Processing: {file_path.name}")
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split into chunks
        chunks = split_text(text, chunk_size=500)
        
        if not chunks:
            continue
        
        # Generate embeddings
        embeddings = get_openai_embeddings(chunks, env['openai_api_key'])
        
        # Create vectors for Pinecone
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector = {
                'id': f"{file_path.stem}_{i}",
                'values': embedding,
                'metadata': {
                    'text': chunk,
                    'filename': file_path.name,
                    'chunk_id': i,
                    'total_chunks': len(chunks)
                }
            }
            all_vectors.append(vector)
        
        file_count += 1
        total_chunks += len(chunks)
        print(f"  - Created {len(chunks)} chunks")
    
    # Upsert all vectors to Pinecone
    if all_vectors:
        # Upsert in batches (Pinecone has limits)
        batch_size = 100
        for i in range(0, len(all_vectors), batch_size):
            batch = all_vectors[i:i + batch_size]
            index.upsert(vectors=batch, namespace="default")
    
    # Print summary
    print(f"\n✓ Ingestion complete!")
    print(f"  Files processed: {file_count}")
    print(f"  Total chunks ingested: {total_chunks}")
    print(f"  Vectors stored in index: {env['pinecone_index_name']}")


if __name__ == "__main__":
    ingest_documents()

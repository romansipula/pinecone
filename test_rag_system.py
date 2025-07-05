#!/usr/bin/env python3
"""
Quick test script to verify RAG system is working.
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, 'src')

from pinecone import Pinecone
from openai import OpenAI

def test_rag_system():
    """Test the RAG system end-to-end."""
    print("🔍 RAG SYSTEM TEST")
    print("=" * 50)
    
    # Load environment
    load_dotenv()
    
    # Initialize clients
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pc.Index(os.getenv('PINECONE_INDEX_NAME'))
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Test query
    query = "bicycle benefits for senior employees"
    print(f"Query: {query}")
    
    # Generate query embedding
    response = client.embeddings.create(
        model='text-embedding-ada-002',
        input=[query]
    )
    query_vector = response.data[0].embedding
    
    # Search Pinecone
    results = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True,
        namespace='default'
    )
    
    print(f"Results found: {len(results.matches)}")
    print("-" * 50)
    
    for i, match in enumerate(results.matches, 1):
        score = match.score
        filename = match.metadata.get('filename', 'Unknown')
        text = match.metadata.get('text', 'No text')[:100]
        
        print(f"{i}. Score: {score:.3f}")
        print(f"   Source: {filename}")
        print(f"   Text: {text}...")
        print()
    
    print("✅ RAG system test completed successfully!")
    return len(results.matches) > 0

if __name__ == "__main__":
    success = test_rag_system()
    if success:
        print("🎉 RAG system is working properly!")
    else:
        print("❌ No results found - check ingestion")

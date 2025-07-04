#!/usr/bin/env python3
"""Simple debug test."""

import numpy as np
from unittest.mock import Mock

print("Testing numpy array .tolist() method...")
mock_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
print("Mock embeddings:", mock_embeddings)
print("Type:", type(mock_embeddings))
print("First element:", mock_embeddings[0])
print("First element tolist():", mock_embeddings[0].tolist())

print("\nTesting generation agent...")
try:
    from src.agents.generation_agent import GenerationAgent
    
    generation_agent = GenerationAgent("test-key", "gpt-3.5-turbo")
    contexts = [
        {'text': 'Test context', 'filename': 'test.txt', 'score': 0.9}
    ]
    formatted_contexts = generation_agent._format_contexts(contexts)
    print("Formatted contexts:", formatted_contexts)
    print("✓ GenerationAgent works")
    
except Exception as e:
    print(f"❌ GenerationAgent error: {e}")
    import traceback
    traceback.print_exc()

print("\nDone!")

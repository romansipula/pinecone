#!/usr/bin/env python3
"""Test just the failing query context test."""

import sys
import numpy as np
from unittest.mock import Mock, patch

# Add project root to Python path
sys.path.insert(0, '.')

def test_query_context_fix():
    """Test that the query context fix works."""
    print("Testing query context fix...")
    
    try:
        from src.rag_utils import query_context
        
        with patch('src.rag_utils.SentenceTransformer') as mock_transformer:
            # Setup mocks
            mock_model = Mock()
            mock_transformer.return_value = mock_model
            
            # Create mock embedding that has .tolist() method
            mock_embedding = np.array([[0.1, 0.2, 0.3]])
            mock_model.encode.return_value = mock_embedding
            
            mock_index = Mock()
            mock_index.query.return_value = {
                'matches': [
                    {
                        'score': 0.9,
                        'metadata': {
                            'text': 'Test context 1',
                            'filename': 'test1.txt',
                            'chunk_id': 0
                        }
                    },
                    {
                        'score': 0.8,
                        'metadata': {
                            'text': 'Test context 2',
                            'filename': 'test2.txt',
                            'chunk_id': 1
                        }
                    }
                ]
            }
            
            # This should not raise an AttributeError anymore
            result = query_context(
                query_text="test query",
                embeddings_model="test-model",
                index=mock_index,
                namespace="test",
                top_k=5
            )
            
            # Verify results
            assert len(result) == 2
            assert result[0]['text'] == 'Test context 1'
            assert result[0]['score'] == 0.9
            assert result[1]['text'] == 'Test context 2'
            assert result[1]['score'] == 0.8
            
            print("✓ Query context fix successful")
            return True
            
    except Exception as e:
        print(f"❌ Query context fix failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_query_context_fix()
    if success:
        print("\n🎉 Query context fix validated!")
    else:
        print("\n❌ Query context fix failed!")
    exit(0 if success else 1)

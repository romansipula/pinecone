#!/usr/bin/env python3
"""Debug test to check the specific fixes."""

def test_numpy_mock():
    """Test if numpy array mock works."""
    try:
        import numpy as np
        from unittest.mock import Mock, patch
        from src.rag_utils import ingest_documents
        import tempfile
        from pathlib import Path
        
        # Setup mocks
        with patch('src.rag_utils.SentenceTransformer') as mock_transformer:
            with patch('src.rag_utils._extract_text_file') as mock_extract_text:
                with patch('src.rag_utils._split_text') as mock_split_text:
                    
                    mock_model = Mock()
                    mock_transformer.return_value = mock_model
                    mock_extract_text.return_value = "Test document content"
                    mock_split_text.return_value = ["chunk1", "chunk2"]
                    
                    # Create mock embeddings that have .tolist() method
                    mock_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
                    mock_model.encode.return_value = mock_embeddings
                    
                    mock_index = Mock()
                    
                    # Create temporary directory with test file
                    with tempfile.TemporaryDirectory() as temp_dir:
                        test_file = Path(temp_dir) / "test.txt"
                        test_file.write_text("test content")
                        
                        ingest_documents(
                            directory=temp_dir,
                            embeddings_model="test-model",
                            index=mock_index,
                            namespace="test"
                        )
                    
                    print("✓ Numpy array mock test passed")
                    return True
        
    except Exception as e:
        print(f"❌ Numpy array mock test failed: {e}")
        return False

def test_generation_agent():
    """Test generation agent without context manager."""
    try:
        from src.agents.generation_agent import GenerationAgent
        
        generation_agent = GenerationAgent("test-key", "gpt-3.5-turbo")
        contexts = [
            {'text': 'Test context', 'filename': 'test.txt', 'score': 0.9}
        ]
        formatted_contexts = generation_agent._format_contexts(contexts)
        assert "[Context 1]" in formatted_contexts
        print("✓ GenerationAgent context formatting works")
        return True
        
    except Exception as e:
        print(f"❌ GenerationAgent test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Debug Test Suite")
    print("==================")
    
    success1 = test_numpy_mock()
    success2 = test_generation_agent()
    
    if success1 and success2:
        print("\n🎉 All debug tests passed!")
    else:
        print("\n❌ Some debug tests failed")

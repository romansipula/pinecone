#!/usr/bin/env python3
"""
Minimal test to validate the fixes.
"""
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

# Add project root to Python path
sys.path.insert(0, '.')

def test_numpy_fix():
    """Test the numpy array fix for the tolist() method."""
    try:
        from src.rag_utils import ingest_documents
        
        with patch('src.rag_utils.SentenceTransformer') as mock_transformer:
            with patch('src.rag_utils._extract_text_file') as mock_extract_text:
                with patch('src.rag_utils._split_text') as mock_split_text:
                    
                    # Setup mocks
                    mock_model = Mock()
                    mock_transformer.return_value = mock_model
                    mock_extract_text.return_value = "Test document content"
                    mock_split_text.return_value = ["chunk1", "chunk2"]
                    
                    # Create mock embeddings with numpy array (has .tolist() method)
                    mock_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
                    mock_model.encode.return_value = mock_embeddings
                    
                    mock_index = Mock()
                    
                    # Create temporary directory with test file
                    with tempfile.TemporaryDirectory() as temp_dir:
                        test_file = Path(temp_dir) / "test.txt"
                        test_file.write_text("test content")
                        
                        # This should not raise an AttributeError anymore
                        ingest_documents(
                            directory=temp_dir,
                            embeddings_model="test-model",
                            index=mock_index,
                            namespace="test"
                        )
                        
                        # Check that upsert was called
                        mock_index.upsert.assert_called_once()
                        
                        print("✓ NumPy array fix successful")
                        return True
                        
    except Exception as e:
        print(f"❌ NumPy array fix failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_generation_agent_fix():
    """Test the generation agent context manager fix."""
    try:
        from src.agents.generation_agent import GenerationAgent
        
        # This should not raise a context manager error anymore
        generation_agent = GenerationAgent("test-key", "gpt-3.5-turbo")
        contexts = [
            {'text': 'Test context', 'filename': 'test.txt', 'score': 0.9}
        ]
        formatted_contexts = generation_agent._format_contexts(contexts)
        
        # Check that formatting works
        assert "[Context 1]" in formatted_contexts
        assert "test.txt" in formatted_contexts
        
        print("✓ GenerationAgent context manager fix successful")
        return True
        
    except Exception as e:
        print(f"❌ GenerationAgent fix failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the validation tests."""
    print("🔧 Validating Fixes")
    print("=" * 20)
    
    test1 = test_numpy_fix()
    test2 = test_generation_agent_fix()
    
    print("\n" + "=" * 40)
    print("VALIDATION SUMMARY")
    print("=" * 40)
    
    if test1 and test2:
        print("🎉 All fixes validated successfully!")
        return True
    else:
        print("❌ Some fixes failed validation.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

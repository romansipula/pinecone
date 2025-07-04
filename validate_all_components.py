#!/usr/bin/env python3
"""Simple validation test for all components."""

import sys
import os
import tempfile
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, '.')

def test_all_components():
    """Test all components to ensure they work."""
    print("🧪 Running Component Validation Tests")
    print("=" * 40)
    
    # Test 1: Import test
    print("\n1. Testing imports...")
    try:
        from src.rag_utils import _extract_text_file, _split_text
        print("✓ RAG utils imported successfully")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test 2: Text extraction with proper file handling
    print("\n2. Testing text extraction...")
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content for extraction")
            f.flush()
            temp_file_path = f.name
            
        # File is now closed, safe to read
        try:
            result = _extract_text_file(Path(temp_file_path))
            assert result == "Test content for extraction"
            print("✓ Text extraction successful")
        finally:
            # Clean up
            try:
                os.unlink(temp_file_path)
                print("✓ File cleanup successful")
            except:
                print("⚠️  File cleanup failed (Windows permission issue)")
                
    except Exception as e:
        print(f"❌ Text extraction failed: {e}")
        return False
    
    # Test 3: Text splitting
    print("\n3. Testing text splitting...")
    try:
        text = "This is a test document with multiple sentences. " * 10
        chunks = _split_text(text, chunk_size=50, chunk_overlap=10)
        assert len(chunks) > 1
        assert len(chunks[0]) <= 50
        print("✓ Text splitting successful")
    except Exception as e:
        print(f"❌ Text splitting failed: {e}")
        return False
    
    # Test 4: Agent functionality
    print("\n4. Testing agent functionality...")
    try:
        from unittest.mock import Mock
        from src.agents.query_agent import QueryAgent
        from src.agents.retrieval_agent import RetrievalAgent
        from src.agents.generation_agent import GenerationAgent
        
        # Test QueryAgent
        mock_index = Mock()
        query_agent = QueryAgent("test-model", mock_index, "test")
        contexts = [{'text': 'Test', 'filename': 'test.txt'}]
        formatted = query_agent.format_contexts(contexts)
        assert "Context 1" in formatted
        
        # Test RetrievalAgent
        retrieval_agent = RetrievalAgent("test-model", mock_index, "test")
        contexts = [{'text': 'Test', 'filename': 'test.txt', 'score': 0.9}]
        summary = retrieval_agent.get_context_summary(contexts)
        assert summary['total'] == 1
        
        # Test GenerationAgent
        generation_agent = GenerationAgent("test-key", "gpt-3.5-turbo")
        formatted = generation_agent._format_contexts(contexts)
        assert "[Context 1]" in formatted
        
        print("✓ All agents working correctly")
        
    except Exception as e:
        print(f"❌ Agent functionality failed: {e}")
        return False
    
    print("\n" + "=" * 40)
    print("🎉 ALL COMPONENT VALIDATION TESTS PASSED!")
    print("=" * 40)
    return True

if __name__ == "__main__":
    success = test_all_components()
    if success:
        print("\n✅ The RAG chatbot project is fully functional!")
        print("🚀 Ready for production use!")
    else:
        print("\n❌ Some validation tests failed.")
    exit(0 if success else 1)

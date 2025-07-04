"""
Comprehensive test runner for the RAG chatbot project.
"""
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test all critical imports."""
    print("Testing imports...")
    
    try:
        # Test core dependencies
        import pinecone
        print("OK Pinecone imported")
        
        import openai
        print("OK OpenAI imported")
        
        from sentence_transformers import SentenceTransformer
        print("OK Sentence Transformers imported")
        
        from pypdf import PdfReader
        print("OK PyPDF imported")
        
        import jinja2
        print("OK Jinja2 imported")
        
        from dotenv import load_dotenv
        print("OK Python-dotenv imported")
        
        # Test our modules
        from src.rag_utils import init_pinecone, ingest_documents, query_context
        print("OK RAG utils imported")
        
        from src.agents.query_agent import QueryAgent
        print("OK QueryAgent imported")
        
        from src.agents.retrieval_agent import RetrievalAgent
        print("OK RetrievalAgent imported")
        
        from src.agents.generation_agent import GenerationAgent
        print("OK GenerationAgent imported")
        
        return True
        
    except Exception as e:
        print(f"ERROR Import failed: {e}")
        return False

def test_agent_functionality():
    """Test agent functionality without external dependencies."""
    print("\nTesting agent functionality...")
    
    try:
        from unittest.mock import Mock
        from src.agents.query_agent import QueryAgent
        from src.agents.retrieval_agent import RetrievalAgent
        from src.agents.generation_agent import GenerationAgent
        
        # Test QueryAgent
        mock_index = Mock()
        query_agent = QueryAgent("test-model", mock_index, "test", 5)
        
        # Test format_contexts
        contexts = [
            {'text': 'Test context', 'filename': 'test.txt'}
        ]
        formatted = query_agent.format_contexts(contexts)
        assert "Context 1 (from test.txt)" in formatted
        print("OK QueryAgent format_contexts works")
        
        # Test RetrievalAgent
        retrieval_agent = RetrievalAgent("test-model", mock_index, "test")
        
        # Test context summary with proper context structure
        contexts = [
            {'text': 'Test context', 'filename': 'test.txt', 'score': 0.9}
        ]
        summary = retrieval_agent.get_context_summary(contexts)
        assert summary['total'] == 1
        print("OK RetrievalAgent context summary works")
        
        # Test GenerationAgent (without OpenAI calls)
        generation_agent = GenerationAgent("test-key", "gpt-3.5-turbo")
        formatted_contexts = generation_agent._format_contexts(contexts)
        assert "[Context 1]" in formatted_contexts
        print("OK GenerationAgent context formatting works")
        
        return True
        
    except Exception as e:
        print(f"ERROR Agent functionality error: {e}")
        return False

def test_pytest_run():
    """Run pytest and capture results."""
    print("\nRunning unit tests...")
    
    try:
        import subprocess
        result = subprocess.run(
            ['python', '-m', 'pytest', 'src/tests/', '-v', '--tb=short'],
            capture_output=True,
            text=True,
            cwd=str(project_root)
        )
        
        if result.returncode == 0:
            print("OK All unit tests passed")
            return True
        else:
            print("ERROR Some unit tests failed")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"ERROR Failed to run pytest: {e}")
        return False

def main():
    """Run all tests."""
    print("RAG Chatbot Test Suite")
    print("=" * 30)
    
    results = []
    
    # Test imports
    print("\n1. Testing Imports")
    print("-" * 20)
    results.append(("Imports", test_imports()))
    
    # Test agent functionality
    print("\n2. Testing Agent Functionality")
    print("-" * 30)
    results.append(("Agent Functionality", test_agent_functionality()))
    
    # Test pytest
    print("\n3. Running Unit Tests")
    print("-" * 20)
    results.append(("Unit Tests", test_pytest_run()))
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{status:4} {test_name}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed. Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

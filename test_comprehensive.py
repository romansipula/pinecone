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
        print("✓ Pinecone imported")
        
        import openai
        print("✓ OpenAI imported")
        
        from sentence_transformers import SentenceTransformer
        print("✓ Sentence Transformers imported")
        
        from pypdf import PdfReader
        print("✓ PyPDF imported")
        
        import jinja2
        print("✓ Jinja2 imported")
        
        from dotenv import load_dotenv
        print("✓ Python-dotenv imported")
        
        # Test our modules
        from src.rag_utils import init_pinecone, ingest_documents, query_context
        print("✓ RAG utils imported")
        
        from src.agents.query_agent import QueryAgent
        print("✓ QueryAgent imported")
        
        from src.agents.retrieval_agent import RetrievalAgent
        print("✓ RetrievalAgent imported")
        
        from src.agents.generation_agent import GenerationAgent
        print("✓ GenerationAgent imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_agent_functionality():
    """Test basic agent functionality without external dependencies."""
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
        print("✓ QueryAgent format_contexts works")
        
        # Test RetrievalAgent
        retrieval_agent = RetrievalAgent("test-model", mock_index, "test")
        
        # Test context summary with proper context structure
        contexts = [
            {'text': 'Test context', 'filename': 'test.txt', 'score': 0.9}
        ]
        summary = retrieval_agent.get_context_summary(contexts)
        assert summary['total'] == 1
        print("✓ RetrievalAgent context summary works")
        
        # Test GenerationAgent (without OpenAI calls)
        generation_agent = GenerationAgent("test-key", "gpt-3.5-turbo")
        formatted_contexts = generation_agent._format_contexts(contexts)
        assert "[Context 1]" in formatted_contexts
        print("✓ GenerationAgent context formatting works")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent functionality error: {e}")
        return False

def test_rag_utils():
    """Test RAG utility functions."""
    print("\nTesting RAG utils...")
    
    try:
        from src.rag_utils import _split_text, _extract_text_file
        import tempfile
        
        # Test text splitting
        text = "This is a test document. " * 20
        chunks = _split_text(text, chunk_size=50, chunk_overlap=10)
        assert len(chunks) > 1
        print("✓ Text splitting works")
        
        # Test text file extraction
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content")
            f.flush()
            temp_file_path = f.name
            
        try:
            content = _extract_text_file(Path(temp_file_path))
            assert content == "Test content"
            print("✓ Text file extraction works")
        finally:
            # Ensure file is closed before deletion
            try:
                os.unlink(temp_file_path)
            except (PermissionError, FileNotFoundError):
                pass  # File might already be deleted or still in use
        
        return True
        
    except Exception as e:
        print(f"❌ RAG utils error: {e}")
        return False

def test_project_structure():
    """Test that all required files exist."""
    print("\nTesting project structure...")
    
    required_files = [
        "src/agents/query_agent.py",
        "src/agents/retrieval_agent.py", 
        "src/agents/generation_agent.py",
        "src/rag_utils.py",
        "src/prompts/system_prompt.jinja2",
        "src/prompts/user_prompt.jinja2",
        "src/scripts/ingest.py",
        "src/tests/test_agents.py",
        "src/tests/test_rag_utils.py",
        "main.py",
        "requirements.txt",
        "README.md",
        ".env.example",
        "pyproject.toml"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print(f"✓ All {len(required_files)} required files exist")
        return True

def run_unit_tests():
    """Run the actual pytest suite."""
    print("\nRunning unit tests...")
    
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            "src/tests/", "-v", "--tb=short", "-x"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✓ All unit tests passed")
            print(result.stdout)
            return True
        else:
            print("❌ Some unit tests failed")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Unit tests timed out")
        return False
    except Exception as e:
        print(f"❌ Error running unit tests: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 RAG Chatbot Comprehensive Test Suite")
    print("=" * 50)
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Imports", test_imports),
        ("Agent Functionality", test_agent_functionality),
        ("RAG Utils", test_rag_utils),
        ("Unit Tests", run_unit_tests),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 30)
        success = test_func()
        results.append((test_name, success))
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if not success:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! RAG Chatbot is ready for production!")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

def test_rag_utils():
    """Test RAG utility functions."""
    print("\nTesting RAG utils...")
    
    try:
        from src.rag_utils import _split_text, _extract_text_file
        import tempfile
        
        # Test text splitting
        text = "This is a test document. " * 20
        chunks = _split_text(text, chunk_size=50, chunk_overlap=10)
        assert len(chunks) > 1
        print("✓ Text splitting works")
        
        # Test text file extraction
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content")
            f.flush()
            temp_file_path = f.name
            
        try:
            content = _extract_text_file(Path(temp_file_path))
            assert content == "Test content"
            print("✓ Text file extraction works")
        finally:
            # Ensure file is closed before deletion
            try:
                os.unlink(temp_file_path)
            except (PermissionError, FileNotFoundError):
                pass  # File might already be deleted or still in use
        
        return True
        
    except Exception as e:
        print(f"❌ RAG utils error: {e}")
        return False

def test_project_structure():
    """Test that all required files exist."""
    print("\nTesting project structure...")
    
    required_files = [
        "src/agents/query_agent.py",
        "src/agents/retrieval_agent.py", 
        "src/agents/generation_agent.py",
        "src/rag_utils.py",
        "src/prompts/system_prompt.jinja2",
        "src/prompts/user_prompt.jinja2",
        "src/scripts/ingest.py",
        "src/tests/test_agents.py",
        "src/tests/test_rag_utils.py",
        "main.py",
        "requirements.txt",
        "README.md",
        ".env.example",
        "pyproject.toml"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print(f"✓ All {len(required_files)} required files exist")
        return True

def run_unit_tests():
    """Run the actual pytest suite."""
    print("\nRunning unit tests...")
    
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "src/tests/", "-v", "--tb=short", "-x"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✓ All unit tests passed")
            print(result.stdout)
            return True
        else:
            print("❌ Some unit tests failed")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Unit tests timed out")
        return False
    except Exception as e:
        print(f"❌ Error running unit tests: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 RAG Chatbot Comprehensive Test Suite")
    print("=" * 50)
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Imports", test_imports),
        ("Agent Functionality", test_agent_functionality),
        ("RAG Utils", test_rag_utils),
        ("Unit Tests", run_unit_tests),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 30)
        success = test_func()
        results.append((test_name, success))
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if not success:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! RAG Chatbot is ready for production!")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

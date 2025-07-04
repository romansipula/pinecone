"""
Simple test to verify the installation works.
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all required modules can be imported."""
    try:
        # Test core dependencies
        import pinecone
        print("✓ Pinecone imported successfully")
        
        import openai
        print("✓ OpenAI imported successfully")
        
        import sentence_transformers
        print("✓ Sentence Transformers imported successfully")
        
        from pypdf import PdfReader
        print("✓ PyPDF imported successfully")
        
        import jinja2
        print("✓ Jinja2 imported successfully")
        
        from dotenv import load_dotenv
        print("✓ Python-dotenv imported successfully")
        
        # Test our modules
        from src.rag_utils import init_pinecone, ingest_documents, query_context
        print("✓ RAG utils imported successfully")
        
        from src.agents.query_agent import QueryAgent
        print("✓ Query Agent imported successfully")
        
        from src.agents.retrieval_agent import RetrievalAgent
        print("✓ Retrieval Agent imported successfully")
        
        from src.agents.generation_agent import GenerationAgent
        print("✓ Generation Agent imported successfully")
        
        print("\n🎉 All imports successful! The RAG chatbot is ready to use.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("Testing RAG Chatbot Installation...")
    print("=" * 50)
    
    success = test_imports()
    
    if success:
        print("\n📋 Next steps:")
        print("1. Create a .env file with your API keys (see .env.example)")
        print("2. Place documents in src/data/ directory")
        print("3. Run: python -m src.scripts.ingest")
        print("4. Run: python main.py")
    else:
        print("\n❌ Installation verification failed.")
        print("Please check the error messages above and install missing dependencies.")

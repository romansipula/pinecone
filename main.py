"""
Interactive RAG chatbot demo.
"""
import os
import sys
import logging
from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.rag_utils import init_pinecone
from src.agents.query_agent import QueryAgent
from src.agents.generation_agent import GenerationAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main chatbot function."""
    # Load environment variables
    load_dotenv()
    
    # Get configuration
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    index_name = os.getenv('PINECONE_INDEX_NAME', 'rag-chatbot')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    if not all([pinecone_api_key, openai_api_key]):
        logger.error("Missing required environment variables. Please check your .env file.")
        logger.error("Required: PINECONE_API_KEY, OPENAI_API_KEY")
        return
    
    try:
        # Initialize Pinecone
        logger.info("Initializing Pinecone...")
        index = init_pinecone(
            api_key=pinecone_api_key,
            environment="",  # Not needed for new Pinecone SDK
            index_name=index_name,
            dimension=384,  # For all-MiniLM-L6-v2
            metric="cosine"
        )
        
        # Initialize agents
        logger.info("Initializing agents...")
        query_agent = QueryAgent(
            embeddings_model="all-MiniLM-L6-v2",
            index=index,
            namespace="default",
            top_k=5
        )
        
        generation_agent = GenerationAgent(
            openai_api_key=openai_api_key,
            model="gpt-3.5-turbo",
            temperature=0.7
        )
        
        # Start chat loop
        print("🤖 RAG Chatbot initialized successfully!")
        print("Type 'quit' to exit, 'help' for commands")
        print("-" * 50)
        
        conversation_history = []
        
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if user_input.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'help':
                    print_help()
                    continue
                
                elif user_input.lower() == 'clear':
                    conversation_history = []
                    print("🗑️ Conversation history cleared.")
                    continue
                
                elif not user_input:
                    continue
                
                # Retrieve relevant context
                logger.info(f"Processing query: {user_input}")
                contexts = query_agent.query(user_input)
                
                if not contexts:
                    print("🤖 Bot: I don't have any relevant information to answer your question.")
                    continue
                
                # Generate response
                response = generation_agent.generate_response(
                    query=user_input,
                    contexts=contexts,
                    conversation_history=conversation_history[-6:]  # Keep last 3 exchanges
                )
                
                print(f"🤖 Bot: {response}")
                
                # Update conversation history
                conversation_history.extend([
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": response}
                ])
                
                # Show context info
                print(f"\n📊 Retrieved {len(contexts)} relevant context(s)")
                for i, ctx in enumerate(contexts[:3], 1):
                    print(f"   {i}. {ctx['filename']} (score: {ctx['score']:.3f})")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print("❌ Sorry, I encountered an error. Please try again.")
    
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        print(f"❌ Failed to initialize chatbot: {e}")


def print_help():
    """Print help message."""
    print("\n🆘 Available commands:")
    print("  help  - Show this help message")
    print("  clear - Clear conversation history")
    print("  quit  - Exit the chatbot")
    print("\nTip: Ask questions about your ingested documents!")


if __name__ == "__main__":
    main()

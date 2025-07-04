"""
Document ingestion script for RAG chatbot.
"""
import os
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

from ..rag_utils import init_pinecone, ingest_documents

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main ingestion function."""
    parser = argparse.ArgumentParser(description='Ingest documents into Pinecone')
    parser.add_argument(
        '--data-dir',
        type=str,
        default='src/data',
        help='Directory containing documents to ingest'
    )
    parser.add_argument(
        '--embeddings-model',
        type=str,
        default='all-MiniLM-L6-v2',
        help='Sentence transformer model for embeddings'
    )
    parser.add_argument(
        '--namespace',
        type=str,
        default='default',
        help='Pinecone namespace'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=1000,
        help='Text chunk size'
    )
    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=200,
        help='Text chunk overlap'
    )
    
    args = parser.parse_args()
    
    # Get environment variables
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    index_name = os.getenv('PINECONE_INDEX_NAME', 'rag-chatbot')
    
    if not pinecone_api_key:
        logger.error("PINECONE_API_KEY must be set")
        return
    
    # Check if data directory exists
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory {data_dir} does not exist")
        return
    
    # Check if there are any files to ingest
    file_count = len(list(data_dir.glob('**/*.pdf'))) + len(list(data_dir.glob('**/*.txt')))
    if file_count == 0:
        logger.warning(f"No PDF or TXT files found in {data_dir}")
        return
    
    logger.info(f"Found {file_count} files to ingest")
    
    try:
        # Initialize Pinecone
        logger.info("Initializing Pinecone...")
        index = init_pinecone(
            api_key=pinecone_api_key,
            environment="",  # Not needed for new Pinecone SDK
            index_name=index_name,
            dimension=384,  # Dimension for all-MiniLM-L6-v2
            metric="cosine"
        )
        
        # Ingest documents
        logger.info("Starting document ingestion...")
        ingest_documents(
            directory=str(data_dir),
            embeddings_model=args.embeddings_model,
            index=index,
            namespace=args.namespace,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        
        logger.info("Document ingestion completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        raise


if __name__ == "__main__":
    main()

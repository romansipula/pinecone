# RAG Chatbot with Pinecone

A production-ready Retrieval-Augmented Generation (RAG) chatbot built with Pinecone vector database, OpenAI, and LangChain.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Document      │    │   Query Agent   │    │  Generation     │
│   Ingestion     │    │                 │    │  Agent          │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │   PDF/TXT   │ │    │ │ User Query  │ │    │ │   OpenAI    │ │
│ │   Files     │ │    │ │ Processing  │ │    │ │   Chat      │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ │ Completion  │ │
│        │        │    │        │        │    │ └─────────────┘ │
│        ▼        │    │        ▼        │    │        ▲        │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │        │        │
│ │   Chunking  │ │    │ │ Embeddings  │ │    │        │        │
│ │  & Embedding│ │    │ │ Generation  │ │    │        │        │
│ └─────────────┘ │    │ └─────────────┘ │    │        │        │
│        │        │    │        │        │    │        │        │
│        ▼        │    │        ▼        │    │        │        │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │  Pinecone   │ │◄───┤ │ Retrieval   │ ├────┤ │  Retrieved  │ │
│ │   Vector    │ │    │ │   Agent     │ │    │ │   Context   │ │
│ │  Database   │ │    │ │             │ │    │ │             │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated agents
- **Document Ingestion**: Support for PDF and text files with intelligent chunking
- **Vector Search**: Powered by Pinecone for fast similarity search
- **Multiple Embedding Models**: Support for OpenAI and Sentence Transformers
- **Conversation Memory**: Maintain context across chat sessions
- **Production Ready**: Type hints, comprehensive tests, and CI/CD pipeline

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd pinecone

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the root directory:

```bash
# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=rag-chatbot

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
```

### 3. Document Ingestion

Place your PDF or text files in the `src/data/` directory, then run:

```bash
python -m src.scripts.ingest --data-dir src/data --embeddings-model all-MiniLM-L6-v2
```

### 4. Run the Chatbot

```python
import os
from dotenv import load_dotenv
from src.rag_utils import init_pinecone
from src.agents.query_agent import QueryAgent
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.generation_agent import GenerationAgent

# Load environment variables
load_dotenv()

# Initialize Pinecone
index = init_pinecone(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment=os.getenv('PINECONE_ENVIRONMENT'),
    index_name=os.getenv('PINECONE_INDEX_NAME')
)

# Initialize agents
query_agent = QueryAgent(
    embeddings_model="all-MiniLM-L6-v2",
    index=index
)

generation_agent = GenerationAgent(
    openai_api_key=os.getenv('OPENAI_API_KEY')
)

# Chat loop
while True:
    user_query = input("You: ")
    if user_query.lower() == 'quit':
        break
    
    # Retrieve relevant context
    contexts = query_agent.query(user_query)
    
    # Generate response
    response = generation_agent.generate_response(user_query, contexts)
    print(f"Bot: {response}")
```

## Project Structure

```
pinecone/
├── src/
│   ├── agents/
│   │   ├── query_agent.py          # Query processing and orchestration
│   │   ├── retrieval_agent.py      # Document retrieval from vector DB
│   │   └── generation_agent.py     # Response generation with OpenAI
│   ├── prompts/
│   │   ├── system_prompt.jinja2    # System role and instructions
│   │   └── user_prompt.jinja2      # User query template
│   ├── data/                       # Document storage (PDF, TXT files)
│   ├── scripts/
│   │   └── ingest.py               # Document ingestion script
│   ├── tests/
│   │   ├── test_agents.py          # Agent tests
│   │   └── test_rag_utils.py       # Utility function tests
│   └── rag_utils.py                # Core RAG utilities
├── .github/
│   └── workflows/
│       └── python-ci.yml           # CI/CD pipeline
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # Project documentation
```

## Configuration

### Embedding Models

The system supports multiple embedding models:

- **Sentence Transformers**: `all-MiniLM-L6-v2` (default, 384 dimensions)
- **OpenAI Embeddings**: `text-embedding-ada-002` (1536 dimensions)

### Ingestion Parameters

- `--chunk-size`: Size of text chunks (default: 1000)
- `--chunk-overlap`: Overlap between chunks (default: 200)
- `--namespace`: Pinecone namespace (default: "default")

### Generation Parameters

- `model`: OpenAI model (default: "gpt-3.5-turbo")
- `temperature`: Generation temperature (default: 0.7)
- `max_tokens`: Maximum response tokens (default: 1000)

## Development

### Running Tests

```bash
# Run all tests
pytest src/tests/ -v

# Run with coverage
pytest src/tests/ -v --cov=src --cov-report=html
```

### Code Formatting

```bash
# Format code
black .

# Check formatting
black --check .

# Lint code
flake8 .
```

### Adding New Document Types

To support additional document types, extend the `ingest_documents` function in `rag_utils.py`:

```python
def _extract_document_text(file_path: Path) -> str:
    """Extract text from various document types."""
    if file_path.suffix.lower() == '.pdf':
        return _extract_pdf_text(file_path)
    elif file_path.suffix.lower() == '.txt':
        return _extract_text_file(file_path)
    elif file_path.suffix.lower() == '.docx':
        return _extract_docx_text(file_path)  # Implement this
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")
```

## API Reference

### RAG Utils

#### `init_pinecone(api_key, environment, index_name, dimension=1536, metric="cosine")`

Initialize and connect to Pinecone vector database.

#### `ingest_documents(directory, embeddings_model, index, namespace="default")`

Process and ingest documents from a directory into the vector database.

#### `query_context(query_text, embeddings_model, index, namespace="default", top_k=5)`

Query the vector database for relevant context.

### Agents

#### `QueryAgent(embeddings_model, index, namespace="default", top_k=5)`

High-level interface for querying and formatting retrieved contexts.

#### `RetrievalAgent(embeddings_model, index, namespace="default")`

Specialized agent for document retrieval with filtering and summarization.

#### `GenerationAgent(openai_api_key, model="gpt-3.5-turbo")`

OpenAI-powered response generation with conversation history support.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support, please open an issue in the GitHub repository.

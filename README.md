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

For detailed setup instructions, see [SETUP.md](SETUP.md).

### 1. Clone and Install

```bash
git clone https://github.com/romansipula/pinecone.git
cd pinecone
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Ingest Sample Data

The repository includes sample documents in `src/data/`:
- `example.txt` - Comprehensive RAG system documentation
- `sample_document.txt` - Additional sample content

```bash
python src/scripts/ingest.py
```

### 4. Run the Chatbot

```bash
python main.py
```

## Sample Usage

Once the chatbot is running, you can ask questions about your ingested documents:

```
User: What is a RAG system?
Bot: A RAG (Retrieval-Augmented Generation) system combines information retrieval 
     with large language models to provide accurate, context-aware responses...

User: How does vector search work?
Bot: Vector search works by converting text into dense numerical representations 
     called embeddings that capture semantic meaning...

User: What are the key components mentioned?
Bot: Based on the documents, the key components include:
     1. Vector Database (Pinecone)
     2. Embeddings Model (Sentence Transformers)
     3. Language Model (OpenAI GPT)
     4. Document Processing Pipeline...
```

## Adding Your Own Data

1. Place your documents in `src/data/`:
   ```
   src/data/
   ├── .gitkeep
   ├── example.txt (included)
   ├── your_document.pdf
   ├── another_doc.txt
   └── subfolder/
       └── more_docs.pdf
   ```

2. Run ingestion:
   ```bash
   python src/scripts/ingest.py
   ```

3. Start chatting:
   ```bash
   python main.py
   ```
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

Initialize and connect to Pinecone vector database. Note: The environment parameter is no longer required for Pinecone SDK v3.0+.

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

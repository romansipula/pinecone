# Quick Setup Guide

## Prerequisites

- Python 3.8+
- Git
- Pinecone account with API key
- OpenAI account with API key

## 1. Clone and Setup

```bash
git clone https://github.com/romansipula/pinecone.git
cd pinecone
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

## 2. Environment Configuration

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` file with your API keys:
```bash
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment_here
PINECONE_INDEX_NAME=rag-chatbot
OPENAI_API_KEY=your_openai_api_key_here
```

## 3. Prepare Data

### Using Sample Data
The repository includes sample data in `src/data/`:
- `example.txt` - Comprehensive example document about RAG systems
- `sample_document.txt` - Additional sample content

### Adding Your Own Data
1. Place your documents in the `src/data/` directory
2. Supported formats: `.txt` and `.pdf`
3. The system will automatically process all files in subdirectories

Example data structure:
```
src/data/
├── .gitkeep
├── example.txt
├── sample_document.txt
├── your_documents/
│   ├── document1.txt
│   ├── document2.pdf
│   └── more_files.txt
└── other_files.txt
```

## 4. Test Installation

```bash
python test_comprehensive.py
```

This will verify:
- All dependencies are installed correctly
- Import statements work
- Basic functionality is operational

## 5. Ingest Documents

Run the ingestion script to process your documents:

```bash
python src/scripts/ingest.py
```

This will:
- Process all documents in `src/data/`
- Split them into chunks
- Generate embeddings
- Store vectors in Pinecone

## 6. Run the Chatbot

```bash
python main.py
```

The chatbot will start and you can begin asking questions about your documents.

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure virtual environment is activated and all dependencies are installed
2. **API Key Errors**: Verify your `.env` file contains valid API keys
3. **Pinecone Connection**: Check your Pinecone environment and index settings
4. **Empty Results**: Ensure documents were successfully ingested

### Verification Commands

```bash
# Run comprehensive tests
python test_comprehensive.py

# Check specific components
python -c "from src.rag_utils import init_pinecone; print('RAG utils working')"
python -c "from src.agents.query_agent import QueryAgent; print('Agents working')"

# Test with sample query
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
from src.rag_utils import init_pinecone, query_context

index = init_pinecone(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment=os.getenv('PINECONE_ENVIRONMENT'),
    index_name=os.getenv('PINECONE_INDEX_NAME')
)
results = query_context('What is RAG?', 'all-MiniLM-L6-v2', index)
print(f'Found {len(results)} results')
"
```

For more detailed information, see the main [README.md](README.md).
```

## 5. Start Chatbot

```bash
python main.py
```

## Quick Commands

- **Add documents**: Place PDF/TXT files in `src/data/` then run ingestion
- **Run tests**: `pytest src/tests/`
- **Format code**: `black .`
- **Lint code**: `flake8 .`

## Troubleshooting

1. **Import errors**: Make sure virtual environment is activated
2. **API errors**: Check your API keys in `.env` file
3. **No context found**: Ensure documents are ingested first
4. **Permission errors**: Check file permissions in data directory

## Getting Help

- Check the main [README.md](README.md) for detailed documentation
- See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines
- Open an issue on GitHub for bugs or questions

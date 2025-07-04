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

## 3. Test Installation

```bash
python test_installation.py
```

## 4. Ingest Sample Data

```bash
python -m src.scripts.ingest --data-dir src/data
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

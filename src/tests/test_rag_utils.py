"""
Tests for rag_utils module.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os

from src.rag_utils import (
    init_pinecone,
    ingest_documents,
    query_context,
    _extract_text_file,
    _split_text
)


class TestInitPinecone:
    """Test cases for init_pinecone function."""
    
    @patch('src.rag_utils.Pinecone')
    def test_init_existing_index(self, mock_pinecone_class):
        """Test initialization with existing index."""
        mock_pc = Mock()
        mock_pinecone_class.return_value = mock_pc
        
        # Mock existing index
        mock_index_info = Mock()
        mock_index_info.name = 'test-index'
        mock_pc.list_indexes.return_value = [mock_index_info]
        
        mock_index = Mock()
        mock_pc.Index.return_value = mock_index
        
        result = init_pinecone(
            api_key="test-key",
            environment="test-env",
            index_name="test-index"
        )
        
        mock_pinecone_class.assert_called_once_with(api_key="test-key")
        assert result == mock_index
        mock_pc.create_index.assert_not_called()
    
    @patch('src.rag_utils.Pinecone')
    @patch('src.rag_utils.ServerlessSpec')
    def test_init_new_index(self, mock_serverless_spec, mock_pinecone_class):
        """Test initialization with new index."""
        mock_pc = Mock()
        mock_pinecone_class.return_value = mock_pc
        mock_pc.list_indexes.return_value = []
        
        mock_index = Mock()
        mock_pc.Index.return_value = mock_index
        
        mock_spec = Mock()
        mock_serverless_spec.return_value = mock_spec
        
        result = init_pinecone(
            api_key="test-key",
            environment="test-env",
            index_name="new-index",
            dimension=512,
            metric="euclidean"
        )
        
        mock_pc.create_index.assert_called_once()
        assert result == mock_index


class TestIngestDocuments:
    """Test cases for ingest_documents function."""
    
    @patch('src.rag_utils.SentenceTransformer')
    @patch('src.rag_utils._extract_text_file')
    @patch('src.rag_utils._split_text')
    def test_ingest_text_files(self, mock_split_text, mock_extract_text, mock_transformer):
        """Test ingesting text files."""
        # Setup mocks
        mock_model = Mock()
        mock_transformer.return_value = mock_model
        mock_extract_text.return_value = "Test document content"
        mock_split_text.return_value = ["chunk1", "chunk2"]
        mock_model.encode.return_value = [[0.1, 0.2], [0.3, 0.4]]
        
        mock_index = Mock()
        
        # Create temporary directory with test file
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.txt"
            test_file.write_text("test content")
            
            ingest_documents(
                directory=temp_dir,
                embeddings_model="test-model",
                index=mock_index,
                namespace="test"
            )
        
        # Verify calls
        mock_transformer.assert_called_once_with("test-model")
        mock_index.upsert.assert_called_once()
        
        # Check upsert call
        upsert_call = mock_index.upsert.call_args
        vectors = upsert_call[1]['vectors']
        assert len(vectors) == 2
        assert vectors[0]['id'] == 'test_0'
        assert vectors[1]['id'] == 'test_1'
    
    @patch('src.rag_utils.SentenceTransformer')
    def test_ingest_empty_directory(self, mock_transformer):
        """Test ingesting from empty directory."""
        mock_index = Mock()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            ingest_documents(
                directory=temp_dir,
                embeddings_model="test-model",
                index=mock_index
            )
        
        # Should not call upsert for empty directory
        mock_index.upsert.assert_not_called()


class TestQueryContext:
    """Test cases for query_context function."""
    
    @patch('src.rag_utils.SentenceTransformer')
    def test_query_context_success(self, mock_transformer):
        """Test successful context query."""
        # Setup mocks
        mock_model = Mock()
        mock_transformer.return_value = mock_model
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        
        mock_index = Mock()
        mock_index.query.return_value = {
            'matches': [
                {
                    'score': 0.9,
                    'metadata': {
                        'text': 'Test context 1',
                        'filename': 'test1.txt',
                        'chunk_id': 0
                    }
                },
                {
                    'score': 0.8,
                    'metadata': {
                        'text': 'Test context 2',
                        'filename': 'test2.txt',
                        'chunk_id': 1
                    }
                }
            ]
        }
        
        result = query_context(
            query_text="test query",
            embeddings_model="test-model",
            index=mock_index,
            namespace="test",
            top_k=5
        )
        
        # Verify results
        assert len(result) == 2
        assert result[0]['text'] == 'Test context 1'
        assert result[0]['score'] == 0.9
        assert result[1]['text'] == 'Test context 2'
        assert result[1]['score'] == 0.8
        
        # Verify index query call
        mock_index.query.assert_called_once_with(
            vector=[0.1, 0.2, 0.3],
            top_k=5,
            namespace="test",
            include_metadata=True
        )


class TestTextProcessing:
    """Test cases for text processing functions."""
    
    def test_extract_text_file(self):
        """Test text file extraction."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test file content")
            f.flush()
            
            result = _extract_text_file(Path(f.name))
            assert result == "Test file content"
            
            os.unlink(f.name)
    
    def test_split_text(self):
        """Test text splitting."""
        text = "This is a test document with multiple sentences. " * 10
        chunks = _split_text(text, chunk_size=50, chunk_overlap=10)
        
        assert len(chunks) > 1
        assert len(chunks[0]) <= 50
        
        # Check overlap
        if len(chunks) > 1:
            # Should have some overlap between chunks
            assert chunks[0][-10:] == chunks[1][:10]
    
    def test_split_text_small(self):
        """Test splitting small text."""
        text = "Short text"
        chunks = _split_text(text, chunk_size=100, chunk_overlap=20)
        
        assert len(chunks) == 1
        assert chunks[0] == text

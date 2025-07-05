"""
Unit tests for RAG utilities module.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
import tempfile
import os
import numpy as np

from src.rag_utils import (
    init_pinecone,
    ingest_documents,
    query_context,
    _extract_pdf_file,
    _extract_text_file,
    _split_text
)


class TestInitPinecone:
    """Test cases for init_pinecone function."""
    
    @patch('src.rag_utils.pinecone')
    def test_init_existing_index(self, mock_pinecone):
        """Test initialization with existing index."""
        mock_pinecone.list_indexes.return_value = ['test-index']
        mock_index = Mock()
        mock_pinecone.Index.return_value = mock_index
        
        result = init_pinecone(
            api_key="test-key",
            environment="test-env",
            index_name="test-index"
        )
        
        mock_pinecone.init.assert_called_once_with(
            api_key="test-key",
            environment="test-env"
        )
        assert result == mock_index
        mock_pinecone.create_index.assert_not_called()
    
    @patch('src.rag_utils.pinecone')
    def test_init_new_index(self, mock_pinecone):
        """Test initialization with new index."""
        mock_pinecone.list_indexes.return_value = []
        mock_index = Mock()
        mock_pinecone.Index.return_value = mock_index
        
        result = init_pinecone(
            api_key="test-key",
            environment="test-env",
            index_name="test-index",
            dimension=768,
            metric="euclidean"
        )
        
        mock_pinecone.init.assert_called_once_with(
            api_key="test-key",
            environment="test-env"
        )
        mock_pinecone.create_index.assert_called_once_with(
            name="test-index",
            dimension=768,
            metric="euclidean"
        )
        assert result == mock_index
    
    @patch('src.rag_utils.pinecone')
    def test_init_with_defaults(self, mock_pinecone):
        """Test initialization with default parameters."""
        mock_pinecone.list_indexes.return_value = []
        mock_index = Mock()
        mock_pinecone.Index.return_value = mock_index
        
        result = init_pinecone(
            api_key="test-key",
            environment="test-env",
            index_name="test-index"
        )
        
        mock_pinecone.create_index.assert_called_once_with(
            name="test-index",
            dimension=1536,
            metric="cosine"
        )
        assert result == mock_index
        mock_pinecone.Index.return_value = mock_index
        
        result = init_pinecone(
            api_key="test-key",
            environment="test-env",
            index_name="new-index",
            dimension=512,
            metric="euclidean"
        )
        
        mock_pinecone.create_index.assert_called_once_with(
            name="new-index",
            dimension=512,
            metric="euclidean"
        )
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
        
        # Create mock embeddings that have .tolist() method
        import numpy as np
        mock_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_model.encode.return_value = mock_embeddings
        
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
    
    @patch('src.rag_utils.SentenceTransformer')
    @patch('src.rag_utils._extract_pdf_file')
    @patch('src.rag_utils._split_text')
    def test_ingest_pdf_files(self, mock_split_text, mock_extract_pdf, mock_transformer):
        """Test ingesting PDF files."""
        # Setup mocks
        mock_model = Mock()
        mock_transformer.return_value = mock_model
        mock_extract_pdf.return_value = "PDF document content"
        mock_split_text.return_value = ["pdf_chunk1", "pdf_chunk2"]
        
        # Create mock embeddings with numpy array (has .tolist() method)
        mock_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_model.encode.return_value = mock_embeddings
        
        mock_index = Mock()
        
        # Create temporary directory with test PDF file
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.pdf"
            test_file.write_text("dummy pdf content")  # Mock PDF file
            
            ingest_documents(
                directory=temp_dir,
                embeddings_model="test-model",
                index=mock_index,
                namespace="test"
            )
        
        # Verify calls
        mock_transformer.assert_called_once_with("test-model")
        mock_extract_pdf.assert_called_once()
        mock_split_text.assert_called_once()
        mock_index.upsert.assert_called_once()
        
        # Check upsert call
        upsert_call = mock_index.upsert.call_args
        vectors = upsert_call[1]['vectors']
        assert len(vectors) == 2
        assert vectors[0]['id'] == 'test_0'
        assert vectors[1]['id'] == 'test_1'
    
    @patch('src.rag_utils.SentenceTransformer')
    @patch('src.rag_utils._extract_text_file')
    @patch('src.rag_utils._split_text')
    def test_ingest_with_namespace(self, mock_split_text, mock_extract_text, mock_transformer):
        """Test ingesting documents with custom namespace."""
        # Setup mocks
        mock_model = Mock()
        mock_transformer.return_value = mock_model
        mock_extract_text.return_value = "Test content"
        mock_split_text.return_value = ["chunk1"]
        
        mock_embeddings = np.array([[0.1, 0.2]])
        mock_model.encode.return_value = mock_embeddings
        
        mock_index = Mock()
        
        # Create temporary directory with test file
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.txt"
            test_file.write_text("test content")
            
            ingest_documents(
                directory=temp_dir,
                embeddings_model="test-model",
                index=mock_index,
                namespace="custom-namespace"
            )
        
        # Verify upsert called with custom namespace
        upsert_call = mock_index.upsert.call_args
        assert upsert_call[1]['namespace'] == 'custom-namespace'
    
    @patch('src.rag_utils.SentenceTransformer')
    @patch('src.rag_utils._extract_text_file')
    @patch('src.rag_utils._split_text')
    def test_ingest_with_custom_chunk_params(self, mock_split_text, mock_extract_text, mock_transformer):
        """Test ingesting documents with custom chunk parameters."""
        # Setup mocks
        mock_model = Mock()
        mock_transformer.return_value = mock_model
        mock_extract_text.return_value = "Test content"
        mock_split_text.return_value = ["chunk1"]
        
        mock_embeddings = np.array([[0.1, 0.2]])
        mock_model.encode.return_value = mock_embeddings
        
        mock_index = Mock()
        
        # Create temporary directory with test file
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.txt"
            test_file.write_text("test content")
            
            ingest_documents(
                directory=temp_dir,
                embeddings_model="test-model",
                index=mock_index,
                chunk_size=2000,
                chunk_overlap=200
            )
        
        # Verify split_text called with custom parameters
        mock_split_text.assert_called_once_with("Test content", 2000, 200)
    
    @patch('src.rag_utils.SentenceTransformer')
    def test_ingest_unsupported_file_type(self, mock_transformer):
        """Test ingesting unsupported file types."""
        mock_model = Mock()
        mock_transformer.return_value = mock_model
        mock_index = Mock()
        
        # Create temporary directory with unsupported file
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.docx"
            test_file.write_text("unsupported file")
            
            ingest_documents(
                directory=temp_dir,
                embeddings_model="test-model",
                index=mock_index
            )
        
        # Should not call upsert for unsupported files
        mock_index.upsert.assert_not_called()


class TestQueryContext:
    """Test cases for query_context function."""
    
    @patch('src.rag_utils.SentenceTransformer')
    def test_query_context_success(self, mock_transformer):
        """Test successful context query."""
        # Setup mocks
        mock_model = Mock()
        mock_transformer.return_value = mock_model
        
        # Create mock embedding that has .tolist() method
        import numpy as np
        mock_embedding = np.array([[0.1, 0.2, 0.3]])
        mock_model.encode.return_value = mock_embedding
        
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
    
    @patch('src.rag_utils.SentenceTransformer')
    def test_query_context_with_namespace(self, mock_transformer):
        """Test context query with custom namespace."""
        # Setup mocks
        mock_model = Mock()
        mock_transformer.return_value = mock_model
        
        # Create mock embedding that has .tolist() method
        mock_embedding = np.array([[0.1, 0.2, 0.3]])
        mock_model.encode.return_value = mock_embedding
        
        mock_index = Mock()
        mock_index.query.return_value = {
            'matches': [
                {
                    'score': 0.9,
                    'metadata': {
                        'text': 'Test context',
                        'filename': 'test.txt',
                        'chunk_id': 0
                    }
                }
            ]
        }
        
        result = query_context(
            query_text="test query",
            embeddings_model="test-model",
            index=mock_index,
            namespace="custom-namespace",
            top_k=3
        )
        
        # Verify index query called with custom namespace
        mock_index.query.assert_called_once_with(
            vector=[0.1, 0.2, 0.3],
            top_k=3,
            namespace="custom-namespace",
            include_metadata=True
        )
        
        # Verify results
        assert len(result) == 1
        assert result[0]['text'] == 'Test context'
        assert result[0]['filename'] == 'test.txt'
        assert result[0]['chunk_id'] == 0
    
    @patch('src.rag_utils.SentenceTransformer')
    def test_query_context_no_matches(self, mock_transformer):
        """Test context query with no matches."""
        # Setup mocks
        mock_model = Mock()
        mock_transformer.return_value = mock_model
        
        mock_embedding = np.array([[0.1, 0.2, 0.3]])
        mock_model.encode.return_value = mock_embedding
        
        mock_index = Mock()
        mock_index.query.return_value = {'matches': []}
        
        result = query_context(
            query_text="test query",
            embeddings_model="test-model",
            index=mock_index,
            namespace="test",
            top_k=5
        )
        
        # Verify empty results
        assert result == []
    
    @patch('src.rag_utils.SentenceTransformer')
    def test_query_context_exception_handling(self, mock_transformer):
        """Test context query exception handling."""
        # Setup mocks
        mock_model = Mock()
        mock_transformer.return_value = mock_model
        
        mock_embedding = np.array([[0.1, 0.2, 0.3]])
        mock_model.encode.return_value = mock_embedding
        
        mock_index = Mock()
        mock_index.query.side_effect = Exception("Query failed")
        
        # Should handle exception gracefully
        with pytest.raises(Exception):
            query_context(
                query_text="test query",
                embeddings_model="test-model",
                index=mock_index,
                namespace="test",
                top_k=5
            )
    

class TestTextProcessing:
    """Test cases for text processing functions."""
    
    def test_extract_text_file(self):
        """Test text file extraction."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test file content")
            f.flush()
            temp_file_path = f.name
            
        # File is now properly closed, safe to read and delete
        try:
            result = _extract_text_file(Path(temp_file_path))
            assert result == "Test file content"
        finally:
            # Ensure file is deleted even if test fails
            try:
                os.unlink(temp_file_path)
            except (PermissionError, FileNotFoundError):
                pass  # File might already be deleted or still in use
    
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


class TestHelperFunctions:
    """Test cases for helper functions."""
    
    def test_extract_text_file(self):
        """Test text file extraction."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test file content for extraction")
            f.flush()
            temp_file_path = f.name
            
        # File is now properly closed, safe to read and delete
        try:
            result = _extract_text_file(Path(temp_file_path))
            assert result == "Test file content for extraction"
        finally:
            # Ensure file is deleted even if test fails
            try:
                os.unlink(temp_file_path)
            except (PermissionError, FileNotFoundError):
                pass  # File might already be deleted or still in use
    
    def test_extract_text_file_with_encoding(self):
        """Test text file extraction with different encodings."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("Test content with unicode: αβγ")
            f.flush()
            temp_file_path = f.name
            
        try:
            result = _extract_text_file(Path(temp_file_path))
            assert "unicode: αβγ" in result
        finally:
            try:
                os.unlink(temp_file_path)
            except (PermissionError, FileNotFoundError):
                pass
    
    @patch('src.rag_utils.PdfReader')
    def test_extract_pdf_file(self, mock_pdf_reader):
        """Test PDF file extraction."""
        # Mock PDF reader
        mock_page = Mock()
        mock_page.extract_text.return_value = "PDF page content"
        mock_reader = Mock()
        mock_reader.pages = [mock_page]
        mock_pdf_reader.return_value = mock_reader
        
        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            temp_file_path = f.name
            
        try:
            result = _extract_pdf_file(Path(temp_file_path))
            assert result == "PDF page content"
            mock_pdf_reader.assert_called_once()
        finally:
            try:
                os.unlink(temp_file_path)
            except (PermissionError, FileNotFoundError):
                pass
    
    @patch('src.rag_utils.PdfReader')
    def test_extract_pdf_file_multiple_pages(self, mock_pdf_reader):
        """Test PDF file extraction with multiple pages."""
        # Mock PDF reader with multiple pages
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "Page 1 content"
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "Page 2 content"
        mock_reader = Mock()
        mock_reader.pages = [mock_page1, mock_page2]
        mock_pdf_reader.return_value = mock_reader
        
        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            temp_file_path = f.name
            
        try:
            result = _extract_pdf_file(Path(temp_file_path))
            assert result == "Page 1 contentPage 2 content"
        finally:
            try:
                os.unlink(temp_file_path)
            except (PermissionError, FileNotFoundError):
                pass
    
    def test_split_text_basic(self):
        """Test basic text splitting."""
        text = "This is a test document with multiple sentences. " * 10
        chunks = _split_text(text, chunk_size=100, chunk_overlap=20)
        
        assert len(chunks) > 1
        assert len(chunks[0]) <= 100
        assert len(chunks[-1]) <= 100
    
    def test_split_text_with_overlap(self):
        """Test text splitting with overlap."""
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        chunks = _split_text(text, chunk_size=10, chunk_overlap=5)
        
        assert len(chunks) > 1
        # Check that overlap is working (chunks should share characters)
        if len(chunks) > 1:
            # Last 5 chars of first chunk should be first 5 chars of second chunk
            assert chunks[0][-5:] == chunks[1][:5]
    
    def test_split_text_short_text(self):
        """Test splitting text shorter than chunk size."""
        text = "Short text"
        chunks = _split_text(text, chunk_size=100, chunk_overlap=10)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_split_text_empty_text(self):
        """Test splitting empty text."""
        text = ""
        chunks = _split_text(text, chunk_size=100, chunk_overlap=10)
        
        assert len(chunks) == 1
        assert chunks[0] == ""
    
    def test_split_text_zero_overlap(self):
        """Test splitting with zero overlap."""
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        chunks = _split_text(text, chunk_size=10, chunk_overlap=0)
        
        assert len(chunks) == 3  # 26 chars / 10 = 2.6, so 3 chunks
        assert chunks[0] == "ABCDEFGHIJ"
        assert chunks[1] == "KLMNOPQRST"
        assert chunks[2] == "UVWXYZ"

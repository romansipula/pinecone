"""
Basic test without sentence_transformers to verify pinecone API works
"""
import sys
sys.path.insert(0, 'src')

from unittest.mock import Mock, patch
from src.rag_utils import init_pinecone

def test_init_pinecone():
    """Test basic pinecone initialization"""
    # Test with mocked Pinecone
    with patch('src.rag_utils.Pinecone') as mock_pinecone_class:
        mock_pc = Mock()
        mock_index = Mock()
        mock_pinecone_class.return_value = mock_pc
        mock_pc.has_index.return_value = False
        mock_pc.create_index.return_value = None
        mock_pc.Index.return_value = mock_index
        
        result = init_pinecone(
            api_key="test-key",
            index_name="test-index"
        )
        
        assert result == mock_index
        mock_pc.create_index.assert_called_once()
        print("✓ Basic pinecone init test passed")

if __name__ == "__main__":
    test_init_pinecone()
    print("All basic tests passed!")

"""
Tests for agent classes.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.agents.query_agent import QueryAgent
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.generation_agent import GenerationAgent


class TestQueryAgent:
    """Test cases for QueryAgent."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_index = Mock()
        self.agent = QueryAgent(
            embeddings_model="test-model",
            index=self.mock_index,
            namespace="test",
            top_k=3
        )
    
    @patch('src.agents.query_agent.query_context')
    def test_query_success(self, mock_query_context):
        """Test successful query execution."""
        # Mock response
        mock_contexts = [
            {
                'text': 'Test context 1',
                'score': 0.9,
                'filename': 'test1.txt',
                'chunk_id': 0
            },
            {
                'text': 'Test context 2',
                'score': 0.8,
                'filename': 'test2.txt',
                'chunk_id': 1
            }
        ]
        mock_query_context.return_value = mock_contexts
        
        # Execute query
        result = self.agent.query("test query")
        
        # Assertions
        assert result == mock_contexts
        mock_query_context.assert_called_once_with(
            query_text="test query",
            embeddings_model="test-model",
            index=self.mock_index,
            namespace="test",
            top_k=3
        )
    
    @patch('src.agents.query_agent.query_context')
    def test_query_exception(self, mock_query_context):
        """Test query with exception."""
        mock_query_context.side_effect = Exception("Test error")
        
        result = self.agent.query("test query")
        
        assert result == []
    
    def test_format_contexts(self):
        """Test context formatting."""
        contexts = [
            {
                'text': 'Test context 1',
                'filename': 'test1.txt'
            },
            {
                'text': 'Test context 2',
                'filename': 'test2.txt'
            }
        ]
        
        result = self.agent.format_contexts(contexts)
        
        assert "Context 1 (from test1.txt)" in result
        assert "Context 2 (from test2.txt)" in result
        assert "Test context 1" in result
        assert "Test context 2" in result
    
    def test_format_contexts_empty(self):
        """Test formatting empty contexts."""
        result = self.agent.format_contexts([])
        assert result == "No relevant context found."


class TestRetrievalAgent:
    """Test cases for RetrievalAgent."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_index = Mock()
        self.agent = RetrievalAgent(
            embeddings_model="test-model",
            index=self.mock_index,
            namespace="test"
        )
    
    @patch('src.agents.retrieval_agent.query_context')
    def test_retrieve_success(self, mock_query_context):
        """Test successful retrieval."""
        mock_contexts = [
            {'text': 'Test 1', 'score': 0.9, 'filename': 'test1.txt', 'chunk_id': 0},
            {'text': 'Test 2', 'score': 0.8, 'filename': 'test2.txt', 'chunk_id': 1},
            {'text': 'Test 3', 'score': 0.6, 'filename': 'test3.txt', 'chunk_id': 2}
        ]
        mock_query_context.return_value = mock_contexts
        
        result = self.agent.retrieve("test query", top_k=5, score_threshold=0.7)
        
        # Should filter out the context with score 0.6
        assert len(result) == 2
        assert all(ctx['score'] >= 0.7 for ctx in result)
    
    def test_get_context_summary(self):
        """Test context summary generation."""
        contexts = [
            {'text': 'Test 1', 'score': 0.9, 'filename': 'test1.txt'},
            {'text': 'Test 2', 'score': 0.8, 'filename': 'test2.txt'},
            {'text': 'Test 3', 'score': 0.7, 'filename': 'test1.txt'}
        ]
        
        summary = self.agent.get_context_summary(contexts)
        
        assert summary['total'] == 3
        assert summary['avg_score'] == 0.8
        assert set(summary['sources']) == {'test1.txt', 'test2.txt'}
        assert summary['score_range']['min'] == 0.7
        assert summary['score_range']['max'] == 0.9
    
    def test_get_context_summary_empty(self):
        """Test summary for empty contexts."""
        summary = self.agent.get_context_summary([])
        
        assert summary['total'] == 0
        assert summary['avg_score'] == 0.0
        assert summary['sources'] == []


class TestGenerationAgent:
    """Test cases for GenerationAgent."""
    
    def setup_method(self):
        """Setup test fixtures."""
        with patch('src.agents.generation_agent.OpenAI'):
            self.agent = GenerationAgent(
                openai_api_key="test-key",
                model="gpt-3.5-turbo"
            )
    
    @patch('src.agents.generation_agent.OpenAI')
    def test_generate_response_success(self, mock_openai):
        """Test successful response generation."""
        # Mock OpenAI response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        agent = GenerationAgent(openai_api_key="test-key")
        contexts = [
            {'text': 'Test context', 'filename': 'test.txt'}
        ]
        
        result = agent.generate_response("Test query", contexts)
        
        assert result == "Test response"
        mock_client.chat.completions.create.assert_called_once()
    
    def test_format_contexts(self):
        """Test context formatting for prompts."""
        contexts = [
            {'text': 'Context 1', 'filename': 'test1.txt'},
            {'text': 'Context 2', 'filename': 'test2.txt'}
        ]
        
        result = self.agent._format_contexts(contexts)
        
        assert "[Context 1]" in result
        assert "[Context 2]" in result
        assert "Source: test1.txt" in result
        assert "Source: test2.txt" in result
    
    def test_format_contexts_empty(self):
        """Test formatting empty contexts."""
        result = self.agent._format_contexts([])
        assert result == "No relevant context available."

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
    
    @patch('src.agents.query_agent.query_context')
    def test_query_with_parameters(self, mock_query_context):
        """Test query with custom parameters."""
        mock_contexts = [
            {'text': 'Test context', 'score': 0.9, 'filename': 'test.txt', 'chunk_id': 0}
        ]
        mock_query_context.return_value = mock_contexts
        
        # Create agent with custom parameters
        agent = QueryAgent(
            embeddings_model="custom-model",
            index=self.mock_index,
            namespace="custom-namespace",
            top_k=10
        )
        
        result = agent.query("test query")
        
        assert result == mock_contexts
        mock_query_context.assert_called_once_with(
            query_text="test query",
            embeddings_model="custom-model",
            index=self.mock_index,
            namespace="custom-namespace",
            top_k=10
        )
    
    def test_format_contexts_with_multiple_sources(self):
        """Test context formatting with multiple sources."""
        contexts = [
            {'text': 'First context', 'filename': 'doc1.txt'},
            {'text': 'Second context', 'filename': 'doc2.txt'},
            {'text': 'Third context', 'filename': 'doc1.txt'}
        ]
        
        result = self.agent.format_contexts(contexts)
        
        assert "Context 1 (from doc1.txt)" in result
        assert "Context 2 (from doc2.txt)" in result
        assert "Context 3 (from doc1.txt)" in result
        assert "First context" in result
        assert "Second context" in result
        assert "Third context" in result
    
    def test_format_contexts_with_long_text(self):
        """Test context formatting with long text."""
        long_text = "This is a very long context that contains multiple sentences and should be formatted properly. " * 5
        contexts = [
            {'text': long_text, 'filename': 'long_doc.txt'}
        ]
        
        result = self.agent.format_contexts(contexts)
        
        assert "Context 1 (from long_doc.txt)" in result
        assert long_text in result


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
    
    @patch('src.agents.retrieval_agent.query_context')
    def test_retrieve_with_custom_parameters(self, mock_query_context):
        """Test retrieval with custom parameters."""
        mock_contexts = [
            {'text': 'Test 1', 'score': 0.95, 'filename': 'test1.txt', 'chunk_id': 0},
            {'text': 'Test 2', 'score': 0.85, 'filename': 'test2.txt', 'chunk_id': 1},
            {'text': 'Test 3', 'score': 0.75, 'filename': 'test3.txt', 'chunk_id': 2}
        ]
        mock_query_context.return_value = mock_contexts
        
        result = self.agent.retrieve("test query", top_k=10, score_threshold=0.8)
        
        # Should filter out the context with score 0.75
        assert len(result) == 2
        assert all(ctx['score'] >= 0.8 for ctx in result)
        mock_query_context.assert_called_once_with(
            query_text="test query",
            embeddings_model="test-model",
            index=self.mock_index,
            namespace="test",
            top_k=10
        )
    
    @patch('src.agents.retrieval_agent.query_context')
    def test_retrieve_no_results(self, mock_query_context):
        """Test retrieval with no results."""
        mock_query_context.return_value = []
        
        result = self.agent.retrieve("test query", top_k=5, score_threshold=0.7)
        
        assert result == []
    
    @patch('src.agents.retrieval_agent.query_context')
    def test_retrieve_all_filtered_out(self, mock_query_context):
        """Test retrieval where all results are filtered out by score threshold."""
        mock_contexts = [
            {'text': 'Test 1', 'score': 0.5, 'filename': 'test1.txt', 'chunk_id': 0},
            {'text': 'Test 2', 'score': 0.4, 'filename': 'test2.txt', 'chunk_id': 1}
        ]
        mock_query_context.return_value = mock_contexts
        
        result = self.agent.retrieve("test query", top_k=5, score_threshold=0.7)
        
        assert result == []
    
    def test_get_context_summary(self):
        """Test context summary generation."""
        contexts = [
            {'text': 'Test 1', 'score': 0.9, 'filename': 'test1.txt'},
            {'text': 'Test 2', 'score': 0.8, 'filename': 'test2.txt'},
            {'text': 'Test 3', 'score': 0.7, 'filename': 'test1.txt'}
        ]
        
        summary = self.agent.get_context_summary(contexts)
        
        assert summary['total'] == 3
        assert abs(summary['avg_score'] - 0.8) < 0.001  # Use approximate comparison
        assert set(summary['sources']) == {'test1.txt', 'test2.txt'}
        assert summary['score_range']['min'] == 0.7
        assert summary['score_range']['max'] == 0.9
    
    def test_get_context_summary_empty(self):
        """Test summary for empty contexts."""
        summary = self.agent.get_context_summary([])
        
        assert summary['total'] == 0
        assert summary['avg_score'] == 0.0
        assert summary['sources'] == []
    
    def test_get_context_summary_with_duplicates(self):
        """Test context summary with duplicate sources."""
        contexts = [
            {'text': 'Test 1', 'score': 0.9, 'filename': 'test1.txt'},
            {'text': 'Test 2', 'score': 0.8, 'filename': 'test1.txt'},
            {'text': 'Test 3', 'score': 0.7, 'filename': 'test2.txt'}
        ]
        
        summary = self.agent.get_context_summary(contexts)
        
        assert summary['total'] == 3
        assert abs(summary['avg_score'] - 0.8) < 0.001
        assert set(summary['sources']) == {'test1.txt', 'test2.txt'}
        assert summary['score_range']['min'] == 0.7
        assert summary['score_range']['max'] == 0.9
    
    def test_get_context_summary_single_item(self):
        """Test context summary with single item."""
        contexts = [
            {'text': 'Single test', 'score': 0.85, 'filename': 'single.txt'}
        ]
        
        summary = self.agent.get_context_summary(contexts)
        
        assert summary['total'] == 1
        assert summary['avg_score'] == 0.85
        assert summary['sources'] == ['single.txt']
        assert summary['score_range']['min'] == 0.85
        assert summary['score_range']['max'] == 0.85


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
    
    @patch('src.agents.generation_agent.OpenAI')
    def test_generate_response_with_custom_model(self, mock_openai):
        """Test response generation with custom model."""
        # Mock OpenAI response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Custom model response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        agent = GenerationAgent(openai_api_key="test-key", model="gpt-4")
        contexts = [
            {'text': 'Test context', 'filename': 'test.txt'}
        ]
        
        result = agent.generate_response("Test query", contexts)
        
        assert result == "Custom model response"
        # Check that the correct model was used
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]['model'] == 'gpt-4'
    
    @patch('src.agents.generation_agent.OpenAI')
    def test_generate_response_exception_handling(self, mock_openai):
        """Test response generation exception handling."""
        # Mock OpenAI to raise an exception
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client
        
        agent = GenerationAgent(openai_api_key="test-key")
        contexts = [
            {'text': 'Test context', 'filename': 'test.txt'}
        ]
        
        result = agent.generate_response("Test query", contexts)
        
        # Should return error message instead of crashing
        assert "Error generating response" in result or result == ""
    
    @patch('src.agents.generation_agent.OpenAI')
    def test_generate_response_with_multiple_contexts(self, mock_openai):
        """Test response generation with multiple contexts."""
        # Mock OpenAI response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response with multiple contexts"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        agent = GenerationAgent(openai_api_key="test-key")
        contexts = [
            {'text': 'First context', 'filename': 'doc1.txt'},
            {'text': 'Second context', 'filename': 'doc2.txt'},
            {'text': 'Third context', 'filename': 'doc3.txt'}
        ]
        
        result = agent.generate_response("Test query", contexts)
        
        assert result == "Response with multiple contexts"
        # Verify that all contexts were included in the prompt
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]['messages']
        user_message = next(msg for msg in messages if msg['role'] == 'user')
        assert 'First context' in user_message['content']
        assert 'Second context' in user_message['content']
        assert 'Third context' in user_message['content']
    
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
    
    def test_format_contexts_with_scores(self):
        """Test context formatting with scores."""
        contexts = [
            {'text': 'Context 1', 'filename': 'test1.txt', 'score': 0.95},
            {'text': 'Context 2', 'filename': 'test2.txt', 'score': 0.87}
        ]
        
        result = self.agent._format_contexts(contexts)
        
        assert "[Context 1]" in result
        assert "[Context 2]" in result
        assert "Source: test1.txt" in result
        assert "Source: test2.txt" in result
        assert "Context 1" in result
        assert "Context 2" in result
    
    def test_format_contexts_special_characters(self):
        """Test context formatting with special characters."""
        contexts = [
            {'text': 'Context with "quotes" and symbols: @#$%', 'filename': 'special.txt'}
        ]
        
        result = self.agent._format_contexts(contexts)
        
        assert "[Context 1]" in result
        assert "Source: special.txt" in result
        assert 'Context with "quotes" and symbols: @#$%' in result

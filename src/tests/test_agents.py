"""
Unit tests for agent modules.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
import tempfile
import os
from pathlib import Path

from src.agents.query_agent import QueryAgent
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.generation_agent import GenerationAgent


class TestQueryAgent:
    """Test cases for QueryAgent class."""
    
    def test_init(self):
        """Test QueryAgent initialization."""
        mock_index = Mock()
        agent = QueryAgent(
            embeddings_model="test-model",
            index=mock_index,
            namespace="custom",
            top_k=10
        )
        
        assert agent.embeddings_model == "test-model"
        assert agent.index == mock_index
        assert agent.namespace == "custom"
        assert agent.top_k == 10
    
    def test_init_with_defaults(self):
        """Test QueryAgent initialization with default parameters."""
        mock_index = Mock()
        agent = QueryAgent(
            embeddings_model="test-model",
            index=mock_index
        )
        
        assert agent.namespace == "default"
        assert agent.top_k == 5
    
    @patch('src.agents.query_agent.query_context')
    def test_query_success(self, mock_query_context):
        """Test successful query execution."""
        # Setup
        mock_index = Mock()
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
        
        agent = QueryAgent(
            embeddings_model="test-model",
            index=mock_index,
            namespace="test",
            top_k=2
        )
        
        # Execute
        result = agent.query("test query")
        
        # Verify
        mock_query_context.assert_called_once_with(
            query_text="test query",
            embeddings_model="test-model",
            index=mock_index,
            namespace="test",
            top_k=2
        )
        assert result == mock_contexts
    
    @patch('src.agents.query_agent.query_context')
    def test_query_with_exception(self, mock_query_context):
        """Test query handling when exception occurs."""
        # Setup
        mock_index = Mock()
        mock_query_context.side_effect = Exception("Query failed")
        
        agent = QueryAgent(
            embeddings_model="test-model",
            index=mock_index
        )
        
        # Execute
        result = agent.query("test query")
        
        # Verify - should return empty list on exception
        assert result == []
    
    def test_format_contexts_with_data(self):
        """Test formatting contexts with data."""
        mock_index = Mock()
        agent = QueryAgent(
            embeddings_model="test-model",
            index=mock_index
        )
        
        contexts = [
            {
                'text': 'First context text',
                'filename': 'doc1.txt',
                'score': 0.9,
                'chunk_id': 0
            },
            {
                'text': 'Second context text',
                'filename': 'doc2.txt',
                'score': 0.8,
                'chunk_id': 1
            }
        ]
        
        result = agent.format_contexts(contexts)
        
        expected = (
            "Context 1 (from doc1.txt):\nFirst context text\n\n"
            "Context 2 (from doc2.txt):\nSecond context text"
        )
        assert result == expected
    
    def test_format_contexts_empty(self):
        """Test formatting empty contexts."""
        mock_index = Mock()
        agent = QueryAgent(
            embeddings_model="test-model",
            index=mock_index
        )
        
        result = agent.format_contexts([])
        
        assert result == "No relevant context found."


class TestRetrievalAgent:
    """Test cases for RetrievalAgent class."""
    
    def test_init(self):
        """Test RetrievalAgent initialization."""
        mock_index = Mock()
        agent = RetrievalAgent(
            embeddings_model="test-model",
            index=mock_index,
            namespace="custom"
        )
        
        assert agent.embeddings_model == "test-model"
        assert agent.index == mock_index
        assert agent.namespace == "custom"
    
    def test_init_with_defaults(self):
        """Test RetrievalAgent initialization with defaults."""
        mock_index = Mock()
        agent = RetrievalAgent(
            embeddings_model="test-model",
            index=mock_index
        )
        
        assert agent.namespace == "default"
    
    @patch('src.agents.retrieval_agent.query_context')
    def test_retrieve_success(self, mock_query_context):
        """Test successful document retrieval."""
        # Setup
        mock_index = Mock()
        mock_contexts = [
            {
                'text': 'High score context',
                'score': 0.95,
                'filename': 'test1.txt',
                'chunk_id': 0
            },
            {
                'text': 'Medium score context',
                'score': 0.75,
                'filename': 'test2.txt',
                'chunk_id': 1
            },
            {
                'text': 'Low score context',
                'score': 0.45,
                'filename': 'test3.txt',
                'chunk_id': 2
            }
        ]
        mock_query_context.return_value = mock_contexts
        
        agent = RetrievalAgent(
            embeddings_model="test-model",
            index=mock_index,
            namespace="test"
        )
        
        # Execute
        result = agent.retrieve(
            query_text="test query",
            top_k=3,
            score_threshold=0.5
        )
        
        # Verify
        mock_query_context.assert_called_once_with(
            query_text="test query",
            embeddings_model="test-model",
            index=mock_index,
            namespace="test",
            top_k=3
        )
        
        # Should filter out low score context
        assert len(result) == 2
        assert result[0]['score'] == 0.95
        assert result[1]['score'] == 0.75
    
    @patch('src.agents.retrieval_agent.query_context')
    def test_retrieve_with_score_threshold(self, mock_query_context):
        """Test retrieval with score threshold filtering."""
        # Setup
        mock_index = Mock()
        mock_contexts = [
            {'text': 'Context 1', 'score': 0.9, 'filename': 'test1.txt', 'chunk_id': 0},
            {'text': 'Context 2', 'score': 0.6, 'filename': 'test2.txt', 'chunk_id': 1},
            {'text': 'Context 3', 'score': 0.3, 'filename': 'test3.txt', 'chunk_id': 2}
        ]
        mock_query_context.return_value = mock_contexts
        
        agent = RetrievalAgent(
            embeddings_model="test-model",
            index=mock_index
        )
        
        # Execute with high threshold
        result = agent.retrieve(
            query_text="test query",
            score_threshold=0.7
        )
        
        # Verify - only one context should pass threshold
        assert len(result) == 1
        assert result[0]['score'] == 0.9
    
    @patch('src.agents.retrieval_agent.query_context')
    def test_retrieve_with_exception(self, mock_query_context):
        """Test retrieval when exception occurs."""
        # Setup
        mock_index = Mock()
        mock_query_context.side_effect = Exception("Retrieval failed")
        
        agent = RetrievalAgent(
            embeddings_model="test-model",
            index=mock_index
        )
        
        # Execute
        result = agent.retrieve(query_text="test query")
        
        # Verify - should return empty list on exception
        assert result == []
    
    def test_get_context_summary_with_data(self):
        """Test getting summary of contexts with data."""
        mock_index = Mock()
        agent = RetrievalAgent(
            embeddings_model="test-model",
            index=mock_index
        )
        
        contexts = [
            {'text': 'Context 1', 'score': 0.9, 'filename': 'doc1.txt', 'chunk_id': 0},
            {'text': 'Context 2', 'score': 0.8, 'filename': 'doc2.txt', 'chunk_id': 1},
            {'text': 'Context 3', 'score': 0.7, 'filename': 'doc1.txt', 'chunk_id': 2}
        ]
        
        result = agent.get_context_summary(contexts)
        
        assert result['total'] == 3
        assert result['avg_score'] == pytest.approx(0.8, rel=1e-6)  # (0.9 + 0.8 + 0.7) / 3
        assert set(result['sources']) == {'doc1.txt', 'doc2.txt'}
        assert result['score_range']['min'] == 0.7
        assert result['score_range']['max'] == 0.9
    
    def test_get_context_summary_empty(self):
        """Test getting summary of empty contexts."""
        mock_index = Mock()
        agent = RetrievalAgent(
            embeddings_model="test-model",
            index=mock_index
        )
        
        result = agent.get_context_summary([])
        
        assert result['total'] == 0
        assert result['avg_score'] == 0.0
        assert result['sources'] == []


class TestGenerationAgent:
    """Test cases for GenerationAgent class."""
    
    @patch('src.agents.generation_agent.OpenAI')
    def test_init(self, mock_openai):
        """Test GenerationAgent initialization."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        agent = GenerationAgent(
            openai_api_key="test-key",
            model="gpt-4",
            temperature=0.5,
            max_tokens=2000
        )
        
        assert agent.client == mock_client
        assert agent.model == "gpt-4"
        assert agent.temperature == 0.5
        assert agent.max_tokens == 2000
        mock_openai.assert_called_once_with(api_key="test-key")
    
    @patch('src.agents.generation_agent.OpenAI')
    def test_init_with_defaults(self, mock_openai):
        """Test GenerationAgent initialization with default parameters."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        agent = GenerationAgent(openai_api_key="test-key")
        
        assert agent.model == "gpt-3.5-turbo"
        assert agent.temperature == 0.7
        assert agent.max_tokens == 1000
    
    @patch('src.agents.generation_agent.OpenAI')
    def test_generate_response_success(self, mock_openai):
        """Test successful response generation."""
        # Setup OpenAI client mock
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Mock response
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Generated response text"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create agent
        agent = GenerationAgent(openai_api_key="test-key")
        
        # Mock template loading
        with patch.object(agent, '_load_template') as mock_load_template:
            # Setup mock templates
            mock_system_template = Mock()
            mock_system_template.render.return_value = "System prompt"
            mock_user_template = Mock()
            mock_user_template.render.return_value = "User prompt with context"
            
            mock_load_template.side_effect = [mock_system_template, mock_user_template]
            
            # Re-initialize templates
            agent.system_template = mock_system_template
            agent.user_template = mock_user_template
            
            # Test data
            contexts = [
                {
                    'text': 'Context text 1',
                    'filename': 'doc1.txt',
                    'score': 0.9,
                    'chunk_id': 0
                }
            ]
            
            # Execute
            result = agent.generate_response(
                query="test question",
                contexts=contexts
            )
            
            # Verify
            assert result == "Generated response text"
            mock_client.chat.completions.create.assert_called_once()
            
            # Check that templates were rendered
            mock_system_template.render.assert_called_once()
            mock_user_template.render.assert_called_once()
    
    @patch('src.agents.generation_agent.OpenAI')
    def test_generate_response_with_conversation_history(self, mock_openai):
        """Test response generation with conversation history."""
        # Setup
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Response with history"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        
        agent = GenerationAgent(openai_api_key="test-key")
        
        # Mock template loading
        with patch.object(agent, '_load_template') as mock_load_template:
            mock_system_template = Mock()
            mock_system_template.render.return_value = "System prompt"
            mock_user_template = Mock()
            mock_user_template.render.return_value = "User prompt"
            
            mock_load_template.side_effect = [mock_system_template, mock_user_template]
            
            agent.system_template = mock_system_template
            agent.user_template = mock_user_template
            
            # Test with conversation history
            conversation_history = [
                {"role": "user", "content": "Previous question"},
                {"role": "assistant", "content": "Previous answer"}
            ]
            
            result = agent.generate_response(
                query="follow up question",
                contexts=[],
                conversation_history=conversation_history
            )
            
            # Verify
            assert result == "Response with history"
            
            # Check that conversation history was included in messages
            call_args = mock_client.chat.completions.create.call_args
            messages = call_args[1]['messages']
            
            # Should have system + history + user message
            assert len(messages) >= 4  # system + 2 history + user
            assert messages[1]["content"] == "Previous question"
            assert messages[2]["content"] == "Previous answer"
    
    @patch('src.agents.generation_agent.OpenAI')
    def test_generate_response_with_exception(self, mock_openai):
        """Test response generation when exception occurs."""
        # Setup
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        agent = GenerationAgent(openai_api_key="test-key")
        
        # Mock template loading
        with patch.object(agent, '_load_template') as mock_load_template:
            mock_system_template = Mock()
            mock_system_template.render.return_value = "System prompt"
            mock_user_template = Mock()
            mock_user_template.render.return_value = "User prompt"
            
            mock_load_template.side_effect = [mock_system_template, mock_user_template]
            
            agent.system_template = mock_system_template
            agent.user_template = mock_user_template
            
            # Execute
            result = agent.generate_response(
                query="test question",
                contexts=[]
            )
            
            # Verify - should return error message
            assert "error generating a response" in result.lower()
    
    @patch('src.agents.generation_agent.OpenAI')
    def test_format_contexts(self, mock_openai):
        """Test formatting contexts for prompt inclusion."""
        agent = GenerationAgent(openai_api_key="test-key")
        
        contexts = [
            {
                'text': 'First context text',
                'filename': 'doc1.txt',
                'score': 0.9,
                'chunk_id': 0
            },
            {
                'text': 'Second context text',
                'filename': 'doc2.txt',
                'score': 0.8,
                'chunk_id': 1
            }
        ]
        
        result = agent._format_contexts(contexts)
        
        expected = (
            "[Context 1] (Source: doc1.txt)\n"
            "First context text\n\n"
            "[Context 2] (Source: doc2.txt)\n"
            "Second context text"
        )
        assert result == expected
    
    @patch('src.agents.generation_agent.OpenAI')
    def test_format_contexts_empty(self, mock_openai):
        """Test formatting empty contexts."""
        agent = GenerationAgent(openai_api_key="test-key")
        
        result = agent._format_contexts([])
        
        assert result == "No relevant context available."
    
    @patch('src.agents.generation_agent.OpenAI')
    @patch('builtins.open', new_callable=mock_open, read_data="Test template content")
    def test_load_template_success(self, mock_file, mock_openai):
        """Test successful template loading."""
        agent = GenerationAgent(openai_api_key="test-key")
        
        template = agent._load_template("test_template.jinja2")
        
        # Verify file was opened and template created (called multiple times during init)
        assert mock_file.call_count >= 1
        assert template is not None
    
    @patch('src.agents.generation_agent.OpenAI')
    @patch('builtins.open', side_effect=FileNotFoundError())
    def test_load_template_file_not_found(self, mock_file, mock_openai):
        """Test template loading when file not found."""
        agent = GenerationAgent(openai_api_key="test-key")
        
        template = agent._load_template("missing_template.jinja2")
        
        # Should return default template
        assert template is not None
        # Should use default content
        rendered = template.render()
        assert len(rendered) > 0
    
    @patch('src.agents.generation_agent.OpenAI')
    def test_get_default_template_system(self, mock_openai):
        """Test getting default system template."""
        agent = GenerationAgent(openai_api_key="test-key")
        
        result = agent._get_default_template("system_prompt.jinja2")
        
        assert "helpful AI assistant" in result
        assert "context" in result.lower()
    
    @patch('src.agents.generation_agent.OpenAI')
    def test_get_default_template_user(self, mock_openai):
        """Test getting default user template."""
        agent = GenerationAgent(openai_api_key="test-key")
        
        result = agent._get_default_template("user_prompt.jinja2")
        
        assert "{{ query }}" in result
        assert "{{ contexts }}" in result
    
    @patch('src.agents.generation_agent.OpenAI')
    def test_get_default_template_unknown(self, mock_openai):
        """Test getting default template for unknown filename."""
        agent = GenerationAgent(openai_api_key="test-key")
        
        result = agent._get_default_template("unknown_template.jinja2")
        
        assert result == "Template not found"


# Integration tests
class TestAgentIntegration:
    """Integration tests for agent interactions."""
    
    @patch('src.agents.query_agent.query_context')
    @patch('src.agents.generation_agent.OpenAI')
    def test_query_to_generation_workflow(self, mock_openai, mock_query_context):
        """Test complete workflow from query to generation."""
        # Setup query agent
        mock_index = Mock()
        mock_contexts = [
            {
                'text': 'Relevant context for question',
                'score': 0.9,
                'filename': 'doc.txt',
                'chunk_id': 0
            }
        ]
        mock_query_context.return_value = mock_contexts
        
        query_agent = QueryAgent(
            embeddings_model="test-model",
            index=mock_index
        )
        
        # Setup generation agent
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Generated answer based on context"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        
        generation_agent = GenerationAgent(openai_api_key="test-key")
        
        # Mock template loading for generation agent
        with patch.object(generation_agent, '_load_template') as mock_load_template:
            mock_system_template = Mock()
            mock_system_template.render.return_value = "System prompt"
            mock_user_template = Mock()
            mock_user_template.render.return_value = "User prompt"
            
            mock_load_template.side_effect = [mock_system_template, mock_user_template]
            
            generation_agent.system_template = mock_system_template
            generation_agent.user_template = mock_user_template
            
            # Execute workflow
            user_query = "What is the answer?"
            
            # Step 1: Query for contexts
            contexts = query_agent.query(user_query)
            
            # Step 2: Generate response
            response = generation_agent.generate_response(
                query=user_query,
                contexts=contexts
            )
            
            # Verify
            assert contexts == mock_contexts
            assert response == "Generated answer based on context"
            
            # Verify query was called correctly
            mock_query_context.assert_called_once_with(
                query_text=user_query,
                embeddings_model="test-model",
                index=mock_index,
                namespace="default",
                top_k=5
            )
            
            # Verify generation was called
            mock_client.chat.completions.create.assert_called_once()


# Fixtures
@pytest.fixture
def mock_pinecone_index():
    """Fixture providing a mock Pinecone index."""
    index = Mock()
    index.query.return_value = {
        'matches': [
            {
                'id': 'test_doc_0',
                'score': 0.95,
                'metadata': {
                    'text': 'Test context text',
                    'filename': 'test.txt',
                    'chunk_id': 0
                }
            }
        ]
    }
    return index


@pytest.fixture
def sample_contexts():
    """Fixture providing sample context data."""
    return [
        {
            'text': 'First context text',
            'score': 0.9,
            'filename': 'doc1.txt',
            'chunk_id': 0
        },
        {
            'text': 'Second context text',
            'score': 0.8,
            'filename': 'doc2.txt',
            'chunk_id': 1
        }
    ]


@pytest.fixture
def mock_openai_response():
    """Fixture providing a mock OpenAI response."""
    response = Mock()
    choice = Mock()
    message = Mock()
    message.content = "Mock generated response"
    choice.message = message
    response.choices = [choice]
    return response

"""
Generation agent for creating responses using OpenAI's chat completion API.
"""
from typing import List, Dict, Any, Optional
import logging
import os

from openai import OpenAI
from jinja2 import Template

logger = logging.getLogger(__name__)


class GenerationAgent:
    """Agent responsible for generating responses using retrieved contexts."""
    
    def __init__(
        self,
        openai_api_key: str,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        """
        Initialize GenerationAgent.
        
        Args:
            openai_api_key: OpenAI API key
            model: OpenAI model name
            temperature: Generation temperature
            max_tokens: Maximum tokens in response
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Load prompt templates
        self.system_template = self._load_template("system_prompt.jinja2")
        self.user_template = self._load_template("user_prompt.jinja2")
    
    def generate_response(
        self,
        query: str,
        contexts: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate response using retrieved contexts.
        
        Args:
            query: User query
            contexts: Retrieved contexts
            conversation_history: Previous conversation messages
            
        Returns:
            Generated response
        """
        try:
            # Format contexts for prompt
            formatted_contexts = self._format_contexts(contexts)
            
            # Render templates
            system_prompt = self.system_template.render()
            user_prompt = self.user_template.render(
                query=query,
                contexts=formatted_contexts
            )
            
            # Build messages
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            # Add conversation history if provided
            if conversation_history:
                messages.extend(conversation_history)
            
            messages.append({"role": "user", "content": user_prompt})
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            generated_text = response.choices[0].message.content
            logger.info(f"Generated response with {len(generated_text)} characters")
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error generating a response. Please try again."
    
    def _format_contexts(self, contexts: List[Dict[str, Any]]) -> str:
        """Format contexts for inclusion in prompt."""
        if not contexts:
            return "No relevant context available."
        
        formatted = []
        for i, context in enumerate(contexts, 1):
            formatted.append(
                f"[Context {i}] (Source: {context['filename']})\n"
                f"{context['text']}"
            )
        
        return "\n\n".join(formatted)
    
    def _load_template(self, filename: str) -> Template:
        """Load Jinja2 template from prompts directory."""
        template_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "prompts",
            filename
        )
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
            return Template(template_content)
        except FileNotFoundError:
            logger.warning(f"Template {filename} not found, using default")
            return Template(self._get_default_template(filename))
    
    def _get_default_template(self, filename: str) -> str:
        """Get default template content."""
        if filename == "system_prompt.jinja2":
            return """You are a helpful AI assistant with access to relevant context from documents. 
Use the provided context to answer questions accurately and helpfully. 
If the context doesn't contain enough information to answer a question, say so clearly."""
        
        elif filename == "user_prompt.jinja2":
            return """Question: {{ query }}

Context:
{{ contexts }}

Please provide a comprehensive answer based on the context above."""
        
        return "Template not found"

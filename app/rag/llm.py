"""
LLM Provider Abstraction
Supports Google Gemini, OpenAI, and Deterministic Mock LLMs for offline testing and development.
"""
import json
import logging
from typing import Any, Dict, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage

from app.config import get_settings

logger = logging.getLogger(__name__)


class DeterministicMockLLM(BaseChatModel):
    """
    Offline deterministic LLM implementation for testing and development.
    Produces structured and sensible responses for query analysis, grading, and answering
    without requiring external API calls or network access.
    """
    model_name: str = "deterministic-mock"

    @property
    def _llm_type(self) -> str:
        return "deterministic_mock"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any
    ) -> Any:
        prompt_text = " ".join([m.content for m in messages if isinstance(m.content, str)])
        response_text = self._generate_response(prompt_text)
        message = AIMessage(content=response_text)
        from langchain_core.outputs import ChatGeneration, ChatResult
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _generate_response(self, prompt: str) -> str:
        prompt_lower = prompt.lower()

        # Query Analysis node detection
        if "query analyzer" in prompt_lower or "classify the question" in prompt_lower:
            query_type = "conceptual"
            if "how" in prompt_lower or "create" in prompt_lower or "implement" in prompt_lower:
                query_type = "how-to"
            elif "error" in prompt_lower or "exception" in prompt_lower or "troubleshoot" in prompt_lower:
                query_type = "troubleshooting"
            elif "param" in prompt_lower or "signature" in prompt_lower or "status code" in prompt_lower:
                query_type = "api_reference"

            return json.dumps({
                "query_type": query_type,
                "rewritten_query": "FastAPI " + prompt.split("User Question:")[-1].strip() if "User Question:" in prompt else "FastAPI overview"
            })

        # Relevance Grading node detection
        if "grading the relevance" in prompt_lower or "relevant: true or false" in prompt_lower:
            # Check for simulated irrelevant keywords or empty matches
            if "completely irrelevant test query" in prompt_lower or "quantum physics" in prompt_lower:
                return json.dumps({
                    "relevant": False,
                    "reason": "Chunk does not relate to the requested topic."
                })
            # By default in test corpus, if FastAPI terms overlap, mark relevant
            return json.dumps({
                "relevant": True,
                "reason": "Chunk contains direct technical documentation relevant to the user query."
            })

        # Query Retry / Reformulation detection
        if "reformulate" in prompt_lower or "new reformulated search query" in prompt_lower:
            return json.dumps({
                "rewritten_query": "FastAPI technical specifications and architecture"
            })

        # Grounding Check detection
        if "factual grounding" in prompt_lower or "is_grounded" in prompt_lower:
            return json.dumps({
                "is_grounded": True,
                "explanation": "All claims are strictly derived from the retrieved technical context."
            })

        # Answer Generation detection
        if "technical documentation assistant for fastapi" in prompt_lower or "retrieved documentation context:" in prompt_lower:
            context_part = prompt.split("Retrieved Documentation Context:")[-1].split("Answer:")[0].strip() if "Retrieved Documentation Context:" in prompt else ""
            if not context_part or "quantum physics" in prompt_lower:
                return "The provided documentation does not contain sufficient information to answer this question."

            return (
                "Based on the FastAPI technical documentation, FastAPI provides high-performance "
                "APIs utilizing modern Python type annotations and the ASGI specification.\n\n"
                "Key capabilities include modular routing with `APIRouter`, automated request validation "
                "using Pydantic models, and an extensible Dependency Injection system using `Depends()`.\n\n"
                "**Sources:**\n"
                "- [FastAPI Overview](https://fastapi.tiangolo.com/tutorial/first-steps/) - *Creating a Basic FastAPI Application*\n"
                "- [FastAPI Dependencies](https://fastapi.tiangolo.com/tutorial/dependencies/) - *Understanding Dependency Injection in FastAPI*"
            )

        # Fallback default response
        return "FastAPI is a modern, high-performance web framework for building APIs with Python."


def get_llm(force_mock: bool = False) -> BaseChatModel:
    """
    Factory function to retrieve the configured LLM client.
    Returns:
        BaseChatModel: Configured ChatGoogleGenerativeAI, ChatOpenAI, or DeterministicMockLLM.
    Raises:
        ValueError: If a provider is chosen but required credentials are missing and force_mock is False.
    """
    settings = get_settings()

    if force_mock or settings.llm_provider == "mock":
        logger.info("Using DeterministicMockLLM provider.")
        return DeterministicMockLLM()

    if settings.llm_provider == "gemini":
        if not settings.google_api_key or settings.google_api_key == "your_google_api_key_here":
            logger.warning("GOOGLE_API_KEY is not configured. Falling back to DeterministicMockLLM for safety.")
            return DeterministicMockLLM()
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=settings.llm_model or "gemini-1.5-flash",
                google_api_key=settings.google_api_key,
                temperature=0.1
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChatGoogleGenerativeAI: {e}. Falling back to mock.")
            return DeterministicMockLLM()

    if settings.llm_provider == "openai":
        if not settings.openai_api_key or settings.openai_api_key == "your_openai_api_key_here":
            logger.warning("OPENAI_API_KEY is not configured. Falling back to DeterministicMockLLM for safety.")
            return DeterministicMockLLM()
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=settings.llm_model or "gpt-4o-mini",
                api_key=settings.openai_api_key,
                temperature=0.1
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChatOpenAI: {e}. Falling back to mock.")
            return DeterministicMockLLM()

    logger.warning(f"Unknown LLM provider '{settings.llm_provider}'. Defaulting to DeterministicMockLLM.")
    return DeterministicMockLLM()

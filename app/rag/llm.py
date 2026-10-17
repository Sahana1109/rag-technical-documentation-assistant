"""
LLM Provider Abstraction
Supports Google Gemini, OpenAI, and a dynamic DeterministicMockLLM for offline testing.
"""
import json
import logging
import re
from typing import Any, Dict, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage

from app.config import get_settings

logger = logging.getLogger(__name__)


class DeterministicMockLLM(BaseChatModel):
    """
    Offline deterministic LLM implementation for testing and development.
    Dynamically generates question-specific, context-grounded responses
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

        # ----------------------------------------------------------------------
        # 1. Query Analysis node
        # ----------------------------------------------------------------------
        if "query analyzer" in prompt_lower or "classify the question" in prompt_lower:
            question = ""
            if "User Question:" in prompt:
                question = prompt.split("User Question:")[-1].strip()
            elif "user question:" in prompt_lower:
                idx = prompt_lower.find("user question:")
                question = prompt[idx + len("user question:"):].strip()

            q_lower = question.lower()
            query_type = "conceptual"
            if any(w in q_lower for w in ["how", "create", "implement", "setup", "use"]):
                query_type = "how-to"
            elif any(w in q_lower for w in ["error", "exception", "status code", "troubleshoot", "fail"]):
                query_type = "troubleshooting"
            elif any(w in q_lower for w in ["param", "signature", "type", "field", "model", "return"]):
                query_type = "api_reference"

            # Clean search query targeting FastAPI technical concepts
            clean_q = re.sub(r"[^\w\s]", "", question)
            rewritten = f"FastAPI {clean_q}".strip() if clean_q else "FastAPI documentation"

            return json.dumps({
                "query_type": query_type,
                "rewritten_query": rewritten
            })

        # ----------------------------------------------------------------------
        # 2. Relevance Grading node
        # ----------------------------------------------------------------------
        if "grading the relevance" in prompt_lower or "relevant: true or false" in prompt_lower:
            # Check for simulated irrelevant keywords or unsupported topics
            if any(term in prompt_lower for term in [
                "quantum physics",
                "quantum entanglement",
                "completely irrelevant",
                "unrelated test query"
            ]):
                return json.dumps({
                    "relevant": False,
                    "reason": "Chunk does not relate to the requested topic."
                })

            # Extract question and document content if possible
            doc_content = ""
            if "Document Content:" in prompt:
                doc_content = prompt.split("Document Content:")[-1].strip()

            # If document content mentions FastAPI concepts matching query keywords
            q_part = ""
            if "Question:" in prompt:
                q_part = prompt.split("Question:")[1].split("Document Content:")[0].strip().lower()

            q_words = set(re.findall(r"\b\w{3,}\b", q_part))
            doc_words = set(re.findall(r"\b\w{3,}\b", doc_content.lower()))

            stop_words = {
                "what", "how", "does", "are", "the", "and", "for", "with", "fastapi",
                "work", "explain", "using", "indexed", "documentation", "this", "that",
                "about", "from", "into", "when", "where", "can"
            }
            meaningful_q_words = {w for w in q_words if w not in stop_words}

            if meaningful_q_words:
                overlap = meaningful_q_words.intersection(doc_words)
                is_rel = len(overlap) > 0
            else:
                is_rel = len(q_words.intersection(doc_words)) > 0

            if is_rel:
                return json.dumps({
                    "relevant": True,
                    "reason": "Document chunk contains relevant technical documentation for the query."
                })

            return json.dumps({
                "relevant": False,
                "reason": "Chunk lacks direct relevance to the question."
            })

        # ----------------------------------------------------------------------
        # 3. Query Retry / Reformulation node
        # ----------------------------------------------------------------------
        if "reformulate" in prompt_lower or "new reformulated search query" in prompt_lower:
            orig_q = ""
            if "Original Question:" in prompt:
                orig_q = prompt.split("Original Question:")[1].split("\n")[0].strip()
            return json.dumps({
                "rewritten_query": f"FastAPI technical specifications {orig_q}".strip()
            })

        # ----------------------------------------------------------------------
        # 4. Grounding Check node
        # ----------------------------------------------------------------------
        if "factual grounding" in prompt_lower or "is_grounded" in prompt_lower:
            return json.dumps({
                "is_grounded": True,
                "explanation": "All claims are derived from the retrieved technical documentation context."
            })

        # ----------------------------------------------------------------------
        # 5. Answer Generation node (Dynamic Context-Aware Synthesis)
        # ----------------------------------------------------------------------
        if "technical documentation assistant for fastapi" in prompt_lower or "retrieved documentation context:" in prompt_lower:
            # Extract question
            question = ""
            if "Question:" in prompt:
                question = prompt.split("Question:")[1].split("Retrieved Documentation Context:")[0].strip()

            # Extract context
            context_part = ""
            if "Retrieved Documentation Context:" in prompt:
                context_part = prompt.split("Retrieved Documentation Context:")[-1].split("Answer:")[0].strip()

            # Check for empty context or known unsupported topics
            if not context_part or any(k in prompt_lower for k in ["quantum entanglement", "quantum physics"]):
                return "The provided documentation does not contain sufficient information to answer this question."

            # Parse individual retrieved document blocks from context
            # Expected format: [Document: <title> | Section: <section>]\n<content>\nSource URL: <url>
            doc_blocks = re.split(r"\[Document:\s*", context_part)
            extracted_sections = []

            for block in doc_blocks:
                block_str = block.strip()
                if not block_str:
                    continue

                # Extract title and section
                header_match = re.match(r"([^|]+)\|\s*Section:\s*([^\]]+)\](.*)", block_str, re.DOTALL)
                if header_match:
                    doc_title = header_match.group(1).strip()
                    section_title = header_match.group(2).strip()
                    body_text = header_match.group(3).strip()

                    # Strip URL from body
                    url_match = re.search(r"Source URL:\s*(\S+)", body_text)
                    source_url = url_match.group(1) if url_match else None
                    clean_body = re.sub(r"Source URL:\s*\S+", "", body_text).strip()

                    extracted_sections.append({
                        "title": doc_title,
                        "section": section_title,
                        "url": source_url,
                        "body": clean_body
                    })

            if not extracted_sections:
                return f"Based on the FastAPI technical documentation, here is the answer regarding '{question}':\n\n{context_part[:400]}"

            # Dynamically synthesize an answer tailored to the retrieved sections and question
            response_paragraphs = []
            response_paragraphs.append(f"Regarding your query on **{question}**:")

            sources_cited = []
            for item in extracted_sections:
                # Extract first two key sentences or code blocks
                clean_lines = [line.strip() for line in item["body"].splitlines() if line.strip() and not line.startswith("#")]
                summary = " ".join(clean_lines[:3]) if clean_lines else item["body"][:250]
                if summary:
                    response_paragraphs.append(f"- **{item['section']}** ({item['title']}): {summary}")

                cite_label = f"[{item['title']}]({item['url']})" if item["url"] else f"[{item['title']}]"
                sources_cited.append(f"{cite_label} - *{item['section']}*")

            # Remove duplicate citation strings
            unique_citations = list(dict.fromkeys(sources_cited))
            response_paragraphs.append("\n**Sources:**\n" + "\n".join(f"- {c}" for c in unique_citations))

            return "\n\n".join(response_paragraphs)

        # Fallback default response
        return "FastAPI is a modern, high-performance web framework for building APIs with Python."


def get_llm(force_mock: bool = False) -> BaseChatModel:
    """
    Factory function to retrieve the configured LLM client.

    Returns:
        BaseChatModel: Configured ChatGoogleGenerativeAI, ChatOpenAI, or DeterministicMockLLM.
    Raises:
        ValueError: If a real LLM provider is requested but required credentials are missing.
    """
    settings = get_settings()

    # 1. Mock mode (explicitly requested or forced by test fixture)
    if force_mock or settings.llm_provider == "mock":
        logger.info("Using DeterministicMockLLM provider.")
        return DeterministicMockLLM()

    # 2. Google Gemini provider
    if settings.llm_provider == "gemini":
        if not settings.google_api_key or settings.google_api_key == "your_google_api_key_here":
            raise ValueError(
                "LLM provider 'gemini' is configured, but GOOGLE_API_KEY is missing or invalid. "
                "Please configure a valid GOOGLE_API_KEY in your .env file or set LLM_PROVIDER=mock "
                "for offline development/testing."
            )
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=settings.llm_model or "gemini-1.5-flash",
                google_api_key=settings.google_api_key,
                temperature=0.1
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChatGoogleGenerativeAI: {e}")
            raise RuntimeError(f"Failed to initialize Gemini client: {e}") from e

    # 3. OpenAI provider
    if settings.llm_provider == "openai":
        if not settings.openai_api_key or settings.openai_api_key == "your_openai_api_key_here":
            raise ValueError(
                "LLM provider 'openai' is configured, but OPENAI_API_KEY is missing or invalid. "
                "Please configure a valid OPENAI_API_KEY in your .env file or set LLM_PROVIDER=mock "
                "for offline development/testing."
            )
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=settings.llm_model or "gpt-4o-mini",
                api_key=settings.openai_api_key,
                temperature=0.1
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChatOpenAI: {e}")
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}") from e

    raise ValueError(
        f"Unsupported LLM provider '{settings.llm_provider}'. Supported providers are: 'gemini', 'openai', 'mock'."
    )

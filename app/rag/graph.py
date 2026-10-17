"""
LangGraph Self-Corrective RAG StateGraph Implementation
Orchestrates Query Analysis, Dense Retrieval, Relevance Grading,
Conditional Query Reformulation, Grounded Answer Generation, and Grounding Verification.
"""
import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

from app.config import get_settings
from app.rag.state import GraphState
from app.rag.retriever import retrieve_documents
from app.rag.llm import get_llm
from app.rag.prompts import (
    QUERY_ANALYSIS_PROMPT,
    DOCUMENT_GRADING_PROMPT,
    QUERY_RETRY_PROMPT,
    ANSWER_GENERATION_PROMPT,
    GROUNDING_VERIFICATION_PROMPT,
)

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Helper to extract JSON payload from LLM responses even if wrapped in markdown code blocks."""
    try:
        # Direct parse attempt
        return json.loads(text.strip())
    except Exception:
        pass

    # Regex search for ```json ... ``` or { ... }
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    match_bare = re.search(r"(\{.*\})", text, re.DOTALL)
    if match_bare:
        try:
            return json.loads(match_bare.group(1))
        except Exception:
            pass

    return None


# ------------------------------------------------------------------------------
# LangGraph Nodes
# ------------------------------------------------------------------------------

def query_analysis_node(state: GraphState) -> Dict[str, Any]:
    """
    Node A: Analyzes user query, classifies the type, and rewrites/expands it
    for optimal dense technical documentation retrieval.
    """
    question = state.get("original_question", "").strip()
    logger.info(f"[Node: Query Analysis] Analyzing question: '{question}'")

    llm = get_llm()
    prompt = QUERY_ANALYSIS_PROMPT.format(question=question)

    query_type = "conceptual"
    rewritten_query = question

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        parsed = _extract_json(response.content)
        if parsed:
            query_type = parsed.get("query_type", "conceptual")
            rewritten_query = parsed.get("rewritten_query", question)
    except (ValueError, RuntimeError):
        raise
    except Exception as e:
        logger.warning(f"Query analysis LLM call failed: {e}. Using original question as fallback.")

    return {
        "query_type": query_type,
        "rewritten_query": rewritten_query
    }


def retrieval_node(state: GraphState) -> Dict[str, Any]:
    """
    Node B: Searches persistent ChromaDB collection using semantic embeddings.
    """
    query = state.get("rewritten_query") or state.get("original_question", "")
    logger.info(f"[Node: Retrieval] Searching ChromaDB for: '{query}'")

    docs = retrieve_documents(query)
    return {
        "retrieved_documents": docs
    }


def document_grading_node(state: GraphState) -> Dict[str, Any]:
    """
    Node C: Evaluates relevance of retrieved chunks against the query.
    Filters out irrelevant chunks before answer generation.
    """
    retrieved = state.get("retrieved_documents", [])
    target_question = state.get("original_question") or state.get("rewritten_query", "")
    logger.info(f"[Node: Document Grading] Grading {len(retrieved)} retrieved chunks against question: '{target_question}'.")

    if not retrieved:
        return {
            "graded_documents": [],
            "relevant_documents": []
        }

    llm = get_llm()
    graded_docs = []
    relevant_docs = []

    for doc in retrieved:
        prompt = DOCUMENT_GRADING_PROMPT.format(
            question=target_question,
            document_content=doc.get("content", "")
        )
        is_relevant = False
        reason = "Unable to verify relevance."

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            parsed = _extract_json(response.content)
            if parsed:
                is_relevant = bool(parsed.get("relevant", False))
                reason = parsed.get("reason", "")
            else:
                # Fallback heuristic: check word overlap
                query_words = set(query.lower().split())
                doc_words = set(doc.get("content", "").lower().split())
                is_relevant = len(query_words.intersection(doc_words)) > 0
                reason = "Heuristic keyword overlap fallback."
        except (ValueError, RuntimeError):
            raise
        except Exception as e:
            logger.warning(f"Grading chunk failed: {e}. Defaulting to relevant.")
            is_relevant = True
            reason = f"Grading error fallback: {e}"

        graded_item = {
            **doc,
            "is_relevant": is_relevant,
            "grading_reason": reason
        }
        graded_docs.append(graded_item)
        if is_relevant:
            relevant_docs.append(graded_item)

    logger.info(f"[Node: Document Grading] {len(relevant_docs)} / {len(retrieved)} chunks marked relevant.")
    return {
        "graded_documents": graded_docs,
        "relevant_documents": relevant_docs
    }


def query_retry_node(state: GraphState) -> Dict[str, Any]:
    """
    Node D: Reformulates the query when retrieval failed to yield relevant context.
    Increments retry count to avoid infinite loops.
    """
    original_question = state.get("original_question", "")
    previous_query = state.get("rewritten_query", original_question)
    current_retry = state.get("retry_count", 0)
    max_retries = state.get("maximum_retries", 2)

    new_retry_count = current_retry + 1
    logger.info(f"[Node: Query Retry] Retry attempt {new_retry_count} of {max_retries}.")

    llm = get_llm()
    prompt = QUERY_RETRY_PROMPT.format(
        original_question=original_question,
        previous_query=previous_query,
        retry_count=new_retry_count,
        maximum_retries=max_retries
    )

    new_query = f"{original_question} technical documentation"
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        parsed = _extract_json(response.content)
        if parsed and parsed.get("rewritten_query"):
            new_query = parsed["rewritten_query"]
    except Exception as e:
        logger.warning(f"Query reformulation failed: {e}. Using appended search query.")

    return {
        "rewritten_query": new_query,
        "retry_count": new_retry_count
    }


def generate_answer_node(state: GraphState) -> Dict[str, Any]:
    """
    Node E: Synthesizes a factual, grounded answer strictly from relevant chunks,
    attaching precise citations.
    """
    question = state.get("original_question", "")
    relevant_docs = state.get("relevant_documents", [])
    logger.info(f"[Node: Answer Generation] Generating answer with {len(relevant_docs)} context chunks.")

    if not relevant_docs:
        return {
            "answer": "The provided documentation does not contain sufficient information to answer this question.",
            "citations": [],
            "sufficient_context": False
        }

    # Format context for prompt
    context_blocks = []
    citations = []
    seen_citations = set()

    for i, doc in enumerate(relevant_docs):
        block = (
            f"[Document: {doc.get('document_title', 'Doc')} | "
            f"Section: {doc.get('section', 'General')}]\n"
            f"{doc.get('content', '')}\n"
            f"Source URL: {doc.get('source_url', 'N/A')}"
        )
        context_blocks.append(block)

        # Build deduplicated citations
        cite_key = (doc.get("document_title"), doc.get("section"), doc.get("chunk_index"))
        if cite_key not in seen_citations:
            seen_citations.add(cite_key)
            citations.append({
                "document_title": doc.get("document_title", "FastAPI Documentation"),
                "section": doc.get("section", "General"),
                "source_filename": doc.get("source_filename", "unknown"),
                "source_url": doc.get("source_url") or None,
                "chunk_index": doc.get("chunk_index", 0)
            })

    full_context = "\n\n---\n\n".join(context_blocks)
    llm = get_llm()
    prompt = ANSWER_GENERATION_PROMPT.format(
        question=question,
        context=full_context
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        answer_text = response.content
        sufficient = True
    except (ValueError, RuntimeError):
        raise
    except Exception as e:
        logger.error(f"Answer generation failed: {e}", exc_info=True)
        answer_text = f"An error occurred while generating the answer: {str(e)}"
        sufficient = False

    return {
        "answer": answer_text,
        "citations": citations,
        "sufficient_context": sufficient
    }


def grounding_check_node(state: GraphState) -> Dict[str, Any]:
    """
    Node F: Checks whether the generated answer is supported by the retrieved context.
    """
    answer = state.get("answer", "")
    relevant_docs = state.get("relevant_documents", [])
    logger.info("[Node: Grounding Check] Verifying answer against retrieved context.")

    if not relevant_docs or not state.get("sufficient_context", True):
        return {
            "grounding_status": "insufficient_evidence"
        }

    context_snippets = "\n".join([d.get("content", "")[:300] for d in relevant_docs])
    llm = get_llm()
    prompt = GROUNDING_VERIFICATION_PROMPT.format(
        context=context_snippets,
        answer=answer
    )

    grounding_status = "grounded"
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        parsed = _extract_json(response.content)
        if parsed and not parsed.get("is_grounded", True):
            grounding_status = "ungrounded"
            logger.warning(f"Answer flagged as potentially ungrounded: {parsed.get('explanation')}")
    except Exception as e:
        logger.debug(f"Grounding check fallback: {e}")

    return {
        "grounding_status": grounding_status
    }


def insufficient_context_node(state: GraphState) -> Dict[str, Any]:
    """
    Terminal node when retrieval retries are exhausted without finding relevant documents.
    """
    logger.info("[Node: Insufficient Context] Max retries exhausted with no relevant documents.")
    return {
        "answer": (
            "The documentation corpus does not contain sufficient information to answer your question "
            "after multiple search and query reformulation attempts."
        ),
        "citations": [],
        "sufficient_context": False,
        "grounding_status": "insufficient_evidence"
    }


# ------------------------------------------------------------------------------
# Conditional Routing Edges
# ------------------------------------------------------------------------------

def decide_after_grading(state: GraphState) -> Literal["generate_answer", "query_retry", "insufficient_context"]:
    """
    Conditional edge evaluator after document grading:
    - If relevant documents exist -> proceed to generation.
    - If no relevant documents and retries remain -> rewrite query and retrieve again.
    - If no relevant documents and retries exhausted -> terminal insufficient-context response.
    """
    relevant_docs = state.get("relevant_documents", [])
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("maximum_retries", 2)

    if len(relevant_docs) > 0:
        logger.info(f"Conditional Router: {len(relevant_docs)} relevant docs found -> generate_answer")
        return "generate_answer"

    if retry_count < max_retries:
        logger.info(f"Conditional Router: 0 relevant docs, retry {retry_count} < {max_retries} -> query_retry")
        return "query_retry"

    logger.info(f"Conditional Router: 0 relevant docs, retries exhausted ({retry_count} >= {max_retries}) -> insufficient_context")
    return "insufficient_context"


# ------------------------------------------------------------------------------
# Graph Construction & Compilation
# ------------------------------------------------------------------------------

def build_rag_graph():
    """
    Constructs and compiles the Self-Corrective RAG LangGraph StateGraph.
    """
    workflow = StateGraph(GraphState)

    # Register Nodes
    workflow.add_node("query_analysis", query_analysis_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("grade_documents", document_grading_node)
    workflow.add_node("query_retry", query_retry_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("grounding_check", grounding_check_node)
    workflow.add_node("insufficient_context", insufficient_context_node)

    # Define Edges
    workflow.add_edge(START, "query_analysis")
    workflow.add_edge("query_analysis", "retrieval")
    workflow.add_edge("retrieval", "grade_documents")

    # Conditional Branching after Grading
    workflow.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {
            "generate_answer": "generate_answer",
            "query_retry": "query_retry",
            "insufficient_context": "insufficient_context"
        }
    )

    # Retry loop back to retrieval
    workflow.add_edge("query_retry", "retrieval")

    # Linear completion edges
    workflow.add_edge("generate_answer", "grounding_check")
    workflow.add_edge("grounding_check", END)
    workflow.add_edge("insufficient_context", END)

    # Compile the graph
    app = workflow.compile()
    return app


# Singleton compiled graph instance
rag_graph = build_rag_graph()

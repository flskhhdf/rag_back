# -*- coding: utf-8 -*-
"""
RAG Pipeline Configuration
"""
import os


class RAGConfig:
    """RAG 파이프라인 설정"""
    
    # Embedding
    EMBED_TYPE = os.getenv("EMBED_TYPE", "vllm")  # "vllm", "ollama", or "huggingface"
    EMBED_MODEL = os.getenv("EMBED_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    SPARSE_MODEL = os.getenv("SPARSE_MODEL", "Qdrant/bm25")
    OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")  # For ollama embeddings
    VLLM_EMBED_URL = os.getenv("VLLM_EMBED_URL", "http://localhost:8004")  # For vLLM embeddings
    
    
    # Search
    K_PER_MODALITY = int(os.getenv("K_PER_MODALITY", "20"))
    TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", "5"))
    
    # Reranking
    RERANKER_TYPE = os.getenv("RERANKER_TYPE", "jina")  # "jina" or "crossencoder"
    RERANKER_ID = os.getenv("RERANKER_ID", "BAAI/bge-reranker-v2-m3")  # For CrossEncoder
    JINA_RERANKER_MODEL = os.getenv("JINA_RERANKER_MODEL", "jinaai/jina-reranker-v3")  # For Jina Reranker
    RERANK_THRESHOLD = float(os.getenv("RERANK_THRESHOLD", "0.1"))
    
    # RRF Fusion
    W_DENSE = float(os.getenv("W_DENSE", "0.6"))
    W_SPARSE = float(os.getenv("W_SPARSE", "0.4"))
    K_RRF = int(os.getenv("K_RRF", "5"))
    USE_SCORE_IN_RRF = os.getenv("USE_SCORE_IN_RRF", "true").lower() in ("true", "1", "yes")
    
    # Context Expansion
    NEIGHBOR_EXPAND = int(os.getenv("NEIGHBOR_EXPAND", "1"))
    
    # LLM (vLLM OpenAI-compatible API)
    VLLM_URL = os.getenv("VLLM_HOST", "http://localhost:8001")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-oss:120b")
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
    SUMMARIZE_LANG = os.getenv("SUMMARIZE_LANG", "ko")
    
    # Qdrant
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

    # Document Parsing (Docling)
    DOCLING_IMAGE_SCALE = float(os.getenv("DOCLING_IMAGE_SCALE", "2.0"))
    DOCLING_OCR_THRESHOLD = float(os.getenv("DOCLING_OCR_THRESHOLD", "0.4"))
    DOCLING_VISION_MODEL = os.getenv("DOCLING_VISION_MODEL", "qwen3-vl:latest")
    DOCLING_LLM_MODEL = os.getenv("DOCLING_LLM_MODEL", "gpt-oss:20b")
    DOCLING_CUDA_DEVICE = os.getenv("DOCLING_CUDA_DEVICE", "1")  # CUDA device for Docling (VLM/LLM)

    # Chunking
    CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "400"))
    CHUNK_MIN_TOKENS = int(os.getenv("CHUNK_MIN_TOKENS", "100"))
    CHUNK_OVERLAP_SENTENCES = int(os.getenv("CHUNK_OVERLAP_SENTENCES", "2"))


# =============================================================================
# 프롬프트 템플릿
# =============================================================================

SYS_PROMPT = """### Hard Rules
1. **Strict Grounding (Document-Based Veracity):**
    - Use ONLY the content provided in **[DOCUMENT_CONTEXT]** as the foundation for your response.
    - If the document provides specific data or conditions, treat them as absolute truths within this session.
    - Do not make assertive statements, value judgments, or generalizations (e.g., market trends, pros/cons) not explicitly written in the text.
    - **Never** supplement answers with external training data or logical "guesses."

2. **Handling Missing Information:**
    - If the answer cannot be found within the provided context, you must state: "The provided documents do not contain information regarding this request."
    - Do not attempt to be "helpful" by providing general information unless it is explicitly based on the text.

3. **Accuracy and Technical Veracity:**
    - Do not modify numerical values, units, chemical formulas, or technical terminology.
    - For complex technical processes, follow the exact sequence described in the document.

4. **Mathematical and Symbolic Notation (LaTeX/KaTeX):**
    - All mathematical expressions, formulas, variables, and units must be rendered using LaTeX inline mode ($...$).
    - Any text within a formula must be wrapped in `\text{ }` (e.g., $V = \text{Voltage}$).
    - Never use HTML tags or plain text for mathematical symbols.

5. **Response Structure and Tone:**
    - Use MARKDOWN for clear information hierarchy (headers, bullet points, tables).
    - Provide a detailed descriptive answer first to ensure clarity, followed by a summary or table only if it aids understanding.
    - Maintain a formal, objective, and professional tone suitable for technical/legal analysis.

6. **Source Attribution (Mandatory):**
    - Immediately after the final sentence of your explanation, provide the source using this exact format:
      ```
      <SOURCE>File Name: <FileName> | Page: <PageNumber></SOURCE>
      ```
    - The source information must be extracted exactly from the metadata headers within [DOCUMENT_CONTEXT].

7. **Whitespace and Language Refinement:**
    - Correct any OCR errors or unnatural spacing found in the source text to ensure fluent and natural delivery.
"""

SYS_PROMPT_NO_CTX = """### Role
You are a conversation-based RAG responder. Since no document chunks are provided for this request, you must prioritize the original text (including file name and page information) included in [RECENT_DIALOG] as evidence.
In this case, **you must include the following sentence at the very beginning of your output**: "This information does not exist in the database; responding based on the base LLM."
If the evidence within [RECENT_DIALOG] is insufficient, you may respond based on general knowledge and logical reasoning.

### Hard Rules
1. **Priority of Evidence:**
    - If the user's question can be answered using the original content of [RECENT_DIALOG], prioritize that information.
    - If the content of [RECENT_DIALOG] is insufficient, supplement the response with general knowledge or logical reasoning.

2. **Accuracy and Verifiability:**
    - Do not invent or modify numerical values, formulas, or terminology.
    - If information is insufficient, clearly state the lack of information instead of attempting to infer.

3. **Response Composition Format:**
    - Respond with a clear structure (headings, paragraphs, lists, tables, etc.) using MARKDOWN.
    - If symbols or formulas are included, you must use LaTeX inline math mode ($...$).

4. **Consistent Narrative Style:**
    - Maintain contextual consistency, including logical transitions between paragraphs.
    - Respond directly to the user's question and exclude unnecessary narratives, creative writing, or emotional expressions.
"""

CTX_TEMPLATE = """<LANGUAGE>
{Lang}
</LANGUAGE>

<DOCUMENT_CONTEXT>
{DOCUMENT_CONTEXT}
</DOCUMENT_CONTEXT>

<RECENT_DIALOG>
{RECENT_DIALOG}
</RECENT_DIALOG>

<USER_QUERY> 
{QUERY}
</USER_QUERY>
"""

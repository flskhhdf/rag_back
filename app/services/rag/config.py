# -*- coding: utf-8 -*-
"""
RAG Pipeline Configuration
"""
import os


class RAGConfig:
    """RAG 파이프라인 설정"""
    
    # Embedding
    # Port 8004: Qwen/Qwen3-Embedding-8B (임베딩)
    EMBED_TYPE = os.getenv("EMBED_TYPE", "vllm")  # "vllm", "ollama", or "huggingface"
    EMBED_MODEL = os.getenv("EMBED_MODEL", "Qwen/Qwen3-Embedding-8B")
    SPARSE_MODEL = os.getenv("SPARSE_MODEL", "Qdrant/bm25")
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")  # For ollama embeddings
    VLLM_EMBED_URL = os.getenv("VLLM_EMBED_URL", "http://localhost:8004")  # For vLLM embeddings
    
    
    # Search
    K_PER_MODALITY = int(os.getenv("K_PER_MODALITY", "20"))
    TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", "5"))
    
    # Reranking
    RERANKER_TYPE = os.getenv("RERANKER_TYPE", "crossencoder")  # "qwen" or "crossencoder"
    RERANKER_ID = os.getenv("RERANKER_ID", "BAAI/bge-reranker-v2-m3")  # For CrossEncoder
    QWEN_RERANKER_MODEL = os.getenv("QWEN_RERANKER_MODEL", "Qwen/Qwen3-Reranker-8B")  # For Qwen Reranker
    RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "cuda:1")  # GPU device for Reranker
    RERANK_THRESHOLD = float(os.getenv("RERANK_THRESHOLD", "0.1"))  # 필터링 임계값 (낮은 점수 제거)

    # Minimum Search Score (검색 결과 신뢰도 임계값)
    # 최고 점수가 이 값보다 낮으면 검색 결과를 무시하고 대화 히스토리만 사용
    # Qwen Reranker 점수 범위: 0 ~ 1 (probability)
    # 권장: 0.15 (0.15 미만은 거의 무관한 질문 또는 후속 질문)
    MIN_SEARCH_SCORE = float(os.getenv("MIN_SEARCH_SCORE", "0.15"))
    
    # RRF Fusion
    W_DENSE = float(os.getenv("W_DENSE", "0.6"))
    W_SPARSE = float(os.getenv("W_SPARSE", "0.4"))
    K_RRF = int(os.getenv("K_RRF", "5"))
    USE_SCORE_IN_RRF = os.getenv("USE_SCORE_IN_RRF", "true").lower() in ("true", "1", "yes")
    
    # Context Expansion
    NEIGHBOR_EXPAND = int(os.getenv("NEIGHBOR_EXPAND", "1"))
    
    # LLM (vLLM OpenAI-compatible API)
    # Port 8001: openai/gpt-oss-120b (최종 사용자 응답)
    VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8001")
    LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
    SUMMARIZE_LANG = os.getenv("SUMMARIZE_LANG", "ko")

    # Follow-up Questions
    # Port 8003: openai/gpt-oss-20b (테이블 description + follow-up question)
    ENABLE_FOLLOW_UP_QUESTIONS = os.getenv("ENABLE_FOLLOW_UP_QUESTIONS", "true").lower() in ("true", "1", "yes")
    FOLLOW_UP_QUESTIONS_COUNT = int(os.getenv("FOLLOW_UP_QUESTIONS_COUNT", "3"))
    FOLLOW_UP_LLM_URL = os.getenv("FOLLOW_UP_LLM_URL", "http://localhost:8003")
    FOLLOW_UP_LLM_MODEL = os.getenv("FOLLOW_UP_LLM_MODEL", "openai/gpt-oss-20b")
    
    # Qdrant
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

    # Document Parsing (Docling)
    DOCLING_IMAGE_SCALE = float(os.getenv("DOCLING_IMAGE_SCALE", "2.0"))
    DOCLING_OCR_THRESHOLD = float(os.getenv("DOCLING_OCR_THRESHOLD", "0.4"))
    DOCLING_VISION_MODEL = os.getenv("DOCLING_VISION_MODEL", "Qwen/Qwen3-VL-8B-Instruct-FP8")
    DOCLING_LLM_MODEL = os.getenv("DOCLING_LLM_MODEL", "openai/gpt-oss-20b")
    # Note: DOCLING_CUDA_DEVICE 제거 (VLM/LLM은 vLLM API 사용, 로컬 GPU 불필요)

    # Port 8002: Qwen/Qwen3-VL-8B-Instruct-FP8 (이미지 description)
    VLM_URL = os.getenv("VLM_URL", "http://localhost:8002")

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
    - If the answer cannot be found within the provided context, you **MUST** respond with ONLY this exact phrase:
      **"해당 문서에 존재하지 않는 내용입니다."**
    - Do NOT attempt to be "helpful" by providing general information or external knowledge.
    - Do NOT make assumptions or inferences beyond what is explicitly stated in the document.

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

6. **Source Attribution (CRITICAL RULE):**
    - **ONLY IF** you provide an answer based on [DOCUMENT_CONTEXT], you **MUST** include the source immediately after your explanation:
      ```
      <SOURCE>File Name: <FileName> | Page: <PageNumber></SOURCE>
      ```
    - Extract source information exactly from the metadata headers within [DOCUMENT_CONTEXT].
    - **NEVER include SOURCE tags** when the document does not contain the requested information.
    - **NEVER fabricate or guess** file names or page numbers.

7. **Whitespace and Language Refinement:**
    - Correct any OCR errors or unnatural spacing found in the source text to ensure fluent and natural delivery.
"""

SYS_PROMPT_NO_CTX = """### Role
You are a conversation-based RAG responder. Since no document chunks are provided for this request, you must prioritize the original text (including file name and page information) included in [RECENT_DIALOG] as evidence.

### CRITICAL: No Database Context Alert
**You must include this sentence at the very beginning of your output**: 
"데이터베이스에 존재하지 않는 정보입니다. 기본 지식으로 응답합니다."

### Hard Rules
1. **Priority of Evidence:**
    - If the user's question can be answered using the original content of [RECENT_DIALOG], prioritize that information and include SOURCE tags from the original conversation.
    - If the content of [RECENT_DIALOG] is insufficient, supplement the response with general knowledge and logical reasoning.

2. **Handling Missing Information:**
    - If you cannot answer the question even with general knowledge, you **MUST** respond:
      **"해당 문서에 존재하지 않는 내용입니다."**
    - Do NOT make up information or fabricate answers.

3. **Accuracy and Verifiability:**
    - Do not invent or modify numerical values, formulas, or terminology.
    - If information is insufficient, clearly state the lack of information instead of attempting to infer.

4. **Response Composition Format:**
    - Respond with a clear structure (headings, paragraphs, lists, tables, etc.) using MARKDOWN.
    - If symbols or formulas are included, you must use LaTeX inline math mode ($...$).

5. **Source Attribution:**
    - **ONLY IF** you reference specific information from [RECENT_DIALOG], include the original SOURCE tags if they exist.
    - **NEVER fabricate** file names or page numbers.
    - If responding with general knowledge, do NOT include SOURCE tags.

6. **Consistent Narrative Style:**
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

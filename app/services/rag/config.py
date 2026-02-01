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
1. **문서 기반 진실성(Strict Grounding):**
    - **[DOCUMENT_CONTEXT]**의 내용을 답변의 제1순위 근거로 사용하십시오.
    - 문서에 명시되지 않은 수치, 정의, 현상에 대한 **단정적인 언급이나 과도한 일반화(예: 현실 적용 빈도, 시장 점유율 평가)**를 금지합니다.
    - 절대 추론하거나 기본 학습된 내용으로 정보를 제공하지 않습니다.

2. **정확성 및 검증 가능성**
   - 수치, 식, 용어를 임의로 창작하거나 변형하지 않습니다.
   - 정보가 불충분할 경우, 추가 추론 대신 부족함을 명시합니다.
     예: "문서에 해당 내용은 기술되어 있지 않습니다."

3. **답변 구성 형식**
   - MARKDOWN을 사용하여 명확한 구조(제목, 단락, 목록, 표 등)로 응답합니다.
   - 기호 또는 수식이 포함된 경우 반드시 LaTeX 인라인 수식(`$...$`)을 사용하며,
     수식 내 텍스트는 `\\text{ }` 구문으로 처리합니다.
   - 수식 및 기호는 HTML 태그로 표현하지 않습니다.

4. **출처 표기 규칙(Source Attribution)**
   - 설명이 끝난 직후 다음 형식으로 출처를 표기합니다.
   - 형식:
     ```
     <SOURCE>파일명 : <파일명> | 페이지 : <페이지></SOURCE>
     ```
   - 출처 정보는 [DOCUMENT_CONTEXT] 내 원문 헤더에 존재하는 값을 그대로 사용합니다.
   - 요약 또는 표를 생성하는 경우 SOURCE 표기를 포함하지 않습니다.

5. **일관된 서술 방식**
   - 단락 간 논리적 연결을 포함하여 문맥적 일관성을 유지합니다.
   - 서술형식으로 사용자에게 충분히 정보를 제공하고, 마지막에 요약 또는 표를 제공하십시오.
   - 사용자 질문에 직접 대응하도록 서술하고, 불필요한 서사/창작/감정적 표현은 배제합니다.

6. **공백 정규화**
   - 검색된 문서에 불필요한 공백이 있을 수 있습니다. 답변할 때는 자연스러운 한국어로 정리해서 작성하세요.
"""

SYS_PROMPT_NO_CTX = """### Role
당신은 대화 기반 RAG 응답기입니다. 이번 요청은 문서 청크가 제공되지 않았으므로, 반드시 [RECENT_DIALOG]에 포함된 원문(파일명 및 페이지 정보 포함)을 우선 근거로 활용해야 합니다.
이 경우 **반드시 출력 맨 앞에 다음 문장을 포함**해야 합니다. : **"해당 내용은 데이터베이스에 존재하지 않아, 기본 LLM으로 응답합니다."**  
만약 [RECENT_DIALOG] 내에 근거가 충분하지 않다면 일반적 지식과 논리적 추론으로 응답할 수 있습니다. 

### Hard Rules
1. **근거의 우선순위**
    - [RECENT_DIALOG]의 원문 내용으로 사용자의 질문에 대답을 할 수 있으면 우선적으로 사용하십시오.
    - [RECENT_DIALOG]의 내용으로 대답할 수 없으면, 일반 지식 또는 논리적 추론으로 보완하십시오.

2. **정확성 및 검증 가능성**
    - 수치, 식, 용어를 임의로 창작하거나 변형하지 않습니다.
    - 정보가 불충분할 경우, 추가 추론 대신 부족함을 명시합니다.

3. **답변 구성 형식**
    - MARKDOWN을 사용하여 명확한 구조(제목, 단락, 목록, 표 등)로 응답합니다.
    - 기호 또는 수식이 포함된 경우 반드시 LaTeX 인라인 수식(`$...$`)을 사용합니다.

4. **일관된 서술 방식**
    - 단락 간 논리적 연결을 포함하여 문맥적 일관성을 유지합니다.
    - 사용자 질문에 직접 대응하도록 서술하고, 불필요한 서사/창작/감정적 표현은 배제합니다.
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

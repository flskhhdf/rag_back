"""
MySQL 서비스 모듈 - 파일 메타데이터 관리
"""
import os
import hashlib
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

import pymysql
from pymysql.cursors import DictCursor

logger = logging.getLogger(__name__)


# ===== Config =====
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "llm")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "asdf!@#$")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "rag_service")


def get_connection():
    """MySQL 연결 생성"""
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        charset="utf8mb4",
        cursorclass=DictCursor,
        autocommit=True,
    )


@contextmanager
def get_cursor():
    """컨텍스트 매니저로 커서 관리"""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        yield cursor
    finally:
        cursor.close()
        conn.close()


def calculate_file_hash(content: bytes) -> str:
    """파일 해시 계산 (SHA-256)"""
    return hashlib.sha256(content).hexdigest()


# ===== File Info CRUD =====

def create_file_info(
    file_id: str,
    file_name: str,
    file_path: str,
    file_hash: str,
    uploaded_by: Optional[str] = None,
    extend: Optional[str] = None,
) -> bool:
    """파일 정보 생성"""
    try:
        with get_cursor() as cursor:
            sql = """
                INSERT INTO file_info (id, file_name, extend, file_path, file_hash, uploaded_by)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (file_id, file_name, extend, file_path, file_hash, uploaded_by))
            return True
    except Exception as e:
        print(f"[ERROR] create_file_info: {e}")
        return False


def get_file_by_id(file_id: str) -> Optional[Dict[str, Any]]:
    """ID로 파일 정보 조회"""
    try:
        with get_cursor() as cursor:
            sql = "SELECT * FROM file_info WHERE id = %s"
            cursor.execute(sql, (file_id,))
            return cursor.fetchone()
    except Exception as e:
        print(f"[ERROR] get_file_by_id: {e}")
        return None


def get_file_by_hash(file_hash: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """해시로 파일 조회 (중복 체크용)

    Args:
        file_hash: 파일 해시
        user_id: 사용자 ID (선택, 제공되면 해당 사용자의 파일만 검색)
    """
    try:
        with get_cursor() as cursor:
            if user_id:
                sql = "SELECT * FROM file_info WHERE file_hash = %s AND uploaded_by = %s LIMIT 1"
                cursor.execute(sql, (file_hash, user_id))
            else:
                sql = "SELECT * FROM file_info WHERE file_hash = %s LIMIT 1"
                cursor.execute(sql, (file_hash,))
            return cursor.fetchone()
    except Exception as e:
        print(f"[ERROR] get_file_by_hash: {e}")
        return None


def delete_file_info(file_id: str) -> bool:
    """파일 정보 삭제 (notebook_file_link와 file_info에서 모두 삭제)"""
    try:
        with get_cursor() as cursor:
            # notebook_file_link에서 먼저 삭제 (FK 제약)
            cursor.execute("DELETE FROM notebook_file_link WHERE file_id = %s", (file_id,))
            link_deleted = cursor.rowcount
            logger.info(f"[DB] Deleted {link_deleted} notebook_file_link records for file_id={file_id}")

            # file_info에서 삭제
            cursor.execute("DELETE FROM file_info WHERE id = %s", (file_id,))
            file_deleted = cursor.rowcount
            logger.info(f"[DB] Deleted {file_deleted} file_info records for file_id={file_id}")

            if file_deleted == 0:
                logger.warning(f"[DB] No file_info record found with id={file_id}")
                return False

            return True
    except Exception as e:
        logger.error(f"[DB] Failed to delete_file_info: {e}")
        import traceback
        logger.error(f"[DB] Traceback: {traceback.format_exc()}")
        return False


def get_file_list(
    uploaded_by: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """파일 목록 조회"""
    try:
        with get_cursor() as cursor:
            if uploaded_by:
                sql = """
                    SELECT id, file_name, extend, file_path, created_at
                    FROM file_info
                    WHERE uploaded_by = %s
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                """
                cursor.execute(sql, (uploaded_by, limit, offset))
            else:
                sql = """
                    SELECT id, file_name, extend, file_path, created_at
                    FROM file_info
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                """
                cursor.execute(sql, (limit, offset))
            return cursor.fetchall()
    except Exception as e:
        print(f"[ERROR] get_file_list: {e}")
        return []


# ===== Notebook-File Link =====

def link_file_to_notebook(notebook_id: str, file_id: str) -> bool:
    """파일을 노트북에 연결"""
    try:
        with get_cursor() as cursor:
            sql = """
                INSERT INTO notebook_file_link (notebook_id, file_id)
                VALUES (%s, %s)
                ON DUPLICATE KEY UPDATE linked_at = CURRENT_TIMESTAMP
            """
            cursor.execute(sql, (notebook_id, file_id))
            return True
    except Exception as e:
        print(f"[ERROR] link_file_to_notebook: {e}")
        return False


def get_files_by_notebook(notebook_id: str) -> List[Dict[str, Any]]:
    """노트북에 연결된 파일 목록 조회"""
    try:
        with get_cursor() as cursor:
            sql = """
                SELECT f.id, f.file_name, f.extend, f.file_path, f.created_at
                FROM file_info f
                JOIN notebook_file_link nfl ON f.id = nfl.file_id
                WHERE nfl.notebook_id = %s
                ORDER BY nfl.linked_at DESC
            """
            cursor.execute(sql, (notebook_id,))
            return cursor.fetchall()
    except Exception as e:
        print(f"[ERROR] get_files_by_notebook: {e}")
        return []


def unlink_file_from_notebook(notebook_id: str, file_id: str) -> bool:
    """노트북에서 파일 연결 해제"""
    try:
        with get_cursor() as cursor:
            sql = "DELETE FROM notebook_file_link WHERE notebook_id = %s AND file_id = %s"
            cursor.execute(sql, (notebook_id, file_id))
            return True
    except Exception as e:
        print(f"[ERROR] unlink_file_from_notebook: {e}")
        return False


# ===== Notebook CRUD =====

def create_notebook(notebook_id: str, title: str, created_by: str) -> bool:
    """노트북 생성 (유저가 없으면 자동 생성)"""
    try:
        with get_cursor() as cursor:
            # 1. 유저 존재 여부 확인
            cursor.execute("SELECT id FROM users WHERE id = %s", (created_by,))
            if not cursor.fetchone():
                # 유저가 없으면 자동 생성
                print(f"[INFO] User {created_by} not found. Creating user automatically.")
                # 이메일은 필수 필드가 아니거나 더미 값으로 처리 (스키마에 따라 다름)
                # 여기서는 user_id를 기반으로 더미 이메일 생성
                dummy_email = f"{created_by}@tenergy-x.com"
                cursor.execute(
                    "INSERT INTO users (id, username, email) VALUES (%s, %s, %s)",
                    (created_by, "Auto Created User", dummy_email)
                )

            # 2. 노트북 생성
            sql = """
                INSERT INTO notebook (id, title, created_by)
                VALUES (%s, %s, %s)
            """
            cursor.execute(sql, (notebook_id, title, created_by))
            return True
    except Exception as e:
        print(f"[ERROR] create_notebook: {e}")
        return False


def get_notebook_by_id(notebook_id: str) -> Optional[Dict[str, Any]]:
    """ID로 노트북 조회"""
    try:
        with get_cursor() as cursor:
            sql = "SELECT * FROM notebook WHERE id = %s"
            cursor.execute(sql, (notebook_id,))
            return cursor.fetchone()
    except Exception as e:
        print(f"[ERROR] get_notebook_by_id: {e}")
        return None


def get_notebooks_by_user(
    user_id: str,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """사용자의 노트북 목록 조회"""
    try:
        with get_cursor() as cursor:
            sql = """
                SELECT id, title, created_by, created_at
                FROM notebook
                WHERE created_by = %s
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
            """
            cursor.execute(sql, (user_id, limit, offset))
            return cursor.fetchall()
    except Exception as e:
        print(f"[ERROR] get_notebooks_by_user: {e}")
        return []


def get_all_notebooks(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """모든 노트북 목록 조회"""
    try:
        with get_cursor() as cursor:
            sql = """
                SELECT id, title, created_by, created_at
                FROM notebook
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
            """
            cursor.execute(sql, (limit, offset))
            return cursor.fetchall()
    except Exception as e:
        print(f"[ERROR] get_all_notebooks: {e}")
        return []


def update_notebook(notebook_id: str, title: str) -> bool:
    """노트북 제목 수정"""
    try:
        with get_cursor() as cursor:
            sql = "UPDATE notebook SET title = %s WHERE id = %s"
            cursor.execute(sql, (title, notebook_id))
            return cursor.rowcount > 0
    except Exception as e:
        print(f"[ERROR] update_notebook: {e}")
        return False


def delete_notebook(notebook_id: str) -> bool:
    """노트북 삭제"""
    try:
        with get_cursor() as cursor:
            # notebook_file_link에서 먼저 삭제
            cursor.execute("DELETE FROM notebook_file_link WHERE notebook_id = %s", (notebook_id,))
            # notebook에서 삭제
            cursor.execute("DELETE FROM notebook WHERE id = %s", (notebook_id,))
            return True
    except Exception as e:
        print(f"[ERROR] delete_notebook: {e}")
        return False


# ===== User CRUD =====

def create_user(user_id: str, username: str, email: str) -> bool:
    """사용자 생성"""
    try:
        with get_cursor() as cursor:
            sql = """
                INSERT INTO users (id, username, email)
                VALUES (%s, %s, %s)
            """
            cursor.execute(sql, (user_id, username, email))
            return True
    except Exception as e:
        print(f"[ERROR] create_user: {e}")
        return False


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """ID로 사용자 조회"""
    try:
        with get_cursor() as cursor:
            sql = "SELECT * FROM users WHERE id = %s"
            cursor.execute(sql, (user_id,))
            return cursor.fetchone()
    except Exception as e:
        print(f"[ERROR] get_user_by_id: {e}")
        return None


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """이메일로 사용자 조회"""
    try:
        with get_cursor() as cursor:
            sql = "SELECT * FROM users WHERE email = %s"
            cursor.execute(sql, (email,))
            return cursor.fetchone()
    except Exception as e:
        print(f"[ERROR] get_user_by_email: {e}")
        return None


def update_user(user_id: str, username: Optional[str] = None, email: Optional[str] = None) -> bool:
    """사용자 정보 수정"""
    try:
        with get_cursor() as cursor:
            updates = []
            params = []
            if username:
                updates.append("username = %s")
                params.append(username)
            if email:
                updates.append("email = %s")
                params.append(email)
            
            if not updates:
                return False
            
            params.append(user_id)
            sql = f"UPDATE users SET {', '.join(updates)} WHERE id = %s"
            cursor.execute(sql, tuple(params))
            return cursor.rowcount > 0
    except Exception as e:
        print(f"[ERROR] update_user: {e}")
        return False


def delete_user(user_id: str) -> bool:
    """사용자 삭제"""
    try:
        with get_cursor() as cursor:
            # 사용자가 만든 노트북들의 연결 먼저 삭제
            cursor.execute("""
                DELETE FROM notebook_file_link 
                WHERE notebook_id IN (SELECT id FROM notebook WHERE created_by = %s)
            """, (user_id,))
            # 사용자가 만든 노트북 삭제
            cursor.execute("DELETE FROM notebook WHERE created_by = %s", (user_id,))
            # 사용자가 업로드한 파일의 연결 삭제
            cursor.execute("""
                DELETE FROM notebook_file_link 
                WHERE file_id IN (SELECT id FROM file_info WHERE uploaded_by = %s)
            """, (user_id,))
            # 사용자가 업로드한 파일 삭제
            cursor.execute("DELETE FROM file_info WHERE uploaded_by = %s", (user_id,))
            # 사용자 삭제
            cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
            return True
    except Exception as e:
        print(f"[ERROR] delete_user: {e}")
        return False


# ===== Chat History CRUD =====

def create_chat_message(
    message_id: str,
    notebook_id: str,
    role: str,
    content: str,
    metadata: Optional[str] = None,
) -> bool:
    """채팅 메시지 저장
    
    Args:
        message_id: 메시지 ID (UUID)
        notebook_id: 노트북 ID
        role: 'user' 또는 'assistant'
        content: 메시지 내용
        metadata: JSON 문자열 (선택사항) - pdf_id, filename, sources 등
    """
    try:
        with get_cursor() as cursor:
            sql = """
                INSERT INTO chat_history (id, notebook_id, role, content, metadata)
                VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (message_id, notebook_id, role, content, metadata))
            logger.info(f"[DB] Saved chat message: {message_id} ({role}) to notebook {notebook_id}")
            return True
    except Exception as e:
        logger.error(f"[DB] Failed to save chat message: {e}")
        logger.error(f"[DB] Message details - notebook_id: {notebook_id}, role: {role}, content_len: {len(content)}")
        import traceback
        logger.error(f"[DB] Traceback: {traceback.format_exc()}")
        return False


def get_chat_history(
    notebook_id: str,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """노트북의 채팅 기록 조회 (시간순 정렬)
    
    Args:
        notebook_id: 노트북 ID
        limit: 최대 조회 개수
        offset: 오프셋
        
    Returns:
        채팅 메시지 리스트 (오래된 순서 -> 최신순)
    """
    try:
        with get_cursor() as cursor:
            sql = """
                SELECT id, notebook_id, role, content, metadata, created_at
                FROM chat_history
                WHERE notebook_id = %s
                ORDER BY created_at ASC
                LIMIT %s OFFSET %s
            """
            cursor.execute(sql, (notebook_id, limit, offset))
            results = cursor.fetchall()
            logger.info(f"[DB] Retrieved {len(results)} chat messages for notebook {notebook_id}")
            return results
    except Exception as e:
        logger.error(f"[DB] Failed to get chat history: {e}")
        logger.error(f"[DB] notebook_id: {notebook_id}, limit: {limit}, offset: {offset}")
        import traceback
        logger.error(f"[DB] Traceback: {traceback.format_exc()}")
        return []


def get_recent_chat_history(
    notebook_id: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """노트북의 최근 채팅 기록 조회
    
    Args:
        notebook_id: 노트북 ID
        limit: 최대 조회 개수 (기본 10개)
        
    Returns:
        최근 채팅 메시지 리스트 (오래된 순서로 반환)
    """
    try:
        with get_cursor() as cursor:
            # 최근 N개를 먼저 DESC로 가져온 후, 다시 ASC로 정렬
            sql = """
                SELECT * FROM (
                    SELECT id, notebook_id, role, content, metadata, created_at
                    FROM chat_history
                    WHERE notebook_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                ) AS recent_messages
                ORDER BY created_at ASC
            """
            cursor.execute(sql, (notebook_id, limit))
            return cursor.fetchall()
    except Exception as e:
        print(f"[ERROR] get_recent_chat_history: {e}")
        return []


def delete_chat_history(notebook_id: str) -> bool:
    """노트북의 모든 채팅 기록 삭제
    
    Args:
        notebook_id: 노트북 ID
    """
    try:
        with get_cursor() as cursor:
            sql = "DELETE FROM chat_history WHERE notebook_id = %s"
            cursor.execute(sql, (notebook_id,))
            return True
    except Exception as e:
        print(f"[ERROR] delete_chat_history: {e}")
        return False


def delete_chat_message(message_id: str) -> bool:
    """특정 채팅 메시지 삭제

    Args:
        message_id: 메시지 ID
    """
    try:
        with get_cursor() as cursor:
            sql = "DELETE FROM chat_history WHERE id = %s"
            cursor.execute(sql, (message_id,))
            return cursor.rowcount > 0
    except Exception as e:
        print(f"[ERROR] delete_chat_message: {e}")
        return False


# ===== Chat Feedback CRUD =====

def create_feedback(
    feedback_id: str,
    message_id: str,
    notebook_id: str,
    is_positive: bool,
    comment: Optional[str] = None,
    question_content: str = "",
    answer_content: str = "",
    sources: Optional[str] = None,
) -> bool:
    """피드백 생성 또는 업데이트 (UPSERT)

    Args:
        feedback_id: 피드백 ID (UUID)
        message_id: 메시지 ID (chat_history FK)
        notebook_id: 노트북 ID
        is_positive: 긍정적 피드백 여부 (True=👍, False=👎)
        comment: 추가 코멘트 (선택)
        question_content: 질문 내용 (저장용)
        answer_content: 답변 내용 (저장용)
        sources: RAG 소스 JSON 문자열 (선택)
    """
    try:
        with get_cursor() as cursor:
            sql = """
                INSERT INTO chat_feedback
                (id, message_id, notebook_id, is_positive, comment,
                 question_content, answer_content, sources)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    is_positive = VALUES(is_positive),
                    comment = VALUES(comment),
                    updated_at = CURRENT_TIMESTAMP
            """
            cursor.execute(sql, (
                feedback_id, message_id, notebook_id, is_positive,
                comment, question_content, answer_content, sources
            ))
            logger.info(f"[DB] Saved feedback: {feedback_id} for message {message_id}")
            return True
    except Exception as e:
        logger.error(f"[DB] Failed to save feedback: {e}")
        import traceback
        logger.error(f"[DB] Traceback: {traceback.format_exc()}")
        return False


def get_feedback_by_message_id(message_id: str) -> Optional[Dict[str, Any]]:
    """메시지 ID로 피드백 조회

    Args:
        message_id: 메시지 ID

    Returns:
        피드백 정보 또는 None
    """
    try:
        with get_cursor() as cursor:
            sql = "SELECT * FROM chat_feedback WHERE message_id = %s"
            cursor.execute(sql, (message_id,))
            return cursor.fetchone()
    except Exception as e:
        logger.error(f"[DB] Failed to get feedback by message_id: {e}")
        return None


def get_feedback_by_id(feedback_id: str) -> Optional[Dict[str, Any]]:
    """피드백 ID로 조회

    Args:
        feedback_id: 피드백 ID

    Returns:
        피드백 정보 또는 None
    """
    try:
        with get_cursor() as cursor:
            sql = "SELECT * FROM chat_feedback WHERE id = %s"
            cursor.execute(sql, (feedback_id,))
            return cursor.fetchone()
    except Exception as e:
        logger.error(f"[DB] Failed to get feedback by id: {e}")
        return None


def update_feedback(
    feedback_id: str,
    is_positive: Optional[bool] = None,
    comment: Optional[str] = None,
) -> bool:
    """피드백 수정 (thumbs up/down 변경 또는 코멘트 수정)

    Args:
        feedback_id: 피드백 ID
        is_positive: 긍정적 피드백 여부 (선택)
        comment: 코멘트 (선택)

    Returns:
        성공 여부
    """
    try:
        with get_cursor() as cursor:
            updates = []
            params = []

            if is_positive is not None:
                updates.append("is_positive = %s")
                params.append(is_positive)

            if comment is not None:
                updates.append("comment = %s")
                params.append(comment)

            if not updates:
                return False

            updates.append("updated_at = CURRENT_TIMESTAMP")
            params.append(feedback_id)

            sql = f"UPDATE chat_feedback SET {', '.join(updates)} WHERE id = %s"
            cursor.execute(sql, tuple(params))
            logger.info(f"[DB] Updated feedback: {feedback_id}")
            return cursor.rowcount > 0
    except Exception as e:
        logger.error(f"[DB] Failed to update feedback: {e}")
        return False


def get_feedbacks_by_notebook(
    notebook_id: str,
    is_positive: Optional[bool] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """노트북의 피드백 목록 조회 (분석용)

    Args:
        notebook_id: 노트북 ID
        is_positive: 필터링 (None=전체, True=긍정만, False=부정만)
        limit: 최대 조회 개수
        offset: 오프셋

    Returns:
        피드백 목록
    """
    try:
        with get_cursor() as cursor:
            if is_positive is not None:
                sql = """
                    SELECT * FROM chat_feedback
                    WHERE notebook_id = %s AND is_positive = %s
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                """
                cursor.execute(sql, (notebook_id, is_positive, limit, offset))
            else:
                sql = """
                    SELECT * FROM chat_feedback
                    WHERE notebook_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                """
                cursor.execute(sql, (notebook_id, limit, offset))
            return cursor.fetchall()
    except Exception as e:
        logger.error(f"[DB] Failed to get feedbacks by notebook: {e}")
        return []


def delete_feedback(feedback_id: str) -> bool:
    """피드백 삭제

    Args:
        feedback_id: 피드백 ID

    Returns:
        성공 여부
    """
    try:
        with get_cursor() as cursor:
            sql = "DELETE FROM chat_feedback WHERE id = %s"
            cursor.execute(sql, (feedback_id,))
            logger.info(f"[DB] Deleted feedback: {feedback_id}")
            return cursor.rowcount > 0
    except Exception as e:
        logger.error(f"[DB] Failed to delete feedback: {e}")
        return False


def get_qa_pair_by_message_id(message_id: str) -> Optional[Dict[str, Any]]:
    """Assistant 메시지 ID로 Q&A 페어 조회

    Args:
        message_id: Assistant 메시지 ID

    Returns:
        {
            "question": {"id": "...", "content": "...", ...},
            "answer": {"id": "...", "content": "...", "metadata": {...}},
            "notebook_id": "..."
        }
        또는 None
    """
    try:
        with get_cursor() as cursor:
            # 1. Assistant 메시지 조회
            cursor.execute(
                "SELECT * FROM chat_history WHERE id = %s AND role = 'assistant'",
                (message_id,)
            )
            answer = cursor.fetchone()

            if not answer:
                logger.warning(f"[DB] Message {message_id} not found or not an assistant message")
                return None

            # 2. 직전 User 메시지 조회
            cursor.execute("""
                SELECT * FROM chat_history
                WHERE notebook_id = %s
                  AND role = 'user'
                  AND created_at < %s
                ORDER BY created_at DESC
                LIMIT 1
            """, (answer["notebook_id"], answer["created_at"]))

            question = cursor.fetchone()

            if not question:
                logger.warning(f"[DB] No question found for answer {message_id}")
                return None

            return {
                "question": question,
                "answer": answer,
                "notebook_id": answer["notebook_id"],
            }
    except Exception as e:
        logger.error(f"[DB] Failed to get Q&A pair: {e}")
        import traceback
        logger.error(f"[DB] Traceback: {traceback.format_exc()}")
        return None


def get_chat_history_with_feedback(
    notebook_id: str,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """채팅 기록과 피드백을 함께 조회

    Args:
        notebook_id: 노트북 ID
        limit: 최대 조회 개수
        offset: 오프셋

    Returns:
        채팅 메시지 리스트 (피드백 정보 포함)
    """
    try:
        with get_cursor() as cursor:
            sql = """
                SELECT
                    ch.id, ch.notebook_id, ch.role, ch.content,
                    ch.metadata, ch.created_at,
                    cf.id as feedback_id, cf.is_positive, cf.comment,
                    cf.created_at as feedback_created_at, cf.updated_at as feedback_updated_at
                FROM chat_history ch
                LEFT JOIN chat_feedback cf ON ch.id = cf.message_id
                WHERE ch.notebook_id = %s
                ORDER BY ch.created_at ASC
                LIMIT %s OFFSET %s
            """
            cursor.execute(sql, (notebook_id, limit, offset))
            return cursor.fetchall()
    except Exception as e:
        logger.error(f"[DB] Failed to get chat history with feedback: {e}")
        import traceback
        logger.error(f"[DB] Traceback: {traceback.format_exc()}")
        return []


# ===== Config Prompt Management =====

def get_config_prompt(config_type: str) -> Optional[str]:
    """Config 테이블에서 프롬프트 조회
    
    Args:
        config_type: 프롬프트 타입 (예: 'follow_up', 'example_q' 등)
    
    Returns:
        프롬프트 문자열 또는 None
    """
    try:
        with get_cursor() as cursor:
            sql = "SELECT prompt FROM config WHERE type = %s"
            cursor.execute(sql, (config_type,))
            result = cursor.fetchone()
            return result['prompt'] if result else None
    except Exception as e:
        logger.error(f"[DB] Failed to get config prompt: {e}")
        return None


def set_config_prompt(config_type: str, prompt: str) -> bool:
    """Config 프롬프트 설정 (INSERT or UPDATE)
    
    Args:
        config_type: 프롬프트 타입
        prompt: 프롬프트 내용
    
    Returns:
        성공 여부
    """
    try:
        with get_cursor() as cursor:
            # 기존 레코드 확인
            cursor.execute("SELECT id FROM config WHERE type = %s", (config_type,))
            existing = cursor.fetchone()
            
            if existing:
                # UPDATE
                sql = "UPDATE config SET prompt = %s WHERE type = %s"
                cursor.execute(sql, (prompt, config_type))
            else:
                # INSERT
                sql = "INSERT INTO config (type, prompt) VALUES (%s, %s)"
                cursor.execute(sql, (config_type, prompt))
            
            return True
    except Exception as e:
        logger.error(f"[DB] Failed to set config prompt: {e}")
        return False

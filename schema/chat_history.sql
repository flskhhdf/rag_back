-- 채팅 기록 테이블
CREATE TABLE IF NOT EXISTS chat_history (
    id VARCHAR(36) PRIMARY KEY,
    notebook_id VARCHAR(36) NOT NULL,
    role ENUM('user', 'assistant') NOT NULL,
    content TEXT NOT NULL,
    metadata JSON DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_notebook_created (notebook_id, created_at DESC),
    FOREIGN KEY (notebook_id) REFERENCES notebook(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- metadata 예시:
-- {
--   "pdf_id": "파일ID",
--   "filename": "파일명",
--   "sources": [...],  // assistant 메시지인 경우 참조한 소스
-- }

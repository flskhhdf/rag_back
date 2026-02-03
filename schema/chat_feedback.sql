CREATE TABLE IF NOT EXISTS chat_feedback (
    id VARCHAR(36) PRIMARY KEY,
    message_id VARCHAR(36) NOT NULL,
    notebook_id VARCHAR(36) NOT NULL,
    is_positive BOOLEAN NOT NULL,
    comment TEXT DEFAULT NULL,
    question_content TEXT NOT NULL,
    answer_content TEXT NOT NULL,
    sources JSON DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_message_id (message_id),
    INDEX idx_notebook_id (notebook_id),
    INDEX idx_created_at (created_at DESC),
    INDEX idx_is_positive (is_positive),

    FOREIGN KEY (message_id) REFERENCES chat_history(id) ON DELETE CASCADE,
    FOREIGN KEY (notebook_id) REFERENCES notebook(id) ON DELETE CASCADE,

    UNIQUE KEY unique_message_feedback (message_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

"""
Database services

This package contains services for interacting with various databases.
"""

from .mysql import (
    get_connection,
    get_cursor,
    calculate_file_hash,
    create_file_info,
    get_file_by_id,
    get_file_by_hash,
    delete_file_info,
    get_file_list,
    link_file_to_notebook,
    get_files_by_notebook,
    unlink_file_from_notebook,
    create_notebook,
    get_notebook_by_id,
    get_notebooks_by_user,
    get_all_notebooks,
    update_notebook,
    delete_notebook,
    create_user,
    get_user_by_id,
    get_user_by_email,
    update_user,
    delete_user,
)

from .qdrant import (
    get_client,
    get_embeddings,
    sanitize_collection_name,
    ensure_collection,
    get_vectorstore,
    upsert_chunks,
    search_similar,
    get_pdf_list,
    delete_pdf,
)

__all__ = [
    # MySQL
    "get_connection",
    "get_cursor",
    "calculate_file_hash",
    "create_file_info",
    "get_file_by_id",
    "get_file_by_hash",
    "delete_file_info",
    "get_file_list",
    "link_file_to_notebook",
    "get_files_by_notebook",
    "unlink_file_from_notebook",
    "create_notebook",
    "get_notebook_by_id",
    "get_notebooks_by_user",
    "get_all_notebooks",
    "update_notebook",
    "delete_notebook",
    "create_user",
    "get_user_by_id",
    "get_user_by_email",
    "update_user",
    "delete_user",
    # Qdrant
    "get_client",
    "get_embeddings",
    "sanitize_collection_name",
    "ensure_collection",
    "get_vectorstore",
    "upsert_chunks",
    "search_similar",
    "get_pdf_list",
    "delete_pdf",
]

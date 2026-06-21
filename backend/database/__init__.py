"""Database models and CRUD operations."""

from backend.database.models import Base, Discovery, CognitionLog, StatusEnum
from backend.database.discovery_db import (
    create_discovery,
    get_discovery,
    get_discovery_by_candidate_id,
    list_discoveries,
    count_discoveries,
    update_discovery,
    delete_discovery,
)

__all__ = [
    "Base", "Discovery", "CognitionLog", "StatusEnum",
    "create_discovery", "get_discovery", "get_discovery_by_candidate_id",
    "list_discoveries", "count_discoveries", "update_discovery",
    "delete_discovery",
]

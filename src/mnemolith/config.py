import os
from pathlib import Path
from urllib.parse import quote_plus, unquote, urlparse

from dotenv import load_dotenv

load_dotenv()


def get_vault_path() -> str:
    """Return the Obsidian vault path from OBSIDIAN_VAULT_PATH env var."""
    vault_path = os.environ.get("OBSIDIAN_VAULT_PATH")
    if not vault_path:
        raise OSError(
            "OBSIDIAN_VAULT_PATH environment variable is not set. "
            "Set it to the absolute path of your Obsidian vault."
        )
    return vault_path


def get_qdrant_url() -> str:
    url = os.environ.get("QDRANT_URL")
    if not url:
        raise OSError("QDRANT_URL environment variable is not set.")
    return url


def get_collection_name() -> str:
    name = os.environ.get("COLLECTION_NAME")
    if not name:
        raise OSError("COLLECTION_NAME environment variable is not set.")
    return name


def get_embedding_provider() -> str:
    provider = os.environ.get("EMBEDDING_PROVIDER")
    if not provider:
        raise OSError("EMBEDDING_PROVIDER environment variable is not set.")
    return provider


def get_postgres_dsn() -> str:
    dsn = os.environ.get("POSTGRES_DSN")
    if dsn:
        return dsn
    user = os.environ.get("POSTGRES_USER")
    password = os.environ.get("POSTGRES_PASSWORD")
    db = os.environ.get("POSTGRES_DB")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    if not all([user, password, db]):
        raise OSError(
            "Set either POSTGRES_DSN or POSTGRES_USER + POSTGRES_PASSWORD + POSTGRES_DB."
        )
    return f"postgresql://{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{db}"


def get_qdrant_api_key() -> str | None:
    """Return the Qdrant API key, or None if not set."""
    return os.environ.get("QDRANT_API_KEY")


def get_vector_backend() -> str:
    return os.environ.get("VECTOR_BACKEND", "qdrant")


def is_sparse_search_enabled() -> bool:
    val = os.environ.get("SPARSE_SEARCH_ENABLED", "true").lower()
    return val not in ("false", "0")


def get_backup_dir() -> Path:
    return Path(os.environ.get("BACKUP_DIR", "~/.mnemolith/backups")).expanduser()


def get_neo4j_uri() -> str:
    return os.environ.get("NEO4J_URI", "bolt://localhost:7687")


def get_neo4j_user() -> str:
    return os.environ.get("NEO4J_USER", "neo4j")


def get_neo4j_password() -> str:
    pw = os.environ.get("NEO4J_PASSWORD")
    if not pw:
        raise OSError(
            "NEO4J_PASSWORD is not set. Required for the Graphiti temporal graph. "
            "Match the value used by the docker-compose neo4j service."
        )
    return pw


def get_neo4j_database() -> str:
    # Aura Free names the database after the instance ID, not "neo4j" —
    # so override via NEO4J_DATABASE when targeting Aura.
    return os.environ.get("NEO4J_DATABASE", "neo4j")


def get_graphiti_llm_provider() -> str:
    return os.environ.get("GRAPHITI_LLM_PROVIDER", "anthropic")


def get_graphiti_llm_model() -> str | None:
    return os.environ.get("GRAPHITI_LLM_MODEL")


def get_graphiti_embedding_model() -> str:
    return os.environ.get("GRAPHITI_EMBEDDING_MODEL", "text-embedding-3-small")


def get_postgres_conn_params() -> dict[str, str]:
    """Return dict with host, port, user, password, dbname for pg_dump/psql."""
    user = os.environ.get("POSTGRES_USER")
    password = os.environ.get("POSTGRES_PASSWORD")
    db = os.environ.get("POSTGRES_DB")
    if all([user, password, db]):
        host = os.environ.get("POSTGRES_HOST", "localhost")
        port = os.environ.get("POSTGRES_PORT", "5432")
        return {"host": host, "port": port, "user": user, "password": password, "dbname": db}
    dsn = os.environ.get("POSTGRES_DSN")
    if dsn:
        parsed = urlparse(dsn)
        return {
            "host": parsed.hostname or "localhost",
            "port": str(parsed.port or 5432),
            "user": unquote(parsed.username or ""),
            "password": unquote(parsed.password or ""),
            "dbname": parsed.path.lstrip("/"),
        }
    raise OSError(
        "Set either POSTGRES_DSN or POSTGRES_USER + POSTGRES_PASSWORD + POSTGRES_DB."
    )

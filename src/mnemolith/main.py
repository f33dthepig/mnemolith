import argparse
import asyncio
import sys
from datetime import UTC
from pathlib import Path

from mnemolith.backup import create_backup, restore_backup
from mnemolith.config import get_collection_name, get_vault_path
from mnemolith.embeddings import build_embedder, build_sparse_embedder
from mnemolith.indexer import index_vault, search
from mnemolith.vector_store import CollectionNotFoundError, get_vector_store

MAX_LIMIT = 50


def cmd_index(args):
    vault_path = args.vault_path or get_vault_path()
    if not Path(vault_path).is_dir():
        print(f"Error: '{vault_path}' is not a valid directory.")
        sys.exit(1)
    full = args.full
    if args.clean:
        print("warning: --clean is deprecated, use --full instead.")
        full = True
    embedder = build_embedder()
    sparse_embedder = build_sparse_embedder()
    store = get_vector_store()
    chunks = index_vault(
        vault_path, embedder, store, get_collection_name(),
        full=full, sparse_embedder=sparse_embedder,
    )
    if chunks:
        print(f"Indexed {len(chunks)} chunks.")


def cmd_search(args):
    embedder = build_embedder()
    sparse_embedder = build_sparse_embedder()
    store = get_vector_store()
    collection = get_collection_name()
    limit = max(1, min(args.limit, MAX_LIMIT))
    try:
        results = search(
            args.query, embedder, store, collection,
            limit=limit, score_threshold=args.score_threshold, sparse_embedder=sparse_embedder,
        )
    except CollectionNotFoundError:
        print(f"Collection '{collection}' not found. Run 'mnemolith index' first.")
        sys.exit(1)
    for r in reversed(results):
        heading = f" > {r['heading']}" if r.get('heading') else ""
        print("-"*70)
        print("\n")
        print(f"[{r['score']:.3f}] {r['path']}: {r['title']}{heading}")
        print("\n")
        print(r["content"])


def cmd_graphiti_index(args):
    """Index vault notes into the Graphiti temporal knowledge graph.

    Each note becomes one episode keyed on its vault-relative path so
    re-runs update entities + relationships rather than duplicating them.
    Reference time falls back to the file's mtime when no frontmatter
    date is present.
    """
    from mnemolith.graphiti_store import (
        VaultEpisode,
        add_episode,
        build_graphiti,
        ensure_indices,
    )
    from mnemolith.parser import parse_vault

    vault_path = args.vault_path or get_vault_path()
    if not Path(vault_path).is_dir():
        print(f"Error: '{vault_path}' is not a valid directory.")
        sys.exit(1)

    notes = parse_vault(vault_path)
    if args.limit:
        notes = notes[: args.limit]
    if not notes:
        print("No notes found in vault.")
        return

    vault_root = Path(vault_path)

    async def run():
        graphiti = build_graphiti()
        try:
            if args.init:
                print("Building Neo4j indices + constraints…")
                await ensure_indices(graphiti)
            ok, fail = 0, 0
            for n in notes:
                ref = _episode_reference_time(n, vault_root)
                episode = VaultEpisode(
                    name=n.path,
                    body=n.content,
                    reference_time=ref,
                    source_description=f"vault note · {n.title or n.path}",
                )
                try:
                    await add_episode(graphiti, episode)
                    ok += 1
                    if ok % 5 == 0:
                        print(f"  {ok}/{len(notes)} ingested…")
                except Exception as e:
                    fail += 1
                    print(f"  fail: {n.path} — {type(e).__name__}: {e}")
            print(f"Graphiti ingest complete: {ok} ok, {fail} failed, {len(notes)} total.")
        finally:
            await graphiti.close()

    asyncio.run(run())


def _episode_reference_time(note, vault_root):
    """Pick a reference time for a Graphiti episode.

    Frontmatter `date` / `created` / `updated` wins if parseable.
    Falls back to the file's mtime, then to now. Always returns
    tz-aware UTC.
    """
    from datetime import datetime

    fm = getattr(note, "frontmatter", None) or {}
    for key in ("date", "created", "updated"):
        v = fm.get(key) if isinstance(fm, dict) else None
        if isinstance(v, datetime):
            return v if v.tzinfo else v.replace(tzinfo=UTC)
        if isinstance(v, str):
            try:
                d = datetime.fromisoformat(v.replace("Z", "+00:00"))
                return d if d.tzinfo else d.replace(tzinfo=UTC)
            except ValueError:
                pass
    abs_path = vault_root / note.path
    if abs_path.exists():
        return datetime.fromtimestamp(abs_path.stat().st_mtime, tz=UTC)
    return datetime.now(UTC)


def cmd_graphiti_search(args):
    """One-off temporal search from the CLI for smoke testing."""
    from mnemolith.graphiti_store import build_graphiti, format_edge, temporal_search

    async def run():
        graphiti = build_graphiti()
        try:
            results = await temporal_search(graphiti, args.query, num_results=args.limit)
            if not results:
                print("(no results)")
                return
            for r in results:
                print("-" * 70)
                print(format_edge(r))
        finally:
            await graphiti.close()

    asyncio.run(run())


def cmd_backup(args):
    backup_dir = Path(args.dir) if args.dir else None
    path = create_backup(backup_dir)
    print(f"Backup created at: {path}")


def cmd_restore(args):
    path = Path(args.backup_path)
    if not path.is_dir():
        print(f"Error: '{path}' is not a valid directory.")
        sys.exit(1)
    restore_backup(path)
    print("Restore complete.")


def main():
    parser = argparse.ArgumentParser(prog="mnemolith")
    sub = parser.add_subparsers(dest="command")

    index_p = sub.add_parser("index", help="Index vault into vector store (incremental by default)")
    index_p.add_argument("vault_path", nargs="?", help="Path to vault (or use OBSIDIAN_VAULT_PATH)")
    index_p.add_argument("--full", action="store_true",
                         help="Drop the collection and rebuild from scratch")
    index_p.add_argument("--clean", action="store_true", help=argparse.SUPPRESS)
    index_p.set_defaults(func=cmd_index)

    search_p = sub.add_parser("search", help="Search indexed documents")
    search_p.add_argument("query", help="Search query")
    search_p.add_argument("--limit", type=int, default=5,
                          help=f"Max results (1-{MAX_LIMIT})")
    search_p.add_argument("--score-threshold", type=float, default=None,
                          help="Minimum similarity score (0-1) to include a result")
    search_p.set_defaults(func=cmd_search)

    backup_p = sub.add_parser("backup", help="Backup PostgreSQL and vector store data")
    backup_p.add_argument("--dir", help="Backup directory (or use BACKUP_DIR env var)")
    backup_p.set_defaults(func=cmd_backup)

    restore_p = sub.add_parser("restore", help="Restore from a backup directory")
    restore_p.add_argument("backup_path", help="Path to timestamped backup folder")
    restore_p.set_defaults(func=cmd_restore)

    g_index_p = sub.add_parser(
        "graphiti-index",
        help="Index vault notes into the Graphiti temporal knowledge graph",
    )
    g_index_p.add_argument("vault_path", nargs="?", help="Vault path (or OBSIDIAN_VAULT_PATH)")
    g_index_p.add_argument("--init", action="store_true",
                           help="Build Neo4j indices + constraints first (run once per Neo4j db)")
    g_index_p.add_argument("--limit", type=int, default=None,
                           help="Only ingest the first N notes (smoke testing)")
    g_index_p.set_defaults(func=cmd_graphiti_index)

    g_search_p = sub.add_parser(
        "graphiti-search",
        help="One-off temporal search across the Graphiti knowledge graph",
    )
    g_search_p.add_argument("query", help="Search query")
    g_search_p.add_argument("--limit", type=int, default=10,
                            help="Max results (default 10)")
    g_search_p.set_defaults(func=cmd_graphiti_search)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()

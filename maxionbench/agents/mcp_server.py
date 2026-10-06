"""MCP server (stdio) exposing `search` and `read` over the HotpotQA paragraph corpus.

    python -m maxionbench.agents.mcp_server [--dataset dataset/processed/hotpot_portable]
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from maxionbench.agents.hotpot_env import DEFAULT_DATASET, HotpotCorpus


def build_server(corpus: HotpotCorpus) -> FastMCP:
    server = FastMCP("maxionbench-hotpotqa", log_level="WARNING")

    @server.tool()
    def search(query: str, k: int = 5) -> str:
        """Search Wikipedia paragraphs by keywords. Returns up to k results as `[doc_id] snippet` lines."""
        hits = corpus.search(query, k)
        return "\n".join(f"[{h.doc_id}] {h.snippet}" for h in hits) or "no results"

    @server.tool()
    def read(doc_id: str) -> str:
        """Return the full text of one paragraph by its doc_id (from search results)."""
        text = corpus.read(doc_id)
        return text if text is not None else f"unknown doc_id {doc_id!r}"

    return server


def main(argv: list[str] | None = None) -> None:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args(argv)
    build_server(HotpotCorpus(args.dataset)).run("stdio")


if __name__ == "__main__":
    main()

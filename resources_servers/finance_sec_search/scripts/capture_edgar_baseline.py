#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Record edgar_search and filing-text behavior against a local EDGAR index.

Produces a fixture that a later run can replay to prove the tool still answers
identically. Requests are issued through the same request model and endpoint the
agent reaches, so argument coercion, date clamping, result shaping, error text
and JSON serialization are all captured rather than just the SQL layer.

Usage:
    python capture_edgar_baseline.py --index INDEX.sqlite --output baseline.json
    python capture_edgar_baseline.py --index INDEX.sqlite --output baseline.json \
        --corpus /path/to/step-0-download/data --from-metrics 'metrics/search-*.jsonl'

Replay with scripts/replay_edgar_baseline.py.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from nemo_gym.server_utils import ServerClient  # noqa: E402
from resources_servers.finance_sec_search.app import (  # noqa: E402
    EdgarSearchRequest,
    FinanceAgentResourcesServer,
    FinanceAgentResourcesServerConfig,
)
from resources_servers.sec_local_index.local_edgar_search import (  # noqa: E402
    MAX_END_DATE,
    fingerprint_source_index,
)


FIXTURE_SCHEMA_VERSION = 1

# Spans the axes the refactor moves: query language, filters, clamping, paging,
# argument coercion and every error string the tool can return.
QUERY_MATRIX: tuple[dict[str, Any], ...] = (
    {"search_query": "revenue"},
    {"search_query": "artificial intelligence"},
    {"search_query": '"artificial intelligence"'},
    {"search_query": "revenue OR earnings"},
    {"search_query": "revenue NOT guidance"},
    {"search_query": "revenue -guidance"},
    {"search_query": "cybersec*"},
    {"search_query": "*"},
    {"search_query": "revenue", "form_types": ["10-K"]},
    {"search_query": "revenue", "form_types": ["10-K", "10-Q"]},
    {"search_query": "revenue", "ciks": ["320193"]},
    {"search_query": "revenue", "ciks": ["0000320193"]},
    {"search_query": "revenue", "ciks": ["320193", "789019"]},
    {"search_query": "revenue", "form_types": ["10-K"], "ciks": ["320193"]},
    {"search_query": "revenue", "start_date": "2020-01-01", "end_date": "2021-12-31"},
    {"search_query": "revenue", "end_date": "2030-01-01"},
    {"search_query": "revenue", "start_date": "2030-01-01"},
    {"search_query": "revenue", "page": 2},
    {"search_query": "revenue", "page": 3},
    {"search_query": "revenue", "top_n_results": 1},
    {"search_query": "revenue", "top_n_results": 100},
    {"search_query": "quantum pineapple", "ciks": ["320193"]},
    {"search_query": "revenue", "form_types": '["10-K"]'},
    {"search_query": "revenue", "form_types": "['10-K']"},
    {"search_query": "revenue", "ciks": '["320193"]'},
    {"search_query": ""},
    {"search_query": "   "},
    {"search_query": "revenue", "start_date": "not-a-date"},
    {"search_query": "revenue", "start_date": "2022-01-01", "end_date": "2021-01-01"},
    {"search_query": "revenue", "ciks": ["APPLE"]},
    {"search_query": "(revenue)"},
    {"search_query": '"unterminated'},
    {"search_query": "revenue AND earnings"},
    {"search_query": "revenue OR"},
    {"search_query": "NOT"},
    {"search_query": "rev*enue"},
    {"search_query": '"phrase"*'},
)


def _server(config: FinanceAgentResourcesServerConfig) -> FinanceAgentResourcesServer:
    return FinanceAgentResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _session_request() -> MagicMock:
    request = MagicMock()
    request.session = {"session_id": "capture"}
    return request


def _index_identity(index_path: Path) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{index_path}?mode=ro&immutable=1", uri=True)
    try:
        documents = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        return {
            "name": index_path.name,
            "document_count": documents,
            "fingerprint": fingerprint_source_index(connection),
        }
    finally:
        connection.close()


def _recorded_queries(patterns: list[str]) -> list[dict[str, Any]]:
    """Load distinct search arguments out of local_edgar_metrics_dir records."""
    fields = ("search_query", "form_types", "ciks", "start_date", "end_date", "page", "top_n_results")
    seen: dict[str, dict[str, Any]] = {}
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open(path, encoding="utf-8") as stream:
                for line in stream:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    arguments = {
                        field: list(record[field]) if isinstance(record.get(field), list) else record.get(field)
                        for field in fields
                        if record.get(field) is not None
                    }
                    if arguments.get("search_query"):
                        seen.setdefault(json.dumps(arguments, sort_keys=True), arguments)
    return [seen[key] for key in sorted(seen)]


async def _capture_searches(
    server: FinanceAgentResourcesServer,
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    captured = []
    for arguments in cases:
        try:
            body = EdgarSearchRequest(**arguments)
        except Exception as error:
            captured.append({"arguments": arguments, "request_error": f"{type(error).__name__}: {error}"})
            continue
        response = await server.edgar_search(_session_request(), body)
        captured.append(
            {
                "arguments": arguments,
                "results": response.results,
                "results_sha256": hashlib.sha256(response.results.encode("utf-8")).hexdigest(),
            }
        )
    return captured


async def _capture_filings(
    server: FinanceAgentResourcesServer,
    searches: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    urls: list[str] = []
    for entry in searches:
        payload = entry.get("results")
        if not payload:
            continue
        parsed = json.loads(payload)
        if not isinstance(parsed, list):
            continue
        for result in parsed:
            url = str(result.get("filingUrl") or "")
            if url and url not in urls:
                urls.append(url)

    captured = []
    for url in urls[:limit]:
        before = Counter(server._filing_read_sources)
        try:
            text = await server._fetch_sec_filing_text(url)
        except Exception as error:
            captured.append({"url": url, "error": f"{type(error).__name__}: {error}"})
            continue
        advanced = Counter(server._filing_read_sources) - before
        source = next(iter(advanced), "unknown")
        captured.append(
            {
                "url": url,
                "source": source,
                "length": len(text),
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )
    return captured


async def capture(arguments: argparse.Namespace) -> dict[str, Any]:
    index_path = Path(arguments.index)
    if not index_path.is_file():
        raise SystemExit(f"Index not found: {index_path}")

    prompt_dir = Path(__file__).resolve().parents[1] / "prompt_templates"
    config = FinanceAgentResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="finance_sec_search_capture",
        cache_dir=arguments.cache_dir,
        use_cache=bool(arguments.cache_dir),
        judge_prompt_template_fpath=str(prompt_dir / "finance_sec_search_judge.yaml"),
        retrieval_system_prompt_fpath=str(prompt_dir / "finance_sec_search_retrieval.yaml"),
        local_edgar_index_path=str(index_path),
        local_edgar_metadata_path=arguments.metadata,
        sec_dump_path=arguments.corpus,
        max_end_date=arguments.max_end_date,
    )

    server = _server(config)
    cases = list(QUERY_MATRIX) + _recorded_queries(arguments.from_metrics or [])
    searches = await _capture_searches(server, cases)
    filings = await _capture_filings(server, searches, arguments.filings) if arguments.corpus else []

    if server._local_edgar_search is not None:
        server._local_edgar_search.close()

    return {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "index": _index_identity(index_path),
        "max_end_date": arguments.max_end_date,
        "edgar_search": searches,
        "filing_text": filings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", required=True, help="Read-only SQLite FTS5 index used by edgar_search")
    parser.add_argument("--output", required=True, help="Destination JSON fixture")
    parser.add_argument("--metadata", default=None, help="Metadata sidecar; defaults to the index plus '.metadata'")
    parser.add_argument("--corpus", default=None, help="Downloaded SEC corpus root, enabling filing-text capture")
    parser.add_argument("--cache-dir", default=None, help="Enables the filing-text cache during capture")
    parser.add_argument("--max-end-date", default=MAX_END_DATE, help="Cutoff applied to every captured search")
    parser.add_argument("--filings", type=int, default=25, help="How many filing texts to digest")
    parser.add_argument(
        "--digest-only",
        action="store_true",
        help="Record only the hash of each response, small enough to commit alongside the tests",
    )
    parser.add_argument(
        "--from-metrics",
        action="append",
        default=None,
        help="Glob of local_edgar_metrics_dir JSONL files whose queries are added to the matrix; repeatable",
    )
    arguments = parser.parse_args()

    fixture = asyncio.run(capture(arguments))

    searches = fixture["edgar_search"]
    errors = sum(1 for entry in searches if "error" in (entry.get("results") or entry.get("request_error") or ""))

    if arguments.digest_only:
        for entry in searches:
            entry.pop("results", None)

    output_path = Path(arguments.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"index documents: {fixture['index']['document_count']}")
    print(f"searches captured: {len(searches)} ({errors} returning an error)")
    print(f"filing texts captured: {len(fixture['filing_text'])}")
    print(f"wrote {output_path} ({output_path.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()

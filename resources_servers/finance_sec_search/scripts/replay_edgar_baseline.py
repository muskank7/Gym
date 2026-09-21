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
"""Replay a captured edgar_search fixture and report any behavior change.

Reissues every request recorded by capture_edgar_baseline.py against the current
code and compares the serialized responses byte for byte. Exits non-zero on the
first difference so it can gate a merge.

Usage:
    python replay_edgar_baseline.py --index INDEX.sqlite --baseline baseline.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from resources_servers.finance_sec_search.scripts.capture_edgar_baseline import (  # noqa: E402
    FIXTURE_SCHEMA_VERSION,
    capture,
)


def _diff_searches(expected: list[dict[str, Any]], actual: list[dict[str, Any]]) -> list[str]:
    by_arguments = {json.dumps(entry["arguments"], sort_keys=True): entry for entry in actual}
    differences = []
    for entry in expected:
        key = json.dumps(entry["arguments"], sort_keys=True)
        current = by_arguments.get(key)
        if current is None:
            differences.append(f"missing request: {key}")
            continue
        # A digest-only baseline carries no response text to show, so the hash is
        # the only comparable field.
        fields = ("results", "request_error") if "results" in entry else ("results_sha256", "request_error")
        for field in fields:
            if entry.get(field) != current.get(field):
                differences.append(
                    f"{field} changed for {key}\n  expected: {entry.get(field)!r}\n  actual:   {current.get(field)!r}"
                )
    return differences


def _diff_filings(expected: list[dict[str, Any]], actual: list[dict[str, Any]]) -> list[str]:
    by_url = {entry["url"]: entry for entry in actual}
    differences = []
    for entry in expected:
        current = by_url.get(entry["url"])
        if current is None:
            differences.append(f"missing filing: {entry['url']}")
            continue
        for field in ("sha256", "length", "error"):
            if entry.get(field) != current.get(field):
                differences.append(
                    f"{field} changed for {entry['url']}: expected {entry.get(field)!r}, got {current.get(field)!r}"
                )
    return differences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", required=True, help="The same index the baseline was captured from")
    parser.add_argument("--baseline", required=True, help="Fixture written by capture_edgar_baseline.py")
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--filings", type=int, default=None)
    parser.add_argument("--from-metrics", action="append", default=None)
    arguments = parser.parse_args()

    baseline = json.loads(Path(arguments.baseline).read_text(encoding="utf-8"))
    if baseline.get("schema_version") != FIXTURE_SCHEMA_VERSION:
        raise SystemExit(
            f"Baseline schema version {baseline.get('schema_version')} does not match {FIXTURE_SCHEMA_VERSION}"
        )

    arguments.max_end_date = baseline["max_end_date"]
    if arguments.filings is None:
        arguments.filings = len(baseline["filing_text"])

    current = asyncio.run(capture(arguments))

    if current["index"]["fingerprint"] != baseline["index"]["fingerprint"]:
        raise SystemExit(
            "Index fingerprint does not match the baseline, so a diff would not be attributable to the code. "
            f"Baseline covered {baseline['index']['document_count']} documents, this index holds "
            f"{current['index']['document_count']}."
        )

    differences = _diff_searches(baseline["edgar_search"], current["edgar_search"])
    differences += _diff_filings(baseline["filing_text"], current["filing_text"])

    if differences:
        print(f"{len(differences)} difference(s) against {arguments.baseline}:\n")
        for difference in differences:
            print(f"- {difference}")
        raise SystemExit(1)

    print(
        f"no change: {len(baseline['edgar_search'])} searches and "
        f"{len(baseline['filing_text'])} filing texts match {arguments.baseline}"
    )


if __name__ == "__main__":
    main()

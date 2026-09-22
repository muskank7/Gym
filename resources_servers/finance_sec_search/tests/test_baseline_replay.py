# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Regression gate over everything edgar_search returns to the model.

Runs the recorded query matrix against a synthetic index and compares digests
of the responses, so any change in query translation, filtering, ordering,
paging, clamping, coverage reporting, error wording or filing text fails here.

This is the CI-sized half of the gate. The real sap-500 index is tens of
gigabytes and cannot be committed, so before merging a change to search
behavior also replay against it by hand — see resources_servers/sec_local_index
/README.md. A run there covers 579,881 documents rather than the handful below.

On a deliberate change, re-record with:
    python resources_servers/finance_sec_search/tests/record_synthetic_baseline.py
"""

import asyncio
import json
from argparse import Namespace
from pathlib import Path

import pytest

from resources_servers.finance_sec_search.scripts.capture_edgar_baseline import capture
from resources_servers.finance_sec_search.scripts.replay_edgar_baseline import (
    _diff_filings,
    _diff_searches,
)
from resources_servers.finance_sec_search.tests.synthetic_index import build


BASELINE_FPATH = Path(__file__).resolve().parent / "data" / "edgar_baseline_synthetic.json"


def capture_arguments(index_path: Path, corpus_root: Path, max_end_date: str, filings: int) -> Namespace:
    return Namespace(
        index=str(index_path),
        metadata=None,
        corpus=str(corpus_root),
        cache_dir=None,
        max_end_date=max_end_date,
        filings=filings,
        from_metrics=None,
        digest_only=True,
    )


@pytest.fixture(scope="module")
def baseline() -> dict:
    return json.loads(BASELINE_FPATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def replayed(tmp_path_factory, baseline) -> dict:
    root = tmp_path_factory.mktemp("synthetic")
    index_path, corpus_root = build(root / "index.sqlite", root / "corpus")
    arguments = capture_arguments(
        index_path,
        corpus_root,
        baseline["max_end_date"],
        len(baseline["filing_text"]),
    )
    return asyncio.run(capture(arguments))


def test_the_synthetic_index_rebuilt_identically(replayed, baseline) -> None:
    """A moved fingerprint means the corpus changed, which would make every
    difference below unattributable."""
    assert replayed["index"]["fingerprint"] == baseline["index"]["fingerprint"]
    assert replayed["index"]["document_count"] == baseline["index"]["document_count"]


def test_every_recorded_search_still_returns_the_same_response(replayed, baseline) -> None:
    differences = _diff_searches(baseline["edgar_search"], replayed["edgar_search"])

    assert differences == [], "\n".join(differences)


def test_every_recorded_filing_still_reads_the_same(replayed, baseline) -> None:
    differences = _diff_filings(baseline["filing_text"], replayed["filing_text"])

    assert differences == [], "\n".join(differences)


def test_the_recorded_responses_actually_differ(baseline) -> None:
    """Guards the fixture itself. A baseline whose queries all returned the same
    thing, empty or an error, would satisfy the comparisons above while pinning
    nothing.
    """
    digests = [entry["results_sha256"] for entry in baseline["edgar_search"]]
    filings = baseline["filing_text"]

    assert len(digests) >= 20
    assert len(set(digests)) >= 15
    assert len(filings) >= 5
    assert len({entry["sha256"] for entry in filings}) == len(filings)


def test_filings_were_read_from_the_corpus(baseline) -> None:
    """A recording that silently fell through to sec.gov would neither be
    reproducible nor be testing local mode."""
    assert {entry["source"] for entry in baseline["filing_text"]} == {"sec-corpus"}

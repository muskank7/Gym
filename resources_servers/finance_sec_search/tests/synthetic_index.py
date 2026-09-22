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
"""A small EDGAR index and filing corpus that rebuild identically every time.

Lets the baseline replay that gates changes to edgar_search run in CI, where the
real index is far too large to keep. The rows are arranged to exercise what the
replay compares: several form types, several CIKs, filings on both sides of the
2025-04-07 cutoff, and dates that leave part of the query matrix outside the
indexed span.

Determinism is the point. Changing anything below invalidates
tests/data/edgar_baseline_synthetic.json, which then has to be re-recorded.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


CORPUS_PREFIX = "/workspace/outputs/finance/synthetic/workflow-2-download-sec/step-0-download/data"

# accession, cik, company, ticker, description, form_type, document_type, filing_date, filename, body
ROWS = (
    (
        "0000320193-22-000001",
        "320193",
        "Apple Inc.",
        "AAPL",
        "10-K",
        "10-K",
        "10-K",
        "2022-10-28",
        "aapl-20220924.htm",
        "revenue increased driven by iPhone net sales and services",
    ),
    (
        "0000320193-24-000001",
        "320193",
        "Apple Inc.",
        "AAPL",
        "10-K",
        "10-K",
        "10-K",
        "2024-11-01",
        "aapl-20240928.htm",
        "revenue of 391.0 billion and net income of 93.7 billion",
    ),
    (
        "0000320193-24-000002",
        "320193",
        "Apple Inc.",
        "AAPL",
        "EX-99.1",
        "8-K",
        "EX-99.1",
        "2024-11-02",
        "exhibit-991.htm",
        "quarterly dividend declared alongside revenue guidance",
    ),
    (
        "0000789019-23-000001",
        "789019",
        "Microsoft Corporation",
        "MSFT",
        "10-Q",
        "10-Q",
        "10-Q",
        "2023-04-25",
        "msft-10q.htm",
        "revenue grew across intelligent cloud and productivity segments",
    ),
    (
        "0000789019-25-000001",
        "789019",
        "Microsoft Corporation",
        "MSFT",
        "10-K",
        "10-K",
        "10-K",
        "2025-04-08",
        "msft-20250331.htm",
        "revenue and climate risk disclosures for the fiscal year",
    ),
    (
        "0001318605-21-000001",
        "1318605",
        "Tesla, Inc.",
        "TSLA",
        "10-K",
        "10-K",
        "10-K",
        "2021-02-08",
        "tsla-10k.htm",
        "automotive revenue and regulatory credits climate risk",
    ),
    (
        "0001045810-24-000001",
        "1045810",
        "NVIDIA Corporation",
        "NVDA",
        "10-K",
        "10-K",
        "10-K",
        "2024-02-21",
        "nvda-10k.htm",
        "data center revenue growth and shares outstanding detail",
    ),
    (
        "0001045810-25-000001",
        "1045810",
        "NVIDIA Corporation",
        "NVDA",
        "10-Q",
        "10-Q",
        "10-Q",
        "2025-02-26",
        "nvda-10q.htm",
        "revenue concentration and supply commitments",
    ),
)

SCHEMA = """
CREATE TABLE documents (
    id INTEGER PRIMARY KEY,
    accession_number TEXT NOT NULL,
    cik TEXT NOT NULL,
    company_name TEXT NOT NULL,
    ticker TEXT NOT NULL,
    description TEXT,
    form_type TEXT NOT NULL,
    document_type TEXT NOT NULL,
    filing_date TEXT NOT NULL,
    url TEXT NOT NULL,
    canonical_url_key TEXT NOT NULL,
    source_path TEXT NOT NULL,
    body TEXT NOT NULL
);
CREATE UNIQUE INDEX documents_url_key ON documents(canonical_url_key);
CREATE INDEX documents_filters ON documents(cik, form_type, filing_date);
CREATE VIRTUAL TABLE documents_fts USING fts5(
    body, content='documents', content_rowid='id'
);
"""


def _relative_path(ticker: str, form_type: str, filing_date: str, accession: str, filename: str) -> str:
    return f"{ticker}/{form_type}/{filing_date[:4]}/{accession}/{filename}"


def build_index(path: Path) -> Path:
    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA)

    records = []
    for index, row in enumerate(ROWS, start=1):
        accession, cik, company, ticker, description, form_type, document_type, filing_date, filename, body = row
        undashed = accession.replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{undashed}/{filename}"
        records.append(
            (
                index,
                accession,
                cik,
                company,
                ticker,
                description,
                form_type,
                document_type,
                filing_date,
                url,
                f"{cik}:{undashed}:{filename.lower()}",
                f"{CORPUS_PREFIX}/{_relative_path(ticker, form_type, filing_date, accession, filename)}",
                body,
            )
        )

    connection.executemany(
        """
        INSERT INTO documents (
            id, accession_number, cik, company_name, ticker, description,
            form_type, document_type, filing_date, url, canonical_url_key,
            source_path, body
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        records,
    )
    connection.executemany(
        "INSERT INTO documents_fts(rowid, body) VALUES (?, ?)",
        [(record[0], record[-1]) for record in records],
    )
    connection.commit()
    connection.close()
    return path


def build_corpus(root: Path) -> Path:
    """Write the filing HTML the index points at, so filing reads are covered too."""
    for row in ROWS:
        accession, _cik, company, ticker, _description, form_type, _type, filing_date, filename, body = row
        document = root / _relative_path(ticker, form_type, filing_date, accession, filename)
        document.parent.mkdir(parents=True, exist_ok=True)
        document.write_text(
            "<html><head><style>.x {color: red}</style></head><body>"
            f"<h1>{company}</h1><script>var a = 1;</script>"
            f"<p>{body}</p><table><tr><td>{form_type}</td><td>{filing_date}</td></tr></table>"
            "</body></html>",
            encoding="utf-8",
        )
    return root


def build(index_path: Path, corpus_root: Path) -> tuple[Path, Path]:
    return build_index(index_path), build_corpus(corpus_root)

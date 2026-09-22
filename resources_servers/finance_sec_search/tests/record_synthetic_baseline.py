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
"""Re-record tests/data/edgar_baseline_synthetic.json.

Run only when a change to edgar_search is deliberate, and read the diff before
committing it: every line is a response the model would have seen.

    python resources_servers/finance_sec_search/tests/record_synthetic_baseline.py
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from resources_servers.finance_sec_search.app import DEFAULT_MAX_END_DATE  # noqa: E402
from resources_servers.finance_sec_search.scripts.capture_edgar_baseline import capture  # noqa: E402
from resources_servers.finance_sec_search.tests.synthetic_index import build  # noqa: E402
from resources_servers.finance_sec_search.tests.test_baseline_replay import (  # noqa: E402
    BASELINE_FPATH,
    capture_arguments,
)


FILING_SAMPLE_SIZE = 8


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        index_path, corpus_root = build(root / "index.sqlite", root / "corpus")
        arguments = capture_arguments(index_path, corpus_root, DEFAULT_MAX_END_DATE, FILING_SAMPLE_SIZE)
        fixture = asyncio.run(capture(arguments))

    searches = fixture["edgar_search"]
    for entry in searches:
        entry.pop("results", None)

    BASELINE_FPATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_FPATH.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    errors = sum(1 for entry in searches if entry.get("request_error"))
    print(
        f"wrote {BASELINE_FPATH}\n"
        f"  searches: {len(searches)} ({errors} returning an error)\n"
        f"  filings:  {len(fixture['filing_text'])}"
    )


if __name__ == "__main__":
    main()

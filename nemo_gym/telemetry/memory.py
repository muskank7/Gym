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
"""Linux process-tree memory sampling for Gym-managed servers."""

import logging
import os
import sys
import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger(__name__)
_INSTRUMENTS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


@dataclass(frozen=True)
class ServerMemoryTarget:
    """One configured Gym server and the root process Gym spawned for it."""

    name: str
    server_type: str
    pid: int


@dataclass(frozen=True)
class ProcessTreeMemory:
    """One process tree's current memory."""

    rss_bytes: int
    pss_bytes: int | None
    process_count: int


def process_tree_memory_supported() -> bool:
    """Return whether this host exposes Linux procfs memory accounting."""
    return sys.platform.startswith("linux") and Path("/proc/self").is_dir()


def collect_process_tree_memory(root_pid: int) -> ProcessTreeMemory:
    """Measure current RSS and PSS for a root process and its descendants."""
    rss_bytes = 0
    pss_bytes = 0
    process_count = 0
    complete_pss = True
    for pid in _descendant_pids(root_pid):
        memory = _read_process_memory(pid)
        if memory is None:
            continue
        rss, pss = memory
        rss_bytes += rss
        process_count += 1
        if pss is None:
            complete_pss = False
        else:
            pss_bytes += pss
    return ProcessTreeMemory(
        rss_bytes=rss_bytes,
        pss_bytes=pss_bytes if complete_pss else None,
        process_count=process_count,
    )


def record_server_memory(
    target: ServerMemoryTarget,
    memory: ProcessTreeMemory,
    peaks: dict[tuple[str, str], int],
) -> None:
    """Record current and sampled-peak process-tree memory through Lens."""
    from nemo_gym.telemetry.setup import get_telemetry

    telemetry = get_telemetry()
    if telemetry is None or not telemetry.is_exporting:
        return
    try:
        instruments = _get_instruments(telemetry.meter)
        base_attributes = {
            "nemo.gym.server.name": target.name,
            "nemo.gym.server.type": target.server_type,
            "process.pid": target.pid,
        }
        instruments["processes"].set(memory.process_count, base_attributes)
        values = {"rss": memory.rss_bytes}
        if memory.pss_bytes is not None:
            values["pss"] = memory.pss_bytes
        for memory_type, value in values.items():
            attributes = {**base_attributes, "memory.type": memory_type}
            peak_key = (target.name, memory_type)
            peaks[peak_key] = max(peaks.get(peak_key, 0), value)
            instruments["usage"].set(value, attributes)
            instruments["sampled_peak"].set(peaks[peak_key], attributes)
    except Exception:
        logger.debug("Failed to record Gym server memory", exc_info=True)


class ServerMemoryProfiler:
    """Periodically measure and report Gym-managed server process trees."""

    def __init__(
        self,
        targets: list[ServerMemoryTarget],
        interval_seconds: float,
        *,
        collector: Callable[[int], ProcessTreeMemory] = collect_process_tree_memory,
    ) -> None:
        self._targets = targets
        self._interval_seconds = interval_seconds
        self._collector = collector
        self._peaks: dict[tuple[str, str], int] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def sample_once(self) -> None:
        """Record one sample for every configured server."""
        for target in self._targets:
            try:
                memory = self._collector(target.pid)
                record_server_memory(target, memory, self._peaks)
            except Exception:
                logger.debug(f"Failed to sample memory for {target.name}", exc_info=True)

    def start(self) -> None:
        """Start background sampling after recording an initial sample."""
        if self._thread is not None:
            return
        self.sample_once()
        self._thread = threading.Thread(
            target=self._run,
            name="nemo-gym-memory-profiler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop sampling and record the final live process-tree values."""
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        self.sample_once()
        self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self.sample_once()


def _descendant_pids(root_pid: int) -> set[int]:
    """Return a snapshot of a Linux process tree, including its root."""
    pids: set[int] = set()
    to_visit = [root_pid]
    while to_visit:
        pid = to_visit.pop()
        if pid in pids:
            continue
        pids.add(pid)
        try:
            with os.scandir(f"/proc/{pid}/task") as task_dirs:
                for task_dir in task_dirs:
                    try:
                        with open(f"{task_dir.path}/children") as children_file:
                            children = children_file.read().split()
                    except (OSError, PermissionError):
                        continue
                    to_visit.extend(int(child) for child in children if child.isdigit())
        except (OSError, PermissionError):
            continue
    return pids


def _read_process_memory(pid: int) -> tuple[int, int | None] | None:
    """Return ``(rss, pss)`` bytes for one process, or ``None`` after an exit race."""
    try:
        rss_kib = 0
        pss_kib = 0
        saw_rss = False
        saw_pss = False
        with open(f"/proc/{pid}/smaps_rollup") as smaps:
            for line in smaps:
                if line.startswith("Rss:"):
                    rss_kib = int(line.split()[1])
                    saw_rss = True
                elif line.startswith("Pss:"):
                    pss_kib = int(line.split()[1])
                    saw_pss = True
        if saw_rss:
            return rss_kib * 1024, pss_kib * 1024 if saw_pss else None
    except (OSError, PermissionError, ValueError, IndexError):
        pass

    try:
        with open(f"/proc/{pid}/statm") as statm:
            rss_pages = int(statm.read().split()[1])
        return rss_pages * os.sysconf("SC_PAGE_SIZE"), None
    except (OSError, PermissionError, ValueError, IndexError):
        return None


def _get_instruments(meter) -> dict:
    instruments = _INSTRUMENTS.get(meter)
    if instruments is None:
        instruments = {
            "usage": meter.create_gauge(
                name="gym.server.memory.usage",
                unit="By",
                description="Current memory used by a Gym server and its descendants.",
            ),
            "sampled_peak": meter.create_gauge(
                name="gym.server.memory.sampled_peak",
                unit="By",
                description="Largest sampled memory usage for a Gym server and its descendants.",
            ),
            "processes": meter.create_gauge(
                name="gym.server.processes",
                unit="{process}",
                description="Processes currently attributed to a Gym server.",
            ),
        }
        _INSTRUMENTS[meter] = instruments
    return instruments

"""Pure rules of the GPU stability check.

A clock/voltage table that survives games for an evening can still return
wrong results, and one that passes a benchmark can still hang on the next
load step. The check runs a load that verifies its own results
(memtest_vulkan), watches the kernel log for amdgpu faults while it runs, and
stops at a temperature limit. This module only decides: how much memory the
load may take, which kernel lines are faults, what the load's output means and
which outcome an attempt gets. Running it lives in
``bc250cc.infrastructure.gpu.stability_check``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

#: memtest_vulkan without a size takes all free RAM but 400 MB on an APU; on
#: a BC-250 that once took 9.6 GB of GTT and the OOM killer took the desktop.
DEFAULT_MEMORY_MIB = 2048
MINIMUM_MEMORY_MIB = 256
DEFAULT_MINUTES = 10
MINIMUM_SECONDS = 60
MAXIMUM_SECONDS = 4 * 60 * 60
#: The governor throttles at 85 °C on the packaged config; the check stops
#: a little above that, before the board's own protection.
DEFAULT_MAX_EDGE_C = 90
DEFAULT_MAX_TCTL_C = 95

OUTCOME_PASS = "PASS"
OUTCOME_ERRORS = "ERRORS"
OUTCOME_FAULT = "FAULT"
OUTCOME_HOT = "HOT"
OUTCOME_LOAD_FAIL = "LOAD-FAIL"
OUTCOME_CRASH = "CRASH"
OUTCOME_ABORTED = "ABORTED"
#: Outcomes that say the settings are not stable.
FAILURES = frozenset({OUTCOME_ERRORS, OUTCOME_FAULT, OUTCOME_CRASH})

# amdgpu trouble worth stopping for. The patterns come from real BC-250 logs:
# GCVM page faults and ring timeouts before a hang, "Dumping IP State" and
# devcoredump on a reset, and "Not enough memory for command submission".
_FAULT = re.compile(
    r"amdgpu.*(PROTECTION_FAULT|page fault|ring [a-z_0-9.]+ timeout|GPU reset|gpu reset"
    r"|Dumping IP State|IB test failed|failed to|\bhang|timed out|\*ERROR\*)"
    r"|devcoredump|drm.*\*ERROR\*",
    re.IGNORECASE,
)
# Lines that match the above on every healthy BC-250 boot: governor clock
# changes on kernels without the clock type, display IRQ/HPD noise and the
# vendor infoframe a DP-to-HDMI dummy plug refuses.
_NOT_A_FAULT = re.compile(r"Unsupported clock type|dal_irq_service|\bhpd\b|hpd\(rx\)|vendor infoframe", re.IGNORECASE)
_PROGRESS = re.compile(r"iteration\. Passed .*?written:\s+\S+\s+([0-9.]+)GB/sec")
_ERROR_REPORT = re.compile(r"^Error found\.", re.MULTILINE)
_RUNTIME_ERROR = re.compile(r"Runtime error|early exit", re.IGNORECASE)
#: GNU timeout's exit status when the time ran out (the normal end here).
TIMEOUT_EXIT = 124


@dataclass(frozen=True)
class StabilityPlan:
    seconds: int
    memory_mib: int
    max_edge_c: int
    max_tctl_c: int


def plan_check(
    *,
    minutes: float = DEFAULT_MINUTES,
    memory_mib: int | None = None,
    available_mib: int,
    max_edge_c: int = DEFAULT_MAX_EDGE_C,
    max_tctl_c: int = DEFAULT_MAX_TCTL_C,
) -> StabilityPlan:
    """The check to run. Memory is capped at half of what is available now."""
    seconds = int(round(float(minutes) * 60))
    if not MINIMUM_SECONDS <= seconds <= MAXIMUM_SECONDS:
        raise ValueError(
            f"The check runs from {MINIMUM_SECONDS // 60} to {MAXIMUM_SECONDS // 60} minutes."
        )
    if not 60 <= int(max_edge_c) <= 100 or not 60 <= int(max_tctl_c) <= 100:
        raise ValueError("Temperature limits must be between 60 and 100 °C.")
    requested = DEFAULT_MEMORY_MIB if memory_mib is None else int(memory_mib)
    if requested < MINIMUM_MEMORY_MIB:
        raise ValueError(f"The load needs at least {MINIMUM_MEMORY_MIB} MiB.")
    ceiling = max(int(available_mib) // 2, 0)
    if ceiling < MINIMUM_MEMORY_MIB:
        raise ValueError(
            f"Only {int(available_mib)} MiB of memory is available; close programs and retry."
        )
    return StabilityPlan(seconds, min(requested, ceiling), int(max_edge_c), int(max_tctl_c))


def fault_lines(lines: Iterable[str]) -> tuple[str, ...]:
    """The kernel log lines that report amdgpu trouble."""
    return tuple(
        line for line in lines
        if _FAULT.search(line) and not _NOT_A_FAULT.search(line)
    )


@dataclass(frozen=True)
class LoadReport:
    error_reports: int
    progress_lines: int
    write_gbps: float | None
    runtime_error: str


def read_load_output(text: str) -> LoadReport:
    """What memtest_vulkan printed: error reports, progress and its last speed."""
    speeds = _PROGRESS.findall(text)
    runtime = next((line.strip() for line in text.splitlines() if _RUNTIME_ERROR.search(line)), "")
    return LoadReport(
        error_reports=len(_ERROR_REPORT.findall(text)),
        progress_lines=len(speeds),
        write_gbps=float(speeds[-1]) if speeds else None,
        runtime_error=runtime,
    )


@dataclass
class SampleStats:
    count: int = 0
    max_edge_c: int = 0
    max_tctl_c: int = 0
    clock_sum: int = 0
    power_sum: float = 0.0
    clock_max: int = 0

    def add(self, *, edge_c: int, tctl_c: int, clock_mhz: int, power_w: float) -> None:
        self.count += 1
        self.max_edge_c = max(self.max_edge_c, int(edge_c))
        self.max_tctl_c = max(self.max_tctl_c, int(tctl_c))
        self.clock_sum += int(clock_mhz)
        self.clock_max = max(self.clock_max, int(clock_mhz))
        self.power_sum += float(power_w)

    @property
    def average_clock_mhz(self) -> int:
        return round(self.clock_sum / self.count) if self.count else 0

    @property
    def average_power_w(self) -> float:
        return round(self.power_sum / self.count, 1) if self.count else 0.0


def too_hot(plan: StabilityPlan, *, edge_c: int, tctl_c: int) -> bool:
    return int(edge_c) >= plan.max_edge_c or int(tctl_c) >= plan.max_tctl_c


@dataclass(frozen=True)
class Verdict:
    outcome: str
    note: str
    faults: tuple[str, ...] = field(default=())


def decide(
    *,
    report: LoadReport,
    exit_code: int,
    faults: Sequence[str],
    stopped_hot: bool,
    stopped_by_user: bool,
) -> Verdict:
    """The outcome of one finished attempt, worst evidence first.

    With arguments memtest_vulkan prints no verdict when it is interrupted,
    so a pass is: it ran until the timeout, reported progress and found
    nothing."""
    if faults:
        return Verdict(OUTCOME_FAULT, faults[0].strip(), tuple(faults))
    if report.error_reports:
        return Verdict(OUTCOME_ERRORS, f"memtest_vulkan reported {report.error_reports} error(s)")
    if stopped_hot:
        return Verdict(OUTCOME_HOT, "stopped at the temperature limit")
    if stopped_by_user:
        return Verdict(OUTCOME_ABORTED, "stopped before the end")
    if report.runtime_error:
        return Verdict(OUTCOME_LOAD_FAIL, report.runtime_error)
    if exit_code != TIMEOUT_EXIT or not report.progress_lines:
        return Verdict(OUTCOME_LOAD_FAIL, f"memtest_vulkan stopped early (exit {exit_code})")
    return Verdict(OUTCOME_PASS, "")

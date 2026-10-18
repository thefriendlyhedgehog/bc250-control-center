"""Runs the GPU stability check and keeps its results.

Everything here runs as the desktop user: memtest_vulkan is a user-space
Vulkan program, the kernel log is readable by the journal's reader groups, and
the sensors are plain amdgpu/k10temp sysfs files (nothing that goes through
I2C or the SMU). Nothing is written to hardware; the clocks and voltages under
test are whatever the governor applies.

Before the load starts a marker records the attempt. If the board hangs, the
next check (or ``status``) finds the marker from an older boot and records the
attempt as a crash, with the amdgpu lines of that boot.
"""

from __future__ import annotations

import hashlib
import io
import json
import lzma
import os
import re
import shutil
import signal
import subprocess  # nosec B404 - fixed argv only
import tarfile
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from bc250cc.application.gpu import stability as rules
from bc250cc.infrastructure.gpu_live_clock import read_hwmon_clock_mhz
from bc250cc.infrastructure.persistence.config_paths import app_data_dir

#: memtest_vulkan, GpuZelenograd (zlib licence). Downloaded at runtime and
#: checked against the SHA-256 GitHub publishes for this release asset.
MEMTEST_VERSION = "0.5.0"
MEMTEST_URL = (
    "https://github.com/GpuZelenograd/memtest_vulkan/releases/download/"
    f"v{MEMTEST_VERSION}/memtest_vulkan-v{MEMTEST_VERSION}_DesktopLinux_X86_64.tar.xz"
)
MEMTEST_SHA256 = "0e058c28b9d0d6eb04f5edeebce73e8106f88c4ea307d46e98c715da61117d52"
MEMTEST_ARCHIVE_BYTES = 272336
MEMTEST_DIR = f"memtest_vulkan-v{MEMTEST_VERSION}"

RESULTS_NAME = "estabilidad.json"
PENDING_NAME = "estabilidad-pending.json"
MAX_RESULTS = 100
SAMPLE_SECONDS = 5
GOVERNOR_CONFIG = Path("/etc/cyan-skillfish-governor-smu/config.toml")
GOVERNOR_BUS = "com.cyanskillfish.Governor"
GOVERNOR_CURRENT_RANGE = "/com/cyanskillfish/Governor/Range/Current"
BC250_PCI_ID = ("0x1002", "0x13fe")


# ------------------------------------------------------------------ places --
def data_dir() -> Path:
    return app_data_dir() / "Data"


def tools_dir() -> Path:
    return app_data_dir() / "ResourceTools"


def runs_dir() -> Path:
    state = os.environ.get("XDG_STATE_HOME", "").strip()
    root = Path(state).expanduser() if state else Path.home() / ".local" / "state"
    return root / "bc250-control-center" / "stability"


def find_memtest(explicit: str | Path | None = None) -> Path | None:
    """memtest_vulkan: the given path, Control Center's tool folder, $PATH, ~/.local/bin."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.append(tools_dir() / MEMTEST_DIR / "memtest_vulkan")
    found = shutil.which("memtest_vulkan")
    if found:
        candidates.append(Path(found))
    candidates.append(Path.home() / ".local" / "bin" / "memtest_vulkan")
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return path
    return None


def install_memtest(
    *, opener: Callable[[str], object] = urllib.request.urlopen, destination: Path | None = None
) -> Path:
    """Download the pinned memtest_vulkan, check it and unpack only its binary."""
    destination = destination or tools_dir() / MEMTEST_DIR
    with opener(MEMTEST_URL) as response:  # nosec B310 - fixed https URL
        archive = response.read(MEMTEST_ARCHIVE_BYTES * 4)
    digest = hashlib.sha256(archive).hexdigest()
    if digest != MEMTEST_SHA256:
        raise RuntimeError(
            f"The memtest_vulkan download does not match its published SHA-256 ({digest}); nothing was installed."
        )
    with tarfile.open(fileobj=io.BytesIO(lzma.decompress(archive)), mode="r:") as tar:
        member = next(
            (m for m in tar.getmembers() if m.isfile() and Path(m.name).name == "memtest_vulkan"),
            None,
        )
        if member is None:
            raise RuntimeError("The memtest_vulkan archive has no memtest_vulkan binary.")
        payload = tar.extractfile(member)
        data = payload.read() if payload else b""
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "memtest_vulkan"
    partial = target.with_suffix(".partial")
    partial.write_bytes(data)
    partial.chmod(0o755)
    partial.replace(target)
    return target


# ----------------------------------------------------------------- sensors --
@dataclass(frozen=True)
class GpuSensors:
    device: Path
    hwmon: Path
    tctl: Path | None

    def edge_c(self) -> int:
        return _milli(self.hwmon / "temp1_input")

    def tctl_c(self) -> int:
        return _milli(self.tctl) if self.tctl else 0

    def clock_mhz(self) -> int:
        return read_hwmon_clock_mhz(self.device) or 0

    def power_w(self) -> float:
        for name in ("power1_average", "power1_input"):
            try:
                return round(int((self.hwmon / name).read_text().strip()) / 1_000_000, 1)
            except (OSError, ValueError):
                continue
        return 0.0


def _milli(path: Path | None) -> int:
    try:
        return int(path.read_text().strip()) // 1000 if path else 0
    except (OSError, ValueError):
        return 0


def find_sensors(sys_root: Path = Path("/sys")) -> GpuSensors | None:
    """The BC-250's amdgpu hwmon and the CPU's k10temp; None on other hardware."""
    gpu = None
    for card in sorted((sys_root / "class" / "drm").glob("card[0-9]*")):
        device = card / "device"
        try:
            ids = ((device / "vendor").read_text().strip(), (device / "device").read_text().strip())
        except OSError:
            continue
        if ids != BC250_PCI_ID:
            continue
        for hwmon in sorted((device / "hwmon").glob("hwmon*")):
            if _read(hwmon / "name") == "amdgpu":
                gpu = (device, hwmon)
                break
        if gpu:
            break
    if gpu is None:
        return None
    tctl = None
    for hwmon in sorted((sys_root / "class" / "hwmon").glob("hwmon*")):
        if _read(hwmon / "name") == "k10temp":
            tctl = hwmon / "temp1_input"
            break
    return GpuSensors(device=gpu[0], hwmon=gpu[1], tctl=tctl)


def _read(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def available_mib(meminfo: Path = Path("/proc/meminfo")) -> int:
    match = re.search(r"^MemAvailable:\s+(\d+)", _read(meminfo), re.MULTILINE)
    return int(match.group(1)) // 1024 if match else 0


def boot_id() -> str:
    return _read(Path("/proc/sys/kernel/random/boot_id")) or "unknown"


# ------------------------------------------------------------- kernel log --
def kernel_log(*args: str, runner=subprocess.run) -> list[str] | None:
    """journalctl -k lines, or None when the log can't be read."""
    try:
        done = runner(  # nosec B603
            ["journalctl", "-k", "--no-pager", "-o", "short-iso", *args],
            capture_output=True, text=True, timeout=20, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout.splitlines()


def kernel_log_readable(runner=subprocess.run) -> bool:
    lines = kernel_log("-b", "0", "-n", "1", runner=runner)
    return bool(lines)


def active_cu_count(runner=subprocess.run) -> int | None:
    for line in reversed(kernel_log("-b", "0", runner=runner) or []):
        match = re.search(r"active_cu_number (\d+)", line)
        if match:
            return int(match.group(1))
    return None


def governor_max_mhz(path: Path = GOVERNOR_CONFIG, runner=subprocess.run) -> int | None:
    """The ceiling Cyan applies now (its live D-Bus range), else its config's.

    The live range is what the check runs at: a range set from the GPU page
    or with SetRange is not written to the TOML."""
    try:
        done = runner(  # nosec B603
            ["busctl", "--system", "get-property", GOVERNOR_BUS, GOVERNOR_CURRENT_RANGE,
             f"{GOVERNOR_BUS}.Range", "Max"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        live = re.fullmatch(r"u (\d+)", (done.stdout or "").strip()) if done.returncode == 0 else None
        if live:
            return int(live.group(1))
    except (OSError, subprocess.SubprocessError):
        pass
    match = re.search(r"^\s*max\s*=\s*(\d+)", _read(path), re.MULTILINE)
    return int(match.group(1)) if match else None


# ----------------------------------------------------------------- results --
@dataclass
class StabilityResult:
    at: float
    boot: str
    outcome: str
    note: str = ""
    seconds: int = 0
    memory_mib: int = 0
    max_edge_c: int = 0
    max_tctl_c: int = 0
    average_clock_mhz: int = 0
    max_clock_mhz: int = 0
    average_power_w: float = 0.0
    write_gbps: float | None = None
    governor_max_mhz: int | None = None
    active_cus: int | None = None
    kernel: str = ""
    faults: list[str] = field(default_factory=list)
    log: str = ""


def read_results(path: Path | None = None) -> list[StabilityResult]:
    """Every recorded check, oldest first."""
    path = path or data_dir() / RESULTS_NAME
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    known = set(StabilityResult.__dataclass_fields__)
    results = []
    for entry in raw.get("runs", []) if isinstance(raw, dict) else []:
        try:
            results.append(StabilityResult(**{k: v for k, v in entry.items() if k in known}))
        except TypeError:
            continue
    return results


def append_result(result: StabilityResult, path: Path | None = None) -> None:
    path = path or data_dir() / RESULTS_NAME
    runs = [asdict(r) for r in read_results(path)] + [asdict(result)]
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    partial.write_text(
        json.dumps({"version": 1, "runs": runs[-MAX_RESULTS:]}, indent=1), encoding="utf-8"
    )
    os.replace(partial, path)


def recover_interrupted(
    *, current_boot: str | None = None, data: Path | None = None, runner=subprocess.run
) -> StabilityResult | None:
    """Turn a marker left by a check that never finished into a result.

    A marker from an older boot means the board went down during the load: a
    CRASH, with that boot's amdgpu fault lines. A marker from this boot means
    the check itself was killed: ABORTED."""
    data = data or data_dir()
    marker = data / PENDING_NAME
    try:
        pending = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    current_boot = current_boot or boot_id()
    crashed = pending.get("boot") != current_boot
    faults = rules.fault_lines(kernel_log("-b", "-1", runner=runner) or []) if crashed else ()
    result = StabilityResult(
        at=float(pending.get("at", time.time())),
        boot=str(pending.get("boot", "")),
        outcome=rules.OUTCOME_CRASH if crashed else rules.OUTCOME_ABORTED,
        note="the system went down during the check" if crashed else "the check was stopped",
        seconds=int(pending.get("seconds", 0)),
        memory_mib=int(pending.get("memory_mib", 0)),
        governor_max_mhz=pending.get("governor_max_mhz"),
        active_cus=pending.get("active_cus"),
        kernel=str(pending.get("kernel", "")),
        faults=list(faults[-5:]),
        log=str(pending.get("log", "")),
    )
    append_result(result, data / RESULTS_NAME)
    marker.unlink(missing_ok=True)
    return result


# --------------------------------------------------------------------- run --
@dataclass(frozen=True)
class Sample:
    elapsed_s: int
    edge_c: int
    tctl_c: int
    clock_mhz: int
    power_w: float


class StabilityCheck:
    """One check: start the load, watch it, stop it, record the verdict."""

    def __init__(
        self,
        plan: rules.StabilityPlan,
        *,
        memtest: Path,
        sensors: GpuSensors,
        data: Path | None = None,
        runs: Path | None = None,
        popen=subprocess.Popen,
        runner=subprocess.run,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
        on_sample: Callable[[Sample], None] | None = None,
    ):
        self.plan = plan
        self.memtest = memtest
        self.sensors = sensors
        self.data = data or data_dir()
        self.runs = runs or runs_dir()
        self._popen = popen
        self._runner = runner
        self._sleep = sleep
        self._clock = clock
        self._on_sample = on_sample
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def run(self) -> StabilityResult:
        started = self._clock()
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(started))
        workdir = self.runs / stamp
        workdir.mkdir(parents=True, exist_ok=True)
        log_path = workdir / "memtest.log"
        monitor_path = workdir / "monitor.csv"
        since = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started))
        context = {
            "governor_max_mhz": governor_max_mhz(runner=self._runner),
            "active_cus": active_cu_count(self._runner),
            "kernel": os.uname().release,
        }
        marker = self.data / PENDING_NAME
        self.data.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({
            "at": started, "boot": boot_id(), "seconds": self.plan.seconds,
            "memory_mib": self.plan.memory_mib, "log": str(log_path), **context,
        }), encoding="utf-8")

        stats = rules.SampleStats()
        stopped_hot = False
        faults: tuple[str, ...] = ()
        argv = [
            "timeout", "-s", "INT", f"{self.plan.seconds}s",
            str(self.memtest), "0", str(self.plan.memory_mib * 1024 * 1024),
        ]
        with log_path.open("w", encoding="utf-8") as log, monitor_path.open("w", encoding="utf-8") as monitor:
            monitor.write("elapsed_s,edge_c,tctl_c,clock_mhz,power_w\n")
            process = self._popen(  # nosec B603
                argv, cwd=workdir, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
            _raise_oom_score(process.pid)
            stopping = False
            try:
                while process.poll() is None:
                    self._sleep(SAMPLE_SECONDS)
                    if process.poll() is not None:
                        break
                    sample = Sample(
                        elapsed_s=int(self._clock() - started),
                        edge_c=self.sensors.edge_c(), tctl_c=self.sensors.tctl_c(),
                        clock_mhz=self.sensors.clock_mhz(), power_w=self.sensors.power_w(),
                    )
                    stats.add(edge_c=sample.edge_c, tctl_c=sample.tctl_c,
                              clock_mhz=sample.clock_mhz, power_w=sample.power_w)
                    monitor.write(
                        f"{sample.elapsed_s},{sample.edge_c},{sample.tctl_c},{sample.clock_mhz},{sample.power_w}\n"
                    )
                    monitor.flush()
                    if self._on_sample:
                        self._on_sample(sample)
                    if not stopping and rules.too_hot(self.plan, edge_c=sample.edge_c, tctl_c=sample.tctl_c):
                        stopped_hot = stopping = True
                        _interrupt(process)
                    faults = rules.fault_lines(kernel_log("--since", since, runner=self._runner) or [])
                    if not stopping and (faults or self._stop_requested):
                        stopping = True
                        _interrupt(process)
            except KeyboardInterrupt:
                self._stop_requested = True
                _interrupt(process)
            while True:
                try:
                    exit_code = _wait(process)
                    break
                except KeyboardInterrupt:
                    # A second Ctrl+C while the load winds down.
                    self._stop_requested = True
                    _interrupt(process)
        faults = rules.fault_lines(kernel_log("--since", since, runner=self._runner) or [])
        report = rules.read_load_output(log_path.read_text(encoding="utf-8", errors="replace"))
        verdict = rules.decide(
            report=report, exit_code=exit_code, faults=faults,
            stopped_hot=stopped_hot, stopped_by_user=self._stop_requested and not faults,
        )
        result = StabilityResult(
            at=started, boot=boot_id(), outcome=verdict.outcome, note=verdict.note,
            seconds=int(self._clock() - started), memory_mib=self.plan.memory_mib,
            max_edge_c=stats.max_edge_c, max_tctl_c=stats.max_tctl_c,
            average_clock_mhz=stats.average_clock_mhz, max_clock_mhz=stats.clock_max,
            average_power_w=stats.average_power_w, write_gbps=report.write_gbps,
            faults=list(faults[:5]), log=str(log_path), **context,
        )
        append_result(result, self.data / RESULTS_NAME)
        marker.unlink(missing_ok=True)
        return result


def _raise_oom_score(pid: int) -> None:
    # If memory runs out, the kernel kills the load first, not the desktop.
    # Raising a child's score needs no privilege; a failure only loses that.
    try:
        Path(f"/proc/{pid}/oom_score_adj").write_text("1000")
    except OSError:
        pass


def _interrupt(process) -> None:
    # SIGINT reaches timeout and memtest_vulkan together (own session).
    try:
        os.killpg(process.pid, signal.SIGINT)
    except (OSError, AttributeError):
        try:
            process.send_signal(signal.SIGINT)
        except OSError:
            pass


def _wait(process, grace: float = 20.0) -> int:
    try:
        return process.wait(timeout=grace if process.poll() is None else None)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.wait()


def describe(result: StabilityResult) -> str:
    """One line for a terminal."""
    when = time.strftime("%Y-%m-%d %H:%M", time.localtime(result.at))
    parts = [f"{when}  {result.outcome}"]
    if result.seconds:
        parts.append(f"{result.seconds // 60} min")
    if result.average_clock_mhz:
        parts.append(f"avg {result.average_clock_mhz} MHz")
    if result.governor_max_mhz:
        parts.append(f"governor max {result.governor_max_mhz} MHz")
    if result.max_edge_c:
        parts.append(f"edge {result.max_edge_c} °C")
    if result.average_power_w:
        parts.append(f"{result.average_power_w} W")
    if result.write_gbps:
        parts.append(f"{result.write_gbps} GB/s")
    if result.active_cus:
        parts.append(f"{result.active_cus} CUs")
    text = ", ".join(parts)
    return f"{text}  ({result.note})" if result.note else text


def summarize(results: Iterable[StabilityResult]) -> dict[str, object]:
    results = list(results)
    last = results[-1] if results else None
    return {
        "checks": len(results),
        "last": asdict(last) if last else None,
        "failures": sum(1 for r in results if r.outcome in rules.FAILURES),
    }

import hashlib
import io
import json
import lzma
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from bc250cc.application.gpu import stability as rules
from bc250cc.infrastructure.gpu import stability_check as check

PROGRESS = (
    "     41 iteration. Passed  1.0018 seconds  written:   90.0GB 370.9GB/sec"
    "        checked:  180.0GB 237.1GB/sec\n"
)
FAULT = "2026-10-18T09:00:00 kevin-board kernel: amdgpu 0000:01:00.0: amdgpu: GPU reset begin!"


class Sensors:
    def __init__(self, edges):
        self.edges = list(edges)

    def edge_c(self):
        return self.edges.pop(0) if len(self.edges) > 1 else self.edges[0]

    def tctl_c(self):
        return 70

    def clock_mhz(self):
        return 2000

    def power_w(self):
        return 122.0


class Load:
    """memtest_vulkan under timeout: runs for a number of polls, then exits."""

    pid = 2**31 - 7

    def __init__(self, polls, exit_code, output, stdout):
        self.polls = polls
        self.exit_code = exit_code
        self.interrupted = False
        stdout.write(output)
        stdout.flush()

    def poll(self):
        if self.interrupted or self.polls <= 0:
            return self.exit_code
        self.polls -= 1
        return None

    def wait(self, timeout=None):
        return self.exit_code


def _journal(lines_for_since=()):
    calls = []

    def runner(argv, **_kwargs):
        calls.append(argv)
        if "--since" in argv:
            return SimpleNamespace(returncode=0, stdout="\n".join(lines_for_since))
        return SimpleNamespace(returncode=0, stdout="kernel: amdgpu: active_cu_number 40\n")

    runner.calls = calls
    return runner


@pytest.fixture
def interrupts(monkeypatch):
    sent = []
    monkeypatch.setattr(check.os, "killpg", lambda pid, sig: sent.append((pid, sig)))
    return sent


def _check(tmp_path, *, polls, exit_code, output, edges=(80,), journal=None):
    loads = []

    def popen(argv, **kwargs):
        assert argv[:4] == ["timeout", "-s", "INT", "600s"]
        assert argv[-2:] == ["0", str(2048 * 1024 * 1024)]
        assert kwargs["start_new_session"] is True
        assert kwargs["cwd"].parent == tmp_path / "runs"
        loads.append(Load(polls, exit_code, output, kwargs["stdout"]))
        return loads[-1]

    now = [1_760_000_000.0]

    def sleep(seconds):
        now[0] += seconds

    plan = rules.StabilityPlan(600, 2048, 90, 95)
    runner = journal or _journal()
    stability = check.StabilityCheck(
        plan, memtest=Path("/opt/memtest_vulkan"), sensors=Sensors(edges),
        data=tmp_path / "data", runs=tmp_path / "runs", popen=popen,
        runner=runner, sleep=sleep, clock=lambda: now[0],
    )
    return stability, loads


def test_a_clean_run_to_the_timeout_passes_and_is_recorded(tmp_path, interrupts):
    stability, _loads = _check(tmp_path, polls=4, exit_code=rules.TIMEOUT_EXIT, output=PROGRESS)
    result = stability.run()

    assert result.outcome == rules.OUTCOME_PASS
    assert (result.average_clock_mhz, result.max_edge_c, result.write_gbps) == (2000, 80, 370.9)
    assert result.active_cus == 40
    assert interrupts == []
    assert not (tmp_path / "data" / check.PENDING_NAME).exists()
    assert check.read_results(tmp_path / "data" / check.RESULTS_NAME) == [result]
    monitor = Path(result.log).with_name("monitor.csv").read_text().splitlines()
    assert monitor[0] == "elapsed_s,edge_c,tctl_c,clock_mhz,power_w"
    assert len(monitor) == 3


def test_the_temperature_limit_stops_the_load(tmp_path, interrupts, monkeypatch):
    stability, loads = _check(
        tmp_path, polls=50, exit_code=rules.TIMEOUT_EXIT, output=PROGRESS, edges=(80, 91),
    )

    def stop(pid, sig):
        interrupts.append((pid, sig))
        loads[0].interrupted = True

    monkeypatch.setattr(check.os, "killpg", stop)
    result = stability.run()
    assert result.outcome == rules.OUTCOME_HOT
    assert result.max_edge_c == 91
    assert len(interrupts) == 1


def test_an_amdgpu_fault_stops_the_load_and_wins(tmp_path, interrupts):
    stability, _loads = _check(
        tmp_path, polls=2, exit_code=rules.TIMEOUT_EXIT, output=PROGRESS,
        journal=_journal([FAULT]),
    )
    result = stability.run()
    assert result.outcome == rules.OUTCOME_FAULT
    assert result.faults == [FAULT]
    assert interrupts


def test_errors_in_the_load_output_fail_the_check(tmp_path, interrupts):
    output = PROGRESS + "Error found. Mode NEXT_RE_READ, total errors 0x20 out of 0x1C000000\n"
    stability, _loads = _check(tmp_path, polls=1, exit_code=rules.TIMEOUT_EXIT, output=output)
    assert stability.run().outcome == rules.OUTCOME_ERRORS


def test_ctrl_c_stops_the_load_and_records_an_aborted_check(tmp_path, interrupts):
    stability, _loads = _check(tmp_path, polls=50, exit_code=130, output=PROGRESS)

    def sleep(_seconds):
        raise KeyboardInterrupt

    stability._sleep = sleep
    result = stability.run()
    assert result.outcome == rules.OUTCOME_ABORTED
    assert interrupts


def test_a_marker_from_an_older_boot_is_recorded_as_a_crash(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / check.PENDING_NAME).write_text(json.dumps({
        "at": 1_760_000_000.0, "boot": "old-boot", "seconds": 600, "memory_mib": 2048,
        "governor_max_mhz": 2000, "active_cus": 40, "kernel": "7.2.10", "log": "/x/memtest.log",
    }))
    runner = _journal()

    def previous_boot(argv, **kwargs):
        if argv[-2:] == ["-b", "-1"]:
            return SimpleNamespace(returncode=0, stdout=FAULT + "\nkernel: usb 1-2: new device\n")
        return runner(argv, **kwargs)

    result = check.recover_interrupted(current_boot="new-boot", data=data, runner=previous_boot)
    assert result.outcome == rules.OUTCOME_CRASH
    assert result.faults == [FAULT]
    assert (result.governor_max_mhz, result.active_cus) == (2000, 40)
    assert not (data / check.PENDING_NAME).exists()
    assert check.read_results(data / check.RESULTS_NAME)[-1].outcome == rules.OUTCOME_CRASH
    assert check.recover_interrupted(current_boot="new-boot", data=data) is None


def test_a_marker_from_this_boot_is_an_aborted_check(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / check.PENDING_NAME).write_text(json.dumps({"at": 1.0, "boot": "same"}))
    result = check.recover_interrupted(current_boot="same", data=data, runner=_journal())
    assert result.outcome == rules.OUTCOME_ABORTED
    assert result.faults == []


def test_results_keep_the_newest_hundred_and_skip_bad_entries(tmp_path):
    path = tmp_path / check.RESULTS_NAME
    for index in range(check.MAX_RESULTS + 5):
        check.append_result(check.StabilityResult(at=float(index), boot="b", outcome="PASS"), path)
    results = check.read_results(path)
    assert len(results) == check.MAX_RESULTS
    assert results[0].at == 5.0
    path.write_text(json.dumps({"version": 1, "runs": [{"at": 1.0}, {"at": 2.0, "boot": "b", "outcome": "PASS", "extra": 1}]}))
    assert [r.at for r in check.read_results(path)] == [2.0]
    path.write_text("not json")
    assert check.read_results(path) == []


def _archive(binary=b"\x7fELF memtest"):
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        info = tarfile.TarInfo(f"{check.MEMTEST_DIR}/memtest_vulkan")
        info.size = len(binary)
        tar.addfile(info, io.BytesIO(binary))
    return lzma.compress(raw.getvalue())


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def test_install_checks_the_sha256_before_writing_anything(tmp_path):
    destination = tmp_path / "tools"
    with pytest.raises(RuntimeError, match="SHA-256"):
        check.install_memtest(opener=lambda _url: _Response(_archive()), destination=destination)
    assert not destination.exists()


def test_install_unpacks_only_the_binary(tmp_path, monkeypatch):
    archive = _archive()
    monkeypatch.setattr(check, "MEMTEST_SHA256", hashlib.sha256(archive).hexdigest())
    seen = []

    def opener(url):
        seen.append(url)
        return _Response(archive)

    target = check.install_memtest(opener=opener, destination=tmp_path / "tools")
    assert seen == [check.MEMTEST_URL]
    assert target.read_bytes() == b"\x7fELF memtest"
    assert target.stat().st_mode & 0o777 == 0o755
    assert sorted(p.name for p in target.parent.iterdir()) == ["memtest_vulkan"]


def test_sensors_are_found_only_on_a_bc250(tmp_path):
    device = tmp_path / "class" / "drm" / "card1" / "device"
    hwmon = device / "hwmon" / "hwmon3"
    hwmon.mkdir(parents=True)
    (device / "vendor").write_text("0x1002\n")
    (device / "device").write_text("0x13fe\n")
    (hwmon / "name").write_text("amdgpu\n")
    (hwmon / "temp1_input").write_text("81000\n")
    (hwmon / "power1_average").write_text("122000000\n")
    k10 = tmp_path / "class" / "hwmon" / "hwmon1"
    k10.mkdir(parents=True)
    (k10 / "name").write_text("k10temp\n")
    (k10 / "temp1_input").write_text("72500\n")

    sensors = check.find_sensors(tmp_path)
    assert (sensors.edge_c(), sensors.tctl_c(), sensors.power_w()) == (81, 72, 122.0)
    (device / "device").write_text("0x73bf\n")
    assert check.find_sensors(tmp_path) is None


def test_kernel_log_reports_an_unreadable_journal_as_none():
    def denied(_argv, **_kwargs):
        return SimpleNamespace(returncode=1, stdout="")

    def missing(_argv, **_kwargs):
        raise FileNotFoundError("journalctl")

    def slow(argv, **_kwargs):
        raise subprocess.TimeoutExpired(argv, 20)

    for runner in (denied, missing, slow):
        assert check.kernel_log("-b", "0", runner=runner) is None
        assert check.kernel_log_readable(runner=runner) is False
    assert check.active_cu_count(_journal()) == 40


def test_memory_and_governor_readers(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 16000000 kB\nMemAvailable:   8192000 kB\n")
    assert check.available_mib(meminfo) == 8000
    assert check.available_mib(tmp_path / "missing") == 0
    config = tmp_path / "config.toml"
    config.write_text("[frequency]\nmin = 1000\nmax = 2000\n")

    def no_bus(_argv, **_kwargs):
        raise FileNotFoundError("busctl")

    def live(argv, **_kwargs):
        assert argv[-3:] == [check.GOVERNOR_CURRENT_RANGE, "com.cyanskillfish.Governor.Range", "Max"]
        return SimpleNamespace(returncode=0, stdout="u 1500\n")

    assert check.governor_max_mhz(config, runner=no_bus) == 2000
    assert check.governor_max_mhz(config, runner=live) == 1500
    assert check.governor_max_mhz(tmp_path / "missing", runner=no_bus) is None


def test_describe_and_summarize():
    passed = check.StabilityResult(
        at=1_760_000_000.0, boot="b", outcome="PASS", seconds=600, average_clock_mhz=1998,
        governor_max_mhz=2000, max_edge_c=84, average_power_w=122.0, write_gbps=370.9, active_cus=40,
    )
    crashed = check.StabilityResult(at=1_760_001_000.0, boot="c", outcome="CRASH", note="went down")
    line = check.describe(passed)
    assert "PASS, 10 min, avg 1998 MHz, governor max 2000 MHz, edge 84 °C" in line
    assert check.describe(crashed).endswith("CRASH  (went down)")
    summary = check.summarize([passed, crashed])
    assert (summary["checks"], summary["failures"], summary["last"]["outcome"]) == (2, 1, "CRASH")
    assert check.summarize([]) == {"checks": 0, "last": None, "failures": 0}


def test_the_repository_opens_the_cli_check_in_the_console():
    from bc250cc.infrastructure.gpu_repository import GPURepository
    from bc250cc.infrastructure.system_service import SistemaService

    opened = []

    class Repository(GPURepository):
        def _abrir_terminal(self, command, title=""):
            opened.append((command, title))
            return "terminal"

    assert Repository().abrir_prueba_estabilidad_gpu(15) == "terminal"
    command, title = opened[0]
    assert title == "GPU stability check"
    assert command.endswith("-m frontends.cli stability run --minutes 15")
    with pytest.raises(ValueError):
        Repository().abrir_prueba_estabilidad_gpu("10; reboot")
    assert len(opened) == 1
    assert callable(SistemaService.abrir_prueba_estabilidad_gpu)
    assert callable(SistemaService.ultima_prueba_estabilidad_gpu)

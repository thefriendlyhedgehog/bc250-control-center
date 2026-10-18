import json
from pathlib import Path

import pytest

from bc250cc.application.gpu import stability as rules
from bc250cc.infrastructure.gpu import stability_check as check
from frontends.cli import main


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(check, "data_dir", lambda: tmp_path / "data")
    monkeypatch.setattr(check, "tools_dir", lambda: tmp_path / "tools")
    monkeypatch.setattr(check, "runs_dir", lambda: tmp_path / "runs")
    return tmp_path / "data"


def _result(outcome="PASS", **values):
    return check.StabilityResult(at=1_760_000_000.0, boot="b", outcome=outcome, seconds=600, **values)


def test_status_without_results(data, capsys):
    assert main(["stability", "status"]) == 0
    assert capsys.readouterr().out.strip() == "No stability check has run yet."


def test_status_lists_results_and_summarizes_as_json(data, capsys):
    check.append_result(_result(average_clock_mhz=2000), data / check.RESULTS_NAME)
    check.append_result(_result("FAULT", note="GPU reset"), data / check.RESULTS_NAME)
    assert main(["stability", "status", "--limit", "1"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1 and "FAULT" in lines[0] and "GPU reset" in lines[0]

    assert main(["--json", "stability", "status"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert (summary["checks"], summary["failures"]) == (2, 1)


def test_status_records_a_check_the_board_never_finished(data, capsys, monkeypatch):
    data.mkdir(parents=True)
    (data / check.PENDING_NAME).write_text(json.dumps({"at": 1.0, "boot": "older"}))
    monkeypatch.setattr(check, "boot_id", lambda: "now")
    monkeypatch.setattr(check, "kernel_log", lambda *_args, **_kwargs: [])
    assert main(["stability", "status"]) == 0
    out = capsys.readouterr().out
    assert "An earlier check did not finish" in out and "CRASH" in out


def test_a_missing_tool_is_not_downloaded_without_consent(data, capsys, monkeypatch):
    monkeypatch.setattr(check, "find_memtest", lambda _explicit: None)
    monkeypatch.setattr(check, "install_memtest", lambda: pytest.fail("downloaded without consent"))
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert main(["stability", "run"]) == 2
    out = capsys.readouterr().out
    assert check.MEMTEST_URL in out and "Not downloaded" in out


def test_install_tool_with_yes_downloads(data, capsys, monkeypatch):
    monkeypatch.setattr(check, "find_memtest", lambda _explicit: None)
    monkeypatch.setattr(check, "install_memtest", lambda: Path("/t/memtest_vulkan"))
    assert main(["stability", "install-tool", "--yes"]) == 0
    assert "SHA-256 verified" in capsys.readouterr().out


def test_run_refuses_other_hardware_and_an_unreadable_kernel_log(data, capsys, monkeypatch):
    monkeypatch.setattr(check, "find_memtest", lambda _explicit: Path("/t/memtest_vulkan"))
    monkeypatch.setattr(check, "find_sensors", lambda: None)
    assert main(["stability", "run"]) == 1
    assert "No BC-250 GPU" in capsys.readouterr().err

    monkeypatch.setattr(check, "find_sensors", lambda: object())
    monkeypatch.setattr(check, "kernel_log_readable", lambda: False)
    assert main(["stability", "run"]) == 1
    assert "systemd-journal" in capsys.readouterr().err


def test_run_plans_runs_and_reports(data, capsys, monkeypatch):
    monkeypatch.setattr(check, "find_memtest", lambda _explicit: Path("/t/memtest_vulkan"))
    monkeypatch.setattr(check, "find_sensors", lambda: object())
    monkeypatch.setattr(check, "kernel_log_readable", lambda: True)
    monkeypatch.setattr(check, "available_mib", lambda: 3000)
    monkeypatch.setattr(check, "governor_max_mhz", lambda: 1850)
    planned = []

    class Check:
        def __init__(self, plan, **kwargs):
            planned.append(plan)
            self.on_sample = kwargs["on_sample"]

        def run(self):
            self.on_sample(check.Sample(60, 80, 70, 1850, 120.0))
            return _result(outcome, log="/r/memtest.log")

    monkeypatch.setattr(check, "StabilityCheck", Check)
    outcome = rules.OUTCOME_PASS
    assert main(["stability", "run", "--minutes", "5"]) == 0
    assert planned[-1] == rules.StabilityPlan(300, 1500, 90, 95)
    out = capsys.readouterr().out
    assert "max 1850 MHz" in out and "Result:" in out and "Log: /r/memtest.log" in out

    outcome = rules.OUTCOME_ERRORS
    assert main(["--json", "stability", "run"]) == 3
    captured = capsys.readouterr()
    assert json.loads(captured.out)["outcome"] == rules.OUTCOME_ERRORS
    assert "GPU stability check" in captured.err


def test_run_rejects_a_plan_outside_the_limits(data, capsys, monkeypatch):
    monkeypatch.setattr(check, "find_memtest", lambda _explicit: Path("/t/memtest_vulkan"))
    monkeypatch.setattr(check, "find_sensors", lambda: object())
    monkeypatch.setattr(check, "kernel_log_readable", lambda: True)
    monkeypatch.setattr(check, "available_mib", lambda: 8000)
    assert main(["stability", "run", "--minutes", "500"]) == 2
    assert "minutes" in capsys.readouterr().err

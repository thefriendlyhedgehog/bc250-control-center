import pytest

from bc250cc.application.gpu import stability as rules

PASSING_OUTPUT = """\
      1 iteration. Passed  0.0250 seconds  written:    2.2GB 377.5GB/sec        checked:    4.5GB 236.6GB/sec
     41 iteration. Passed  1.0018 seconds  written:   90.0GB 370.9GB/sec        checked:  180.0GB 237.1GB/sec
"""
ERROR_OUTPUT = PASSING_OUTPUT + (
    "Error found. Mode NEXT_RE_READ, total errors 0x20 out of 0x1C000000 (0.00000011%)\n"
)


def test_plan_caps_memory_at_half_of_what_is_available():
    plan = rules.plan_check(minutes=10, memory_mib=8192, available_mib=6000)
    assert plan == rules.StabilityPlan(600, 3000, 90, 95)
    assert rules.plan_check(available_mib=12000).memory_mib == rules.DEFAULT_MEMORY_MIB


@pytest.mark.parametrize(
    "kwargs",
    (
        {"minutes": 0.5},
        {"minutes": 241},
        {"memory_mib": 128},
        {"max_edge_c": 110},
        {"max_tctl_c": 40},
        {"available_mib": 300},
    ),
)
def test_plan_refuses_values_outside_the_safe_range(kwargs):
    with pytest.raises(ValueError):
        rules.plan_check(**{"available_mib": 12000, **kwargs})


def test_fault_lines_keep_amdgpu_trouble_and_drop_healthy_boot_noise():
    lines = [
        "2026-10-18T09:00:00 kevin-board kernel: amdgpu 0000:01:00.0: amdgpu: [gfxhub] page fault (src_id:0 ring:24 vmid:3)",
        "2026-10-18T09:00:01 kevin-board kernel: amdgpu 0000:01:00.0: amdgpu: GCVM_L2_PROTECTION_FAULT_STATUS:0x00301031",
        "2026-10-18T09:00:02 kevin-board kernel: [drm:amdgpu_job_timedout [amdgpu]] *ERROR* ring gfx_0.0.0 timeout",
        "2026-10-18T09:00:03 kevin-board kernel: amdgpu 0000:01:00.0: amdgpu: Unsupported clock type",
        "2026-10-18T09:00:04 kevin-board kernel: amdgpu 0000:01:00.0: [drm] *ERROR* Failed to setup vendor infoframe on connector DP-1: -22",
        "2026-10-18T09:00:05 kevin-board kernel: amdgpu 0000:01:00.0: amdgpu: dal_irq_service: failed to ack hpd(rx)",
        "2026-10-18T09:00:06 kevin-board kernel: usb 1-2: new high-speed USB device",
    ]
    assert rules.fault_lines(lines) == tuple(lines[:3])


def test_load_output_counts_progress_errors_and_the_last_write_speed():
    report = rules.read_load_output(PASSING_OUTPUT)
    assert report == rules.LoadReport(0, 2, 370.9, "")
    assert rules.read_load_output(ERROR_OUTPUT).error_reports == 1
    failed = rules.read_load_output("Runtime error: no Vulkan device\n")
    assert failed.runtime_error == "Runtime error: no Vulkan device"
    assert failed.write_gbps is None


def test_sample_stats_average_and_peak():
    stats = rules.SampleStats()
    assert (stats.average_clock_mhz, stats.average_power_w) == (0, 0.0)
    stats.add(edge_c=80, tctl_c=70, clock_mhz=2000, power_w=120.0)
    stats.add(edge_c=84, tctl_c=72, clock_mhz=1900, power_w=124.5)
    assert (stats.max_edge_c, stats.max_tctl_c, stats.clock_max) == (84, 72, 2000)
    assert (stats.average_clock_mhz, stats.average_power_w) == (1950, 122.2)


def test_too_hot_stops_at_either_limit():
    plan = rules.StabilityPlan(600, 2048, 90, 95)
    assert not rules.too_hot(plan, edge_c=89, tctl_c=94)
    assert rules.too_hot(plan, edge_c=90, tctl_c=60)
    assert rules.too_hot(plan, edge_c=60, tctl_c=95)


def _decide(output=PASSING_OUTPUT, *, exit_code=rules.TIMEOUT_EXIT, faults=(), hot=False, user=False):
    return rules.decide(
        report=rules.read_load_output(output), exit_code=exit_code, faults=faults,
        stopped_hot=hot, stopped_by_user=user,
    )


def test_a_pass_needs_the_timeout_progress_and_no_errors():
    assert _decide().outcome == rules.OUTCOME_PASS
    assert _decide(exit_code=0).outcome == rules.OUTCOME_LOAD_FAIL
    assert _decide("").outcome == rules.OUTCOME_LOAD_FAIL
    assert _decide("Runtime error: lost device\n").outcome == rules.OUTCOME_LOAD_FAIL


def test_the_worst_evidence_decides_the_outcome():
    fault = "kernel: amdgpu: GPU reset begin!"
    verdict = _decide(ERROR_OUTPUT, faults=(fault,), hot=True, user=True)
    assert verdict == rules.Verdict(rules.OUTCOME_FAULT, fault, (fault,))
    assert _decide(ERROR_OUTPUT, hot=True, user=True).outcome == rules.OUTCOME_ERRORS
    assert _decide(hot=True, user=True).outcome == rules.OUTCOME_HOT
    assert _decide(user=True).outcome == rules.OUTCOME_ABORTED
    assert rules.FAILURES == {rules.OUTCOME_ERRORS, rules.OUTCOME_FAULT, rules.OUTCOME_CRASH}

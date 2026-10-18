"""Headless diagnostics and planning CLI for BC250 Control Center.

Mutating dependency application is deliberately double-gated. Hardware write
operations are not exposed here; they must continue through typed privileged
helpers and their subsystem-specific safety workflows.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import shutil
import signal

# The argv is fixed; generated commands come only from repository strategies.
import subprocess  # nosec B404
import sys
from pathlib import Path

from bc250cc.application.headless_dispatch import dispatch_safe
from bc250cc.infrastructure.persistence.config_paths import app_data_dir
from bc250cc.platform.packages.strategies.detector import (
    detect_os_info,
    read_os_release,
)
from bc250cc.platform.packages.strategies.factory import create_os_repository
from bc250cc.shared.failure_text import describe_failure
from bc250cc.shared.version import __version__


class HeadlessHost:
    def _os_release(self):
        return read_os_release()

    def _command_path(self, name: str) -> str:
        return shutil.which(name) or ""

    def _tool_dir(self) -> Path:
        return app_data_dir() / "ResourceTools"

    def _home(self) -> Path:
        return Path.home()

    @staticmethod
    def _execute_readonly(command, timeout=2):
        try:
            completed = subprocess.run(  # nosec B603
                command,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
            return completed.returncode, completed.stdout or "", completed.stderr or ""
        except (OSError, subprocess.SubprocessError) as error:
            return 1, "", str(error)


def _emit(payload, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    elif isinstance(payload, dict):
        for key, value in payload.items():
            print(f"{key}: {value}")
    else:
        print(payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bc250-control-center-cli",
        description="BC250 Control Center safe headless interface",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("system", help="Show detected Linux strategy")
    commands.add_parser("telemetry", help="Read APU sensor validity, clocks and firmware diagnostics")
    commands.add_parser("components", help="Show preparation component capabilities")
    integrations = commands.add_parser("integrations", help="Audit external-tool contracts and checkouts")
    integrations.add_argument(
        "--runtime", action="store_true",
        help="Include read-only local checkout origin/revision/worktree evidence",
    )
    commands.add_parser(
        "quick-access",
        help="Inspect optional Decky Quick Access deployment without changing it",
    )
    integrations.add_argument(
        "--tool", default="",
        help="Limit output to one declared integration key",
    )
    release_gates = commands.add_parser(
        "release-gates",
        help="Show static, legal and supervised hardware release blockers",
    )
    release_gates.add_argument(
        "--evidence", action="append", default=[], metavar="PATH",
        help="Validate and include a supervised qualification evidence manifest",
    )
    qualification = commands.add_parser(
        "qualification",
        help="Render or validate supervised hardware qualification evidence",
    )
    qualification.add_argument(
        "action", choices=("template", "validate", "export", "validate-bundle")
    )
    qualification.add_argument("path", nargs="?")
    qualification.add_argument(
        "--output", help="Destination for a portable verified qualification ZIP"
    )
    qualification.add_argument(
        "--section",
        choices=("cpu_smu", "gpu_governors", "compute_units", "fan_pwm", "core_unlock_and_boot"),
    )

    dependency = commands.add_parser("dependencies", help="Plan, check or explicitly apply dependencies")
    dependency.add_argument("--component", choices=("all", "runtime", "governor", "stress", "sensors", "umr"), default="all")
    dependency.add_argument("--mode", choices=("plan", "check", "apply"), default="plan")
    dependency.add_argument("--cu-manager-script", default="")
    dependency.add_argument("--allow-system-changes", action="store_true")
    dependency.add_argument("--yes", action="store_true")
    dependency.add_argument("--render-command", action="store_true")

    log = commands.add_parser("parse-log", help="Parse structured evidence from a workflow log")
    log.add_argument("path")
    log.add_argument("--exit-code", type=int)

    recovery = commands.add_parser("recovery", help="Create, list, verify or inspect a recovery bundle")
    recovery.add_argument("action", choices=("create", "list", "verify", "plan", "export"))
    recovery.add_argument("snapshot", nargs="?")
    recovery.add_argument("--label", default="manual-before-change")
    recovery.add_argument("--output", help="Destination for a portable verified recovery ZIP")

    profiles = commands.add_parser("profiles", help="Export, preview or import portable profiles")
    profiles.add_argument("action", choices=("export", "preview", "import"))
    profiles.add_argument("path")
    profiles.add_argument("--yes", action="store_true", help="Confirm replacement during import")

    metrics = commands.add_parser("metrics", help="Read or export recorded runtime metrics")
    metrics.add_argument("action", choices=("list", "export"))
    metrics.add_argument("--path")
    metrics.add_argument("--format", choices=("jsonl", "csv"), default="jsonl")
    metrics.add_argument("--limit", type=int, default=1000)
    metrics.add_argument("--since", type=float)

    stability = commands.add_parser(
        "stability",
        help="Run the GPU stability check (memtest_vulkan + amdgpu fault watch) or show its results",
    )
    stability.add_argument("action", choices=("run", "status", "install-tool"))
    stability.add_argument("--minutes", type=float, default=10.0)
    stability.add_argument("--memory-mib", type=int, help="Memory the load may use (default 2048, at most half of what is free)")
    stability.add_argument("--max-edge", type=int, default=90, help="Stop at this GPU edge temperature (°C)")
    stability.add_argument("--max-tctl", type=int, default=95, help="Stop at this CPU Tctl (°C)")
    stability.add_argument("--memtest", help="Path of a memtest_vulkan binary to use")
    stability.add_argument("--yes", action="store_true", help="Download memtest_vulkan without asking when it is missing")
    stability.add_argument("--limit", type=int, default=10, help="Results shown by status")
    return parser


def _confirm(question: str) -> bool:
    if not sys.stdin.isatty():
        return False
    try:
        return input(f"{question} [y/N] ").strip().lower() in {"y", "yes"}
    except EOFError:
        return False


def _run_stability(args) -> int:
    """The GPU stability check. It writes nothing to hardware: it runs a
    verifying load at whatever clocks and voltages the governor applies."""
    from bc250cc.application.gpu import stability as rules
    from bc250cc.infrastructure.gpu import stability_check as check

    recovered = check.recover_interrupted()
    if recovered is not None and not args.json:
        print("An earlier check did not finish:")
        print("  " + check.describe(recovered))
        for line in recovered.faults:
            print("    " + line)

    if args.action == "status":
        results = check.read_results()
        if args.json:
            _emit(check.summarize(results), as_json=True)
        elif not results:
            print("No stability check has run yet.")
        else:
            for result in results[-max(args.limit, 1):]:
                print(check.describe(result))
        return 0

    # With --json, stdout carries only the JSON result; the progress goes to stderr.
    say = functools.partial(print, file=sys.stderr if args.json else sys.stdout)
    memtest = check.find_memtest(args.memtest)
    if args.action == "install-tool" or memtest is None:
        if memtest is not None:
            say(f"memtest_vulkan is already installed: {memtest}")
            return 0
        size_kib = check.MEMTEST_ARCHIVE_BYTES // 1024
        say(f"The check needs memtest_vulkan {check.MEMTEST_VERSION} (zlib licence, {size_kib} KiB).")
        say(f"  from: {check.MEMTEST_URL}")
        say(f"  to:   {check.tools_dir() / check.MEMTEST_DIR}")
        if not (args.yes or _confirm("Download it and check its SHA-256?")):
            say("Not downloaded. Put a memtest_vulkan binary on $PATH or pass --memtest PATH.")
            return 2
        try:
            memtest = check.install_memtest()
        except (OSError, RuntimeError, ValueError) as error:
            print(f"memtest_vulkan was not installed: {error}", file=sys.stderr)
            return 1
        say(f"Installed {memtest} (SHA-256 verified).")
        if args.action == "install-tool":
            return 0

    sensors = check.find_sensors()
    if sensors is None:
        print("No BC-250 GPU (1002:13fe) with an amdgpu hwmon was found.", file=sys.stderr)
        return 1
    if not check.kernel_log_readable():
        print(
            "The kernel log can't be read as this user, so GPU faults would go unseen. "
            "Add the user to the systemd-journal group (or wheel) and log in again.",
            file=sys.stderr,
        )
        return 1
    try:
        plan = rules.plan_check(
            minutes=args.minutes, memory_mib=args.memory_mib,
            available_mib=check.available_mib(),
            max_edge_c=args.max_edge, max_tctl_c=args.max_tctl,
        )
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2

    governor_max = check.governor_max_mhz()
    say(f"GPU stability check: memtest_vulkan for {plan.seconds // 60} min, up to {plan.memory_mib} MiB.")
    say(
        f"Clock: the governor's (max {governor_max or '?'} MHz). "
        f"Stops at edge {plan.max_edge_c} °C or Tctl {plan.max_tctl_c} °C, or on an amdgpu fault."
    )
    say("An unstable GPU can hang the board: save your work. After a hang, run this again to record it.")
    say("Ctrl+C stops the check.")
    interactive = (sys.stderr if args.json else sys.stdout).isatty()
    last_line = [0.0]

    def show(sample):
        line = (
            f"{sample.elapsed_s:5d} s  edge {sample.edge_c:3d} °C  Tctl {sample.tctl_c:3d} °C  "
            f"{sample.clock_mhz:4d} MHz  {sample.power_w:5.1f} W"
        )
        if interactive:
            say("\r  " + line, end="", flush=True)
        elif sample.elapsed_s - last_line[0] >= 60:
            last_line[0] = sample.elapsed_s
            say("  " + line, flush=True)

    def stop(_signum, _frame):
        # A closed console sends SIGHUP or SIGTERM: stop the load and record
        # the attempt instead of leaving memtest_vulkan running on its own.
        raise KeyboardInterrupt

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGHUP)}
    try:
        result = check.StabilityCheck(plan, memtest=memtest, sensors=sensors, on_sample=show).run()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if interactive:
        say()
    if args.json:
        _emit(dataclasses.asdict(result), as_json=True)
    else:
        print("Result: " + check.describe(result))
        for line in result.faults:
            print("  " + line)
        print(f"Log: {result.log}")
    return 0 if result.outcome == rules.OUTCOME_PASS else 3


def _run_dependencies(args, host, runner) -> int:
    if (
        args.mode == "apply"
        and not args.render_command
        and not (args.allow_system_changes and args.yes)
    ):
        raise SystemExit(
            "Refusing apply: pass both --allow-system-changes and --yes "
            "after reviewing --mode plan."
        )
    repository = create_os_repository(host)
    command = repository.prepare_dependencies_command(
        args.component, cu_manager_script=args.cu_manager_script, mode=args.mode
    )
    if args.render_command:
        _emit({"mode": args.mode, "command": command}, as_json=args.json)
        return 0
    completed = runner(["bash", "-lc", command], check=False)
    status = int(completed.returncode)
    if status != 0:
        # Returning the bare status left the terminal with a number and no
        # explanation. Keep the status as the process exit code, but say what
        # it means and give the identifier support can search for.
        print(
            describe_failure(status, getattr(completed, "stdout", ""), getattr(completed, "stderr", "")),
            file=sys.stderr,
        )
    return status


def main(argv: list[str] | None = None, *, host=None, runner=subprocess.run) -> int:
    args = _parser().parse_args(argv)
    if args.command == "stability":
        return _run_stability(args)
    host = host or HeadlessHost()
    os_info = detect_os_info(
        host._os_release(), has_rpm_ostree=bool(host._command_path("rpm-ostree"))
    )
    result = dispatch_safe(args, os_info, host=host)
    if result is not None:
        _emit(result.payload, as_json=args.json)
        return result.exit_code
    return _run_dependencies(args, host, runner)
if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Cortex-M4F and Cortex-M7 verification and performance regression gate for
eigen-mcu.

For each MCU profile this script:
  1. Builds and runs test_mcu/test_mcu_kernels.cpp under QEMU with linker-wrapped
     heap traps (--wrap=malloc/calloc/realloc/free) and -Werror.
  2. On Cortex-M4F (float-only), verifies that zero software FP64 routines
     (__aeabi_d*, __aeabi_*2d) are referenced by test or benchmark objects.
  3. Builds and runs test_mcu/bench_mcu_kernels.cpp under QEMU (-icount shift=0)
     and checks instruction counts, stack high-water marks, and .text size
     against test_mcu/accepted.json for the active arm-none-eabi-gcc version.
  4. Optionally compile-checks additional Cortex-M targets (M33, M3).

Usage:
  python3 test_mcu/check_mcu.py [--accept] [--profile NAME ...]
                                [--tolerance 0.03] [--strict]
                                [--output DIR] [--no-smoke]
"""
import argparse
import concurrent.futures
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
BAREMETAL = HERE / "baremetal"
BASELINE = HERE / "accepted.json"

DOUBLE_SYM = re.compile(r"\b__aeabi_(d[a-z0-9]+|[a-z0-9]+2d)\b")
BENCH_LINE = re.compile(r"^([A-Za-z0-9_]+): (\d+) instructions, stack (\d+) bytes$")

PROFILES = {
    "cortex-m7-O2": {
        "arch": ["-mcpu=cortex-m7", "-mthumb", "-mfloat-abi=hard", "-mfpu=fpv5-d16"],
        "opt": ["-O2"],
        "qemu_machine": "mps2-an500",
        "float_only": False,
    },
    "cortex-m7-Os-U0": {
        "arch": ["-mcpu=cortex-m7", "-mthumb", "-mfloat-abi=hard", "-mfpu=fpv5-d16"],
        "opt": ["-Os", "-DEIGEN_UNROLLING_LIMIT=0"],
        "qemu_machine": "mps2-an500",
        "float_only": False,
    },
    "cortex-m4f-O2": {
        "arch": ["-mcpu=cortex-m4", "-mthumb", "-mfloat-abi=hard", "-mfpu=fpv4-sp-d16"],
        "opt": ["-O2"],
        "qemu_machine": "mps2-an386",
        "float_only": True,
    },
    "cortex-m4f-Os-U0": {
        "arch": ["-mcpu=cortex-m4", "-mthumb", "-mfloat-abi=hard", "-mfpu=fpv4-sp-d16"],
        "opt": ["-Os", "-DEIGEN_UNROLLING_LIMIT=0"],
        "qemu_machine": "mps2-an386",
        "float_only": True,
    },
}

SMOKE_TARGETS = {
    "cortex-m33-Os-U0": {
        "arch": ["-mcpu=cortex-m33", "-mthumb", "-mfloat-abi=hard", "-mfpu=fpv5-sp-d16"],
        "opt": ["-Os", "-DEIGEN_UNROLLING_LIMIT=0"],
        "float_only": True,
    },
    "cortex-m3-Os-U0": {
        "arch": ["-mcpu=cortex-m3", "-mthumb", "-mfloat-abi=soft"],
        "opt": ["-Os", "-DEIGEN_UNROLLING_LIMIT=0"],
        "float_only": True,
    },
}


def run_cmd(cmd, **kwargs):
    res = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if res.returncode != 0:
        sys.stderr.write(f"Command failed ({res.returncode}): {' '.join(map(str, cmd))}\n")
        if res.stdout:
            sys.stderr.write(res.stdout)
        if res.stderr:
            sys.stderr.write(res.stderr)
        raise RuntimeError(f"Command failed: {cmd[0]}")
    return res


def compiler_version():
    first = run_cmd(["arm-none-eabi-gcc", "--version"]).stdout.splitlines()[0]
    return first.split(")")[-1].strip().split()[0]


def text_size(elf):
    lines = run_cmd(["arm-none-eabi-size", str(elf)]).stdout.strip().splitlines()
    return int(lines[1].split()[0])


def check_no_fp64_symbols(obj_path):
    out = run_cmd(["arm-none-eabi-nm", "-u", str(obj_path)]).stdout
    found = sorted(set(DOUBLE_SYM.findall(out)))
    return ["__aeabi_" + s for s in found]


def build_and_run_qemu(out_dir, profile_name, cfg):
    pdir = out_dir / profile_name
    pdir.mkdir(parents=True, exist_ok=True)
    arch = cfg["arch"]
    opt = cfg["opt"]

    common_c = (
        arch
        + opt
        + ["-ffunction-sections", "-fdata-sections", "-fno-math-errno", "-std=c11", "-Wall", "-Wextra", "-Werror"]
    )
    startup_o = pdir / "startup.o"
    no_heap_o = pdir / "no_heap.o"
    run_cmd(["arm-none-eabi-gcc"] + common_c + ["-c", str(BAREMETAL / "startup.c"), "-o", str(startup_o)])
    run_cmd(["arm-none-eabi-gcc"] + common_c + ["-c", str(BAREMETAL / "no_heap.c"), "-o", str(no_heap_o)])

    cxx_base = (
        arch
        + opt
        + [
            "-std=c++17",
            "-fno-exceptions",
            "-fno-rtti",
            "-fno-threadsafe-statics",
            "-fno-use-cxa-atexit",
            "-ffunction-sections",
            "-fdata-sections",
            "-fno-math-errno",
            "-DNDEBUG",
            "-DEIGEN_NO_DEBUG",
            "-DEIGEN_NO_MALLOC",
            "-isystem",
            str(ROOT),
            "-Wall",
            "-Wextra",
            "-Wshadow",
            "-Wconversion",
            "-Wdouble-promotion",
            "-Werror",
            "-Wno-psabi",
        ]
    )
    link_base = (
        arch
        + opt
        + [
            "-nostartfiles",
            "-Wl,--no-warn-rwx-segments",
            "-T",
            str(BAREMETAL / "link.ld"),
            "-Wl,--gc-sections",
            "-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free",
            "--specs=nosys.specs",
        ]
    )

    # 1. Correctness test binary
    test_defs = ["-DTEST_MCU_BAREMETAL"]
    if cfg["float_only"]:
        test_defs.append("-DTEST_FLOAT_ONLY")
    test_o = pdir / "test_mcu_kernels.o"
    test_elf = pdir / "test_mcu_kernels.elf"
    run_cmd(["arm-none-eabi-g++"] + cxx_base + test_defs + ["-c", str(HERE / "test_mcu_kernels.cpp"), "-o", str(test_o)])
    if cfg["float_only"]:
        fp64 = check_no_fp64_symbols(test_o)
        if fp64:
            raise RuntimeError(f"{profile_name} test references FP64 routines: {', '.join(fp64)}")
    run_cmd(["arm-none-eabi-g++"] + link_base + [str(startup_o), str(no_heap_o), str(test_o), "-lm", "-o", str(test_elf)])

    qemu_cmd = [
        "qemu-system-arm",
        "-M",
        cfg["qemu_machine"],
        "-icount",
        "shift=0",
        "-semihosting",
        "-nographic",
        "-monitor",
        "none",
        "-serial",
        "none",
        "-kernel",
        str(test_elf),
    ]
    res = subprocess.run(qemu_cmd, capture_output=True, text=True, timeout=120)
    combined = res.stdout + res.stderr
    if res.returncode != 0 or "ALL PASSED" not in combined or "FAULT" in combined or "HEAP_TRAP" in combined:
        raise RuntimeError(f"{profile_name} correctness test failed (rc={res.returncode}):\n{combined}")

    # 2. Benchmark binary
    bench_defs = ["-DBENCH_FLOAT_ONLY"] if cfg["float_only"] else []
    bench_o = pdir / "bench_mcu_kernels.o"
    bench_elf = pdir / "bench_mcu_kernels.elf"
    run_cmd(["arm-none-eabi-g++"] + cxx_base + bench_defs + ["-c", str(HERE / "bench_mcu_kernels.cpp"), "-o", str(bench_o)])
    if cfg["float_only"]:
        fp64 = check_no_fp64_symbols(bench_o)
        if fp64:
            raise RuntimeError(f"{profile_name} benchmark references FP64 routines: {', '.join(fp64)}")
    run_cmd(["arm-none-eabi-g++"] + link_base + [str(startup_o), str(no_heap_o), str(bench_o), "-lm", "-o", str(bench_elf)])

    qemu_bench = qemu_cmd[:-1] + [str(bench_elf)]
    res = subprocess.run(qemu_bench, capture_output=True, text=True, timeout=180)
    combined = res.stdout + res.stderr
    if res.returncode != 0 or "BENCH_OK" not in combined or "FAULT" in combined or "HEAP_TRAP" in combined:
        raise RuntimeError(f"{profile_name} benchmark failed (rc={res.returncode}):\n{combined}")

    instructions = {}
    stacks = {}
    for line in combined.splitlines():
        m = BENCH_LINE.match(line.strip())
        if m:
            insns = int(m.group(2))
            stk = int(m.group(3))
            if insns <= 0:
                raise RuntimeError(f"{profile_name} kernel {m.group(1)} reported {insns} instructions:\n{combined}")
            instructions[m.group(1)] = insns
            stacks[m.group(1)] = stk
    if not instructions:
        raise RuntimeError(f"{profile_name} produced no benchmark measurements:\n{combined}")

    return {
        "test_text_bytes": text_size(test_elf),
        "bench_text_bytes": text_size(bench_elf),
        "instructions": instructions,
        "stack_bytes": stacks,
    }


def run_smoke_compile(out_dir, name, cfg):
    pdir = out_dir / name
    pdir.mkdir(parents=True, exist_ok=True)
    defs = ["-DTEST_MCU_BAREMETAL"]
    if cfg["float_only"]:
        defs.append("-DTEST_FLOAT_ONLY")
    obj = pdir / "test_mcu_kernels.o"
    cmd = (
        ["arm-none-eabi-g++"]
        + cfg["arch"]
        + cfg["opt"]
        + defs
        + [
            "-std=c++17",
            "-fno-exceptions",
            "-fno-rtti",
            "-fno-threadsafe-statics",
            "-ffunction-sections",
            "-fdata-sections",
            "-fno-math-errno",
            "-DNDEBUG",
            "-DEIGEN_NO_DEBUG",
            "-DEIGEN_NO_MALLOC",
            "-isystem",
            str(ROOT),
            "-Wall",
            "-Wextra",
            "-Wshadow",
            "-Wconversion",
            "-Wdouble-promotion",
            "-Werror",
            "-Wno-psabi",
            "-c",
            str(HERE / "test_mcu_kernels.cpp"),
            "-o",
            str(obj),
        ]
    )
    run_cmd(cmd)
    if cfg["float_only"]:
        fp64 = check_no_fp64_symbols(obj)
        if fp64:
            raise RuntimeError(f"{name} smoke build references FP64 routines: {', '.join(fp64)}")
    print(f"ok   smoke compile {name} (zero FP64 symbols)")


def compare_profile(profile_name, measured, expected, insn_tol, size_tol, stack_tol, strict):
    failures = []
    rows = []

    def check_metric(label, got, base, tol, abs_floor):
        if base <= 0:
            delta_pct = 0.0
        else:
            delta_pct = (got - base) / base
        regressed = (got - base) > abs_floor and delta_pct > tol
        shifted = strict and abs(got - base) > abs_floor and abs(delta_pct) > tol
        status = "FAIL" if (regressed or shifted) else ("FAST" if delta_pct < -tol and (base - got) > abs_floor else "ok")
        if status == "FAIL":
            failures.append(f"{profile_name} {label}: {got} vs {base} ({delta_pct:+.1%})")
        rows.append((status, label, got, base, delta_pct))

    check_metric("bench .text (B)", measured["bench_text_bytes"], expected["bench_text_bytes"], size_tol, 64)
    check_metric("test .text (B)", measured["test_text_bytes"], expected["test_text_bytes"], size_tol, 64)

    for k in sorted(expected["instructions"].keys()):
        if k not in measured["instructions"]:
            failures.append(f"{profile_name}: missing benchmark {k}")
            continue
        check_metric(f"{k} (insns)", measured["instructions"][k], expected["instructions"][k], insn_tol, 40)

    for k in sorted(expected["stack_bytes"].keys()):
        if k not in measured["stack_bytes"]:
            continue
        check_metric(f"{k} (stack B)", measured["stack_bytes"][k], expected["stack_bytes"][k], stack_tol, 32)

    return failures, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--accept", action="store_true", help="Record measured results into test_mcu/accepted.json")
    parser.add_argument("--profile", action="append", choices=list(PROFILES.keys()), help="Profiles to run (default: all)")
    parser.add_argument("--tolerance", type=float, default=0.03, help="Instruction regression tolerance (default: 0.03)")
    parser.add_argument("--size-tolerance", type=float, default=0.05, help=".text size regression tolerance (default: 0.05)")
    parser.add_argument("--stack-tolerance", type=float, default=0.05, help="Stack regression tolerance (default: 0.05)")
    parser.add_argument("--strict", action="store_true", help="Fail on both improvements and regressions outside tolerance")
    parser.add_argument("--output", default=None, help="Directory for build artifacts (default: temporary directory)")
    parser.add_argument("--jobs", "-j", type=int, default=1, help="Number of profiles to build/run concurrently")
    parser.add_argument("--no-smoke", action="store_true", help="Skip compile-only M33/M3 smoke builds")
    args = parser.parse_args()

    selected = args.profile or list(PROFILES.keys())
    compiler = compiler_version()
    print(f"arm-none-eabi-gcc {compiler}")

    tmp_ctx = tempfile.TemporaryDirectory(prefix="eigen_mcu_check_") if args.output is None else None
    out_dir = pathlib.Path(args.output) if args.output is not None else pathlib.Path(tmp_ctx.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        measured_all = {}
        workers = max(1, min(args.jobs, len(selected)))
        if workers == 1:
            for pname in selected:
                print(f"\n=== Running {pname} ===")
                measured_all[pname] = build_and_run_qemu(out_dir, pname, PROFILES[pname])
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futs = {pool.submit(build_and_run_qemu, out_dir, pname, PROFILES[pname]): pname for pname in selected}
                for fut in concurrent.futures.as_completed(futs):
                    pname = futs[fut]
                    measured_all[pname] = fut.result()

        for pname in selected:
            res = measured_all[pname]
            fp_note = ", zero FP64 symbols" if PROFILES[pname]["float_only"] else ""
            print(
                f"PASS {pname}: test ALL PASSED (.text={res['test_text_bytes']} B), "
                f"bench BENCH_OK (.text={res['bench_text_bytes']} B, {len(res['instructions'])} kernels){fp_note}"
            )

        if not args.no_smoke:
            print("\n=== Compile-only Cortex-M family smoke checks ===")
            for sname, scfg in SMOKE_TARGETS.items():
                run_smoke_compile(out_dir, sname, scfg)

        baselines = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
        if args.accept:
            comp_entry = baselines.setdefault(compiler, {})
            for pname, res in measured_all.items():
                comp_entry[pname] = res
            BASELINE.write_text(json.dumps(baselines, indent=2, sort_keys=True) + "\n")
            print(f"\nAccepted baselines for arm-none-eabi-gcc {compiler} in {BASELINE}")
            return 0

        comp_base = baselines.get(compiler)
        if comp_base is None:
            print(f"\nNOTICE: No accepted baseline for arm-none-eabi-gcc {compiler}; skipping regression comparison.")
            return 0

        all_failures = []
        summary_lines = [f"## Cortex-M Performance Gate (`arm-none-eabi-gcc {compiler}`)\n"]
        for pname in selected:
            if pname not in comp_base:
                print(f"NOTICE: No accepted baseline for {pname} under {compiler}; skipping.")
                continue
            failures, rows = compare_profile(
                pname,
                measured_all[pname],
                comp_base[pname],
                args.tolerance,
                args.size_tolerance,
                args.stack_tolerance,
                args.strict,
            )
            all_failures.extend(failures)
            print(f"\n--- Baseline comparison: {pname} ---")
            summary_lines.append(f"### `{pname}`\n")
            summary_lines.append("| Status | Metric | Measured | Baseline | Delta |")
            summary_lines.append("| --- | --- | ---: | ---: | ---: |")
            for status, label, got, base, delta_pct in rows:
                if status != "ok" or "(insns)" in label or ".text" in label:
                    print(f"{status:4s} {label:28s}: {got:8d} vs {base:8d} ({delta_pct:+6.1%})")
                    summary_lines.append(f"| {status} | `{label}` | {got:,} | {base:,} | {delta_pct:+.1%} |")
            summary_lines.append("")

        step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if step_summary:
            with open(step_summary, "a", encoding="utf-8") as fh:
                fh.write("\n".join(summary_lines) + "\n")

        if all_failures:
            print("\nPERFORMANCE REGRESSIONS DETECTED:")
            for f in all_failures:
                print("  FAIL:", f)
            return 1

        print("\nAll Cortex-M correctness, zero-heap, zero-FP64, and performance gates passed.")
        return 0
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


if __name__ == "__main__":
    sys.exit(main())

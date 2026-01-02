#!/usr/bin/env python
"""
Run all tests and generate a comprehensive report.

Usage:
    python tests/RUN_ALL_TESTS.py           # Run all tests
    python tests/RUN_ALL_TESTS.py --quick   # Run without coverage
    python tests/RUN_ALL_TESTS.py --html    # Generate HTML coverage report
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def run_tests(
    quick: bool = False,
    html_report: bool = False,
    verbose: bool = True,
) -> int:
    """
    Run all tests with pytest and generate a report.

    Args:
        quick: Skip coverage collection for faster runs
        html_report: Generate HTML coverage report
        verbose: Show verbose output

    Returns:
        Exit code (0 = success, non-zero = failure)
    """
    print("=" * 70)
    print("INTERTEMPORAL PREFERENCE RESEARCH - TEST SUITE")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project root: {PROJECT_ROOT}")
    print()

    # Build pytest command
    cmd = ["uv", "run", "pytest"]

    if verbose:
        cmd.append("-v")

    if not quick:
        cmd.extend([
            "--cov=src",
            "--cov=scripts",
            "--cov-report=term-missing",
        ])
        if html_report:
            cmd.extend(["--cov-report=html:tests/coverage_html"])

    # Add test directory
    cmd.append("tests/")

    print(f"Command: {' '.join(cmd)}")
    print("-" * 70)
    print()

    start_time = time.time()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.time() - start_time

    print()
    print("-" * 70)
    print(f"Completed in {elapsed:.2f} seconds")

    if result.returncode == 0:
        print("\n[PASS] All tests passed!")
    else:
        print(f"\n[FAIL] Tests failed with exit code {result.returncode}")

    if html_report and not quick:
        print(f"\nHTML coverage report: {PROJECT_ROOT / 'tests' / 'coverage_html' / 'index.html'}")

    print("=" * 70)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run all tests with coverage report")
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Skip coverage for faster runs"
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML coverage report"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Less verbose output"
    )

    args = parser.parse_args()

    exit_code = run_tests(
        quick=args.quick,
        html_report=args.html,
        verbose=not args.quiet,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

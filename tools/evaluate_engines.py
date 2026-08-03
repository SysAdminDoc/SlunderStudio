"""Run the fixed Slunder Studio engine evaluation cases."""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.evaluation import (
    DEFAULT_EVALUATION_CASES,
    EvaluationOutput,
    run_evaluation,
    write_evaluation_report,
)


def _load_runner(spec: str):
    module_name, separator, function_name = spec.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("Runner must use module:function syntax")
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"Runner is not callable: {spec}")
    return function


def _skipped_runner(case, _case_dir):
    return EvaluationOutput(
        status="skipped",
        failure="No runner configured; supply --runner module:function for real engine execution.",
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("evaluation-report.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("evaluation-artifacts"))
    parser.add_argument("--runner", help="Callable using module:function syntax")
    parser.add_argument("--case", action="append", dest="case_ids", help="Run only this fixed case ID (repeatable)")
    args = parser.parse_args(argv)

    cases = DEFAULT_EVALUATION_CASES
    if args.case_ids:
        requested = set(args.case_ids)
        cases = tuple(case for case in cases if case.case_id in requested)
        unknown = requested - {case.case_id for case in cases}
        if unknown:
            parser.error(f"Unknown fixed case(s): {', '.join(sorted(unknown))}")
    runner = _load_runner(args.runner) if args.runner else _skipped_runner
    report = run_evaluation(runner, cases=cases, artifact_dir=args.artifact_dir)
    write_evaluation_report(report, args.output)
    completed = sum(item["status"] == "completed" for item in report["cases"])
    failed = sum(item["status"] == "failed" for item in report["cases"])
    skipped = sum(item["status"] == "skipped" for item in report["cases"])
    print(f"Evaluation report: {args.output} (completed={completed}, failed={failed}, skipped={skipped})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

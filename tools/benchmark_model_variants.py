"""Benchmark an installed GGUF lyrics variant without opening the GUI.

Example:
    py -3.12 tools/benchmark_model_variants.py --model-id llama-3.1-8b-q8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Running ``python tools/script.py`` puts ``tools`` first on sys.path.  Make the
# repository-root package imports work without requiring PYTHONPATH setup.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.model_manager import ModelManager
from core.model_variants import (
    DEFAULT_VARIANT_BENCHMARK_CASES,
    measure_variant,
    write_variant_measurement,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure a downloaded GGUF variant's quality proxy, throughput, disk, RAM, and VRAM."
    )
    parser.add_argument("--model-id", required=True, help="Registered quantized lyrics model ID")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON path for the measurement record",
    )
    parser.add_argument(
        "--case-limit",
        type=int,
        default=len(DEFAULT_VARIANT_BENCHMARK_CASES),
        help="Number of fixed prompts to run (default: all)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.case_limit < 1 or args.case_limit > len(DEFAULT_VARIANT_BENCHMARK_CASES):
        print("error: case-limit must select at least one available fixed case", file=sys.stderr)
        return 2

    manager = ModelManager()
    info = manager.get_model_info(args.model_id)
    if info is None or not info.quantization:
        print(f"error: {args.model_id!r} is not a registered quantized model", file=sys.stderr)
        return 2

    from engines.lyrics_engine import LyricsLLM, _find_gguf_file

    model_path = _find_gguf_file(args.model_id)
    if not model_path:
        print(
            f"error: {args.model_id} is not downloaded; download it from Model Hub first",
            file=sys.stderr,
        )
        return 2

    llm = LyricsLLM()
    try:
        llm.load(model_id=args.model_id)

        def runner(case):
            text = llm.generate(
                system_prompt="Write clean, structured lyrics for the requested prompt.",
                user_prompt=case.prompt,
                max_tokens=case.max_tokens,
            )
            return {"text": text, "token_count": len(text.split())}

        measurement = measure_variant(
            args.model_id,
            info.quantization,
            model_path,
            runner,
            cases=DEFAULT_VARIANT_BENCHMARK_CASES[:args.case_limit],
            quality_metric="structural_adherence",
        )
    finally:
        llm.unload()

    if args.output:
        write_variant_measurement(measurement, args.output.expanduser())
    print(json.dumps(measurement.to_dict(), ensure_ascii=False, sort_keys=True))
    return 0 if measurement.complete else 1


if __name__ == "__main__":
    raise SystemExit(main())

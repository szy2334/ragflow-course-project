"""Run MinerU -> Baidu OCR -> chunk build -> RAGFlow import."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from pipeline_common import (
    config_bool,
    config_value,
    load_user_paper_config,
    safe_config_summary,
    sanitize_name,
    utc_now,
    write_json,
)


HERE = Path(__file__).resolve().parent


def run_stage(
    name: str,
    command: Sequence[str],
    *,
    accepted_codes: set[int] | None = None,
) -> int:
    accepted = accepted_codes or {0}
    print(f"\n=== {name} ===", flush=True)
    completed = subprocess.run(list(command), check=False)
    if completed.returncode not in accepted:
        raise RuntimeError(f"Stage {name} failed with exit code {completed.returncode}")
    return completed.returncode


def parse_args() -> argparse.Namespace:
    load_user_paper_config()
    parser = argparse.ArgumentParser()
    pdf_default = config_value("USER_PAPER_PDF")
    output_root_default = config_value("USER_PAPER_OUTPUT_ROOT", "data/user_paper_runs")
    parser.add_argument("--pdf", type=Path, default=Path(pdf_default) if pdf_default else None)
    parser.add_argument("--output-root", type=Path, default=Path(output_root_default))
    parser.add_argument("--run-name", default=config_value("USER_PAPER_RUN_NAME"))
    parser.add_argument("--mineru-dir", type=Path)
    fallback_mineru_dir = config_value("USER_PAPER_FALLBACK_MINERU_DIR")
    parser.add_argument(
        "--fallback-mineru-dir",
        type=Path,
        default=Path(fallback_mineru_dir) if fallback_mineru_dir else None,
    )
    parser.add_argument("--skip-baidu", action="store_true")
    parser.add_argument("--skip-vl", action="store_true")
    parser.add_argument(
        "--strict-specialized",
        action=argparse.BooleanOptionalAction,
        default=config_bool("USER_PAPER_STRICT_SPECIALIZED", True),
    )
    parser.add_argument(
        "--force-ocr",
        action=argparse.BooleanOptionalAction,
        default=config_bool("USER_PAPER_FORCE_OCR", True),
    )
    # User-paper chunks are the application's local PostgreSQL reading index.
    # Keep the historical importer available only as an explicit opt-in for
    # maintenance or non-user corpora; it must never be the default upload
    # path for a user's paper.
    parser.add_argument(
        "--import-ragflow",
        action="store_true",
        default=config_bool("USER_PAPER_IMPORT_RAGFLOW", False),
        help="Explicitly import chunks into RAGFlow (not used for user-paper reading).",
    )
    parser.add_argument(
        "--skip-ragflow",
        action="store_true",
        help="Compatibility switch; overrides --import-ragflow.",
    )
    parser.add_argument(
        "--replace-document",
        action="store_true",
        help="Replace an existing RAGFlow document for this paper instead of appending chunks.",
    )
    parser.add_argument(
        "--ragflow-base-url",
        default=config_value("RAGFLOW_BASE_URL", "http://localhost:9380/api/v1"),
    )
    parser.add_argument("--dataset-id", default=config_value("USER_PAPER_DATASET_ID"))
    parser.add_argument(
        "--dataset-name",
        default=config_value("USER_PAPER_DATASET_NAME", "user_papers_private_v1"),
    )
    parser.add_argument("--user-id", default=config_value("USER_PAPER_USER_ID", "local-user"))
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Show effective non-secret defaults and credential presence, then exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.print_config:
        print(
            json.dumps(
                {
                    "pdf": str(args.pdf) if args.pdf else None,
                    "output_root": str(args.output_root),
                    "run_name": args.run_name,
                    "fallback_mineru_dir": str(args.fallback_mineru_dir)
                    if args.fallback_mineru_dir
                    else None,
                    "strict_specialized": args.strict_specialized,
                    "force_ocr": args.force_ocr,
                    "ragflow_base_url": args.ragflow_base_url,
                    "dataset_id": args.dataset_id,
                    "dataset_name": args.dataset_name,
                    "user_id": args.user_id,
                    "credentials": safe_config_summary(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.pdf is None:
        raise SystemExit(
            "PDF path is missing. Set USER_PAPER_PDF in data_pipeline/user_paper/.env "
            "or provide --pdf."
        )
    pdf = args.pdf.resolve()
    if not pdf.exists():
        raise SystemExit(f"PDF not found: {pdf}")
    run_name = args.run_name or sanitize_name(pdf.stem.lower())
    run_root = (args.output_root / run_name).resolve()
    mineru_clean_dir = run_root / "01_mineru_clean"
    ocr_dir = run_root / "02_baidu_ocr"
    chunks_dir = run_root / "03_chunks"
    ragflow_dir = run_root / "04_ragflow"
    run_root.mkdir(parents=True, exist_ok=True)

    mineru_command = [
        sys.executable,
        str(HERE / "mineru_clean.py"),
        "--pdf",
        str(pdf),
        "--output-dir",
        str(mineru_clean_dir),
    ]
    if args.mineru_dir:
        mineru_command.extend(["--mineru-dir", str(args.mineru_dir.resolve())])
    try:
        run_stage("MinerU acquisition and clean", mineru_command)
    except Exception:
        if not args.fallback_mineru_dir or args.mineru_dir:
            raise
        print("MinerU cloud acquisition failed; using the configured local fallback.")
        run_stage(
            "MinerU local fallback clean",
            [
                sys.executable,
                str(HERE / "mineru_clean.py"),
                "--pdf",
                str(pdf),
                "--output-dir",
                str(mineru_clean_dir),
                "--mineru-dir",
                str(args.fallback_mineru_dir.resolve()),
            ],
        )

    if not args.skip_baidu:
        ocr_command = [
            sys.executable,
            str(HERE / "baidu_ocr.py"),
            "--media-jsonl",
            str(mineru_clean_dir / "media_objects.jsonl"),
            "--output-dir",
            str(ocr_dir),
        ]
        if args.skip_vl:
            ocr_command.append("--skip-vl")
        if args.strict_specialized:
            ocr_command.extend(["--strict-specialized", "--fail-fast"])
        if args.force_ocr:
            ocr_command.append("--force")
        run_stage("Baidu OCR enrichment", ocr_command, accepted_codes={0, 2})
    else:
        ocr_dir.mkdir(parents=True, exist_ok=True)

    run_stage(
        "Second clean and structured chunks",
        [
            sys.executable,
            str(HERE / "second_clean.py"),
            "--mineru-clean-dir",
            str(mineru_clean_dir),
            "--ocr-dir",
            str(ocr_dir),
            "--output-dir",
            str(chunks_dir),
        ],
    )

    should_import_ragflow = args.import_ragflow and not args.skip_ragflow
    if should_import_ragflow:
        import_command = [
            sys.executable,
            str(HERE / "ragflow_import.py"),
            "--chunks-dir",
            str(chunks_dir),
            "--state-dir",
            str(ragflow_dir),
            "--base-url",
            args.ragflow_base_url,
            "--dataset-name",
            args.dataset_name,
            "--user-id",
            args.user_id,
        ]
        if args.dataset_id:
            import_command.extend(["--dataset-id", args.dataset_id])
        if args.replace_document:
            import_command.append("--replace-document")
        run_stage("RAGFlow manual import", import_command)

    summary = {
        "run_name": run_name,
        "pdf": str(pdf),
        "run_root": str(run_root),
        "mineru_clean_dir": str(mineru_clean_dir),
        "ocr_dir": str(ocr_dir),
        "chunks_dir": str(chunks_dir),
        "ragflow_dir": str(ragflow_dir) if should_import_ragflow else None,
        "completed_at": utc_now(),
    }
    write_json(run_root / "pipeline_summary.json", summary)
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

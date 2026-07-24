"""Synchronize ICML/NeurIPS format profiles from reviewed rule manifests."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.db.base import build_engine, build_session_factory
from app.db.models import FormatProfile

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROFILES = {
    "icml_2026": {
        "manifest": REPOSITORY_ROOT / "icml_2026_rule_manifest.json",
        "name": "ICML 2026 投稿格式",
        "version": "2026.1",
        "description": "ICML 2026 投稿与终稿 PDF 格式审查。",
        "venue_id": "icml",
        "dataset_id": "d6a59c2684a811f1bd3a97a1481915ff",
        "retrieval_query": "ICML 2026 atomic manuscript format rules",
        "allowed_modes": ["camera_ready", "initial_submission"],
    },
    "neurips_2020": {
        "manifest": REPOSITORY_ROOT / "neurips_2020_rule_manifest.json",
        "name": "NeurIPS 2020 投稿格式",
        "version": "2020.1",
        "description": "NeurIPS 2020 PDF 格式审查；当前规则清单使用通用模式。",
        "venue_id": "neurips",
        "dataset_id": "e3c8be1a84a811f1bd3a97a1481915ff",
        "retrieval_query": "NeurIPS 2020 atomic manuscript format rules",
        "allowed_modes": ["general"],
    },
}


def load_profile_manifest(profile_key: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = dict(PROFILES[profile_key])
    rules = json.loads(Path(config["manifest"]).read_text(encoding="utf-8"))
    if not isinstance(rules, list) or not rules:
        raise RuntimeError(f"{config['manifest']} must contain a non-empty JSON array")
    required = {
        "rule_id",
        "description",
        "submission_mode",
        "source_document_id",
        "rule_category",
        "applicable_unit_kinds",
        "evidence_selector",
        "supported_checks",
    }
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict) or required - set(rule):
            raise RuntimeError(
                f"{config['manifest']} rule {index} misses {sorted(required - set(rule))}"
            )
    shared_ids = {
        str(rule["source_document_id"])
        for rule in rules
        if str(rule.get("submission_mode")) == "shared"
    }
    if len(shared_ids) != 1:
        raise RuntimeError(f"{config['manifest']} must bind exactly one shared document")
    shared_id = next(iter(shared_ids))
    mode_mapping: dict[str, str] = {}
    for mode in config["allowed_modes"]:
        ids = {
            str(rule["source_document_id"])
            for rule in rules
            if str(rule.get("submission_mode")) == mode
        }
        if len(ids) > 1:
            raise RuntimeError(f"{config['manifest']} binds multiple documents for {mode}")
        # A shared-only manifest intentionally applies the same atomic rules to
        # every UI submission mode and therefore maps each mode to the shared document.
        mode_mapping[mode] = next(iter(ids), shared_id)
    config["shared_document_id"] = shared_id
    config["mode_mapping"] = mode_mapping
    return config, rules


async def synchronize(*, apply: bool) -> None:
    settings = get_settings()
    engine = build_engine(settings.database_url)
    sessions = build_session_factory(engine)
    try:
        async with sessions() as session:
            for profile_key in PROFILES:
                config, rules = load_profile_manifest(profile_key)
                profile = await session.scalar(
                    select(FormatProfile).where(
                        FormatProfile.profile_key == profile_key,
                        FormatProfile.version == config["version"],
                    )
                )
                if profile is None:
                    profile = FormatProfile(profile_key=profile_key, version=config["version"])
                    session.add(profile)
                    action = "create"
                else:
                    action = "update"
                profile.name = config["name"]
                profile.description = config["description"]
                profile.ragflow_dataset_id = config["dataset_id"]
                profile.retrieval_query = config["retrieval_query"]
                profile.venue_id = config["venue_id"]
                profile.allowed_submission_modes = config["allowed_modes"]
                profile.shared_document_id = config["shared_document_id"]
                profile.mode_document_mapping_json = config["mode_mapping"]
                profile.rules_json = rules
                profile.is_active = True
                print(
                    f"{action} {profile_key}: rules={len(rules)}, "
                    f"shared={config['shared_document_id']}, modes={config['mode_mapping']}"
                )
            if apply:
                await session.commit()
                print("Changes committed.")
            else:
                await session.rollback()
                print("Dry run only. Re-run with --apply to commit.")
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    asyncio.run(synchronize(apply=args.apply))


if __name__ == "__main__":
    main()

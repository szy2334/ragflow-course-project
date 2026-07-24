import json
from pathlib import Path

import pytest

from app.format_review.runner import (
    FORMAT_REVIEW_INPUT_TOKEN_LIMIT,
    _output_token_budget,
    format_review_input_tokens,
)
from app.format_review.venue_layout import (
    _first_author_key,
    _is_title_case,
    _is_url_dominant_reference_entry,
    _logical_reference_entries,
    _reference_reading_order,
    review_facts_from_fused,
)
from app.format_review.workflow import (
    _allocate_rules_to_units,
    _applicable_manifest,
    _explicit_rule_id,
    _facts_for_unit,
    _manifest_standard_evidences,
    _model_standard_evidence,
    _plan_review_units,
    _refine_units_for_context_budget,
    _rule_review_policy,
    _unit_context_payload,
)
from tools.sync_format_manifests import load_profile_manifest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("profile_key", "mode", "result_name", "expected_rules"),
    [
        ("icml_2026", "camera_ready", "li22n.fused_facts.json", 75),
        ("icml_2026", "initial_submission", "li22n.fused_facts.json", 63),
        ("neurips_2020", "general", "neurips2020_byol.fused_facts.json", 35),
    ],
)
def test_fused_result_replay_has_exact_manifest_coverage(
    profile_key: str, mode: str, result_name: str, expected_rules: int
):
    config, raw_manifest = load_profile_manifest(profile_key)
    payload = json.loads((ROOT / result_name).read_text(encoding="utf-8"))
    facts = review_facts_from_fused(payload)
    manifest = _applicable_manifest(raw_manifest, mode)
    documents = list(
        dict.fromkeys([config["shared_document_id"], config["mode_mapping"][mode]])
    )
    standards = _manifest_standard_evidences(manifest, documents)
    units, coverage = _allocate_rules_to_units(
        units=_plan_review_units(facts),
        manifest=manifest,
        facts=facts,
        standard_evidences=standards,
        submission_mode=mode,
        retrieval_coverage={"strategy": "manifest_exact"},
    )

    rule_ids = {item["rule_id"] for item in manifest}
    retrieved = {item["canonical_rule_id"] for item in standards}
    allocated = {rule_id for unit in units for rule_id in unit["allocated_rule_ids"]}
    not_applicable = set(coverage["not_applicable_rule_ids"])
    assert len(rule_ids) == expected_rules
    assert retrieved == rule_ids
    assert allocated | not_applicable == rule_ids
    assert coverage["missing_rule_ids"] == []
    assert coverage["unallocated_rule_ids"] == []
    assert {"front_matter", "abstract", "body_section", "figure_table", "reference", "global"} <= {
        unit["unit_kind"] for unit in units
    }
    global_unit = next(unit for unit in units if unit["unit_kind"] == "global")
    global_ids = set(global_unit["expected_rule_ids"])
    context = _unit_context_payload(
        unit=global_unit,
        facts=_facts_for_unit(global_unit, facts),
        standards=[item for item in standards if item["canonical_rule_id"] in global_ids],
        rules_by_id={item["rule_id"]: item for item in manifest},
        submission_mode=mode,
    )
    selected_roles = {str(item.get("role")) for item in context["paper_layout_facts"]}
    assert "derived_page_geometry" in selected_roles
    assert "derived_column_geometry" in selected_roles
    figure_unit = next(unit for unit in units if unit["unit_kind"] == "figure_table")
    figure_context = _unit_context_payload(
        unit=figure_unit,
        facts=_facts_for_unit(figure_unit, facts),
        standards=[
            item
            for item in standards
            if item["canonical_rule_id"] in set(figure_unit["expected_rule_ids"])
        ],
        rules_by_id={item["rule_id"]: item for item in manifest},
        submission_mode=mode,
    )
    figure_roles = {str(item.get("role")) for item in figure_context["paper_layout_facts"]}
    assert "derived_captions" in figure_roles
    if profile_key == "icml_2026":
        assert "derived_caption_geometry" in figure_roles
        inventory = next(
            item["measurements"]
            for item in figure_context["paper_layout_facts"]
            if item.get("role") == "derived_captions"
        )
        assert inventory["figures"]["numbers"] == [1, 2, 3, 4, 5, 6]
        assert inventory["figures"]["numbering_continuous"] is True
        assert inventory["figures"]["caption_pair_count"] == 6
        assert "items" not in inventory["figures"]
        geometry = next(
            item["measurements"]
            for item in figure_context["paper_layout_facts"]
            if item.get("role") == "derived_caption_geometry"
        )
        assert geometry["figures"]["single_line_centered_count"] == 2
        assert geometry["figures"]["multiline_left_aligned_count"] == 4
        assert geometry["figures"]["object_inside_body_count"] == 6
        assert geometry["tables"]["object_inside_body_count"] == 15
        front_unit = next(unit for unit in units if unit["unit_kind"] == "front_matter")
        front_context = _unit_context_payload(
            unit=front_unit,
            facts=_facts_for_unit(front_unit, facts),
            standards=[
                item
                for item in standards
                if item["canonical_rule_id"] in set(front_unit["expected_rule_ids"])
            ],
            rules_by_id={item["rule_id"]: item for item in manifest},
            submission_mode=mode,
        )
        front_geometry = next(
            item["measurements"]
            for item in front_context["paper_layout_facts"]
            if item.get("role") == "derived_front_matter_geometry"
        )
        assert front_geometry["author_center_offset_pt"] < 1
        assert front_geometry["horizontal_rules"][0]["width_pt"] == pytest.approx(0.996)
        front_source = next(
            item["measurements"]
            for item in front_context["paper_layout_facts"]
            if item.get("role") == "derived_front_matter"
        )
        assert front_source["schema"] == "icml_fused_layout"
        assert front_source["horizontal_rule_count"] == 2
        assert front_source["title_between_top_and_bottom_rules"] is True
        abstract_unit = next(unit for unit in units if unit["unit_kind"] == "abstract")
        abstract_context = _unit_context_payload(
            unit=abstract_unit,
            facts=_facts_for_unit(abstract_unit, facts),
            standards=[
                item
                for item in standards
                if item["canonical_rule_id"] in set(abstract_unit["expected_rule_ids"])
            ],
            rules_by_id={item["rule_id"]: item for item in manifest},
            submission_mode=mode,
        )
        abstract_source = next(
            item["measurements"]
            for item in abstract_context["paper_layout_facts"]
            if item.get("role") == "derived_abstract"
        )
        assert abstract_source["schema"] == "icml_fused_layout"
        assert abstract_source["paragraph_count"] == 1
        assert abstract_source["left_extra_indent_pt"] == pytest.approx(19.567)
        assert abstract_source["right_extra_indent_pt"] == pytest.approx(15.7416)
        assert abstract_source["gap_after_pt"] == pytest.approx(29.6631)
        body_unit = next(unit for unit in units if unit["unit_kind"] == "body_section")
        body_context = _unit_context_payload(
            unit=body_unit,
            facts=_facts_for_unit(body_unit, facts),
            standards=[
                item
                for item in standards
                if item["canonical_rule_id"] in set(body_unit["expected_rule_ids"])
            ],
            rules_by_id={item["rule_id"]: item for item in manifest},
            submission_mode=mode,
        )
        body_roles = {str(item.get("role")) for item in body_context["paper_layout_facts"]}
        assert {
            "derived_typography_inventory",
            "derived_heading_inventory",
            "derived_citation_inventory",
        } <= body_roles
        heading_inventory = next(
            item["measurements"]
            for item in body_context["paper_layout_facts"]
            if item.get("role") == "derived_heading_inventory"
        )
        assert heading_inventory["levels"]["level_2"]["count"] == 16
        assert heading_inventory["levels"]["level_1"]["overlap_count"] == 0
        appendix_unit = next(unit for unit in units if unit["unit_kind"] == "appendix")
        appendix_context = _unit_context_payload(
            unit=appendix_unit,
            facts=_facts_for_unit(appendix_unit, facts),
            standards=[
                item
                for item in standards
                if item["canonical_rule_id"] in set(appendix_unit["expected_rule_ids"])
            ],
            rules_by_id={item["rule_id"]: item for item in manifest},
            submission_mode=mode,
        )
        appendix_layout = next(
            item["measurements"]
            for item in appendix_context["paper_layout_facts"]
            if item.get("role") == "derived_appendix_layout"
        )
        assert appendix_layout["appendix_letters"] == ["A", "B", "C", "D"]
        reference_unit = next(unit for unit in units if unit["unit_kind"] == "reference")
        reference_context = _unit_context_payload(
            unit=reference_unit,
            facts=_facts_for_unit(reference_unit, facts),
            standards=[
                item
                for item in standards
                if item["canonical_rule_id"] in set(reference_unit["expected_rule_ids"])
            ],
            rules_by_id={item["rule_id"]: item for item in manifest},
            submission_mode=mode,
        )
        reference_inventory = next(
            item["measurements"]
            for item in reference_context["paper_layout_facts"]
            if item.get("role") == "derived_reference_inventory"
        )
        assert reference_inventory["coverage"]["entry_count"] == 56
        assert reference_inventory["all_entries_have_year"] is True
        assert reference_inventory["first_author_ordered"] is True
        assert reference_inventory["first_author_ordering_violations"] == []
        assert len(reference_inventory["first_author_keys_in_reading_order"]) == (
            reference_inventory["coverage"]["sortable_first_author_count"]
        )
        figure_font_rule = "icml-2026-shared-01-81403e269a914aeb"
        figure_font_standard = next(
            item
            for item in figure_context["standard_evidence"]
            if item["canonical_rule_id"] == figure_font_rule
        )
        assert "NimbusRomNo9L-Regu" in figure_font_standard["font_whitelist"]
    refined, _ = _refine_units_for_context_budget(
        units=units,
        facts=facts,
        standards=standards,
        rules_by_id={item["rule_id"]: item for item in manifest},
        submission_mode=mode,
    )
    for refined_unit in refined:
        refined_ids = set(refined_unit["expected_rule_ids"])
        if not refined_ids:
            continue
        refined_payload = _unit_context_payload(
            unit=refined_unit,
            facts=_facts_for_unit(refined_unit, facts),
            standards=[
                item for item in standards if item["canonical_rule_id"] in refined_ids
            ],
            rules_by_id={item["rule_id"]: item for item in manifest},
            submission_mode=mode,
        )
        assert format_review_input_tokens("format_check", refined_payload) <= 13_000
        assert refined_unit["context_refinement"]["token_limit"] == 13_000


def test_shared_only_manifest_maps_every_submission_mode_to_shared_document():
    config, rules = load_profile_manifest("neurips_2020")

    assert {item["submission_mode"] for item in rules} == {"shared"}
    assert set(config["mode_mapping"].values()) == {config["shared_document_id"]}


def test_format_check_output_budget_scales_with_atomic_rule_count():
    payload = {"review_unit": {"expected_rule_ids": [f"rule-{index}" for index in range(19)]}}

    assert _output_token_budget("format_check", payload) == 8192
    assert _output_token_budget("format_reflect", payload) == 1024
    assert FORMAT_REVIEW_INPUT_TOKEN_LIMIT == 13_000


def test_format_check_token_estimate_includes_json_object_schema_framing():
    payload = {
        "review_unit": {"expected_rule_ids": ["rule-1"]},
        "standard_evidence": [{"evidence_id": "S1", "quote": "A rule."}],
    }

    prompt_only = format_review_input_tokens(
        "format_check", payload, structured_mode="prompt_json"
    )
    json_object = format_review_input_tokens(
        "format_check", payload, structured_mode="json_object"
    )

    assert json_object > prompt_only


def test_reference_inventory_normalizes_wrapped_entries_and_url_fonts():
    entries = [
        {"text": "Chen, X. A paper. 2020.", "style": {"dominant_font": "NimbusRomNo9L-Regu"}},
        {
            "text": "R., Ramapuram, J. Continuation. 2020.",
            "style": {"dominant_font": "NimbusRomNo9L-Regu"},
        },
        {
            "text": "NIPS 2017 Crawford, K. A keynote. 2017. URL https://example.test",
            "style": {"dominant_font": "NimbusMonL-Regu"},
        },
        {
            "text": "https://www. Garvie, C., May 2019. URL flawedfacedata.com/.",
            "style": {"dominant_font": "NimbusMonL-Regu"},
        },
    ]

    logical = _logical_reference_entries(entries)

    assert len(logical) == 3
    assert "Continuation" in logical[0]["text"]
    assert _first_author_key(logical[1]["text"]) == "crawford"
    assert _first_author_key(logical[2]["text"]) == "garvie"
    assert _is_url_dominant_reference_entry(logical[1]) is True


def test_reference_reading_order_is_page_then_left_column_then_right_column():
    entries = [
        {"page_number": 10, "bbox": [307.0, 130.0, 540.0, 150.0], "text": "Delta"},
        {"page_number": 10, "bbox": [55.0, 170.0, 290.0, 190.0], "text": "Bravo"},
        {"page_number": 9, "bbox": [307.0, 700.0, 540.0, 720.0], "text": "Alpha"},
        {"page_number": 10, "bbox": [307.0, 70.0, 540.0, 90.0], "text": "Charlie"},
        {"page_number": 10, "bbox": [55.0, 80.0, 290.0, 100.0], "text": "Able"},
    ]

    ordered = _reference_reading_order(entries)

    assert [item["text"] for item in ordered] == [
        "Alpha",
        "Able",
        "Bravo",
        "Charlie",
        "Delta",
    ]


def test_title_case_accepts_pdf_ligatures():
    assert _is_title_case("Creating a Sufﬁciently Large Dataset") is True


def test_reference_sort_key_retains_standalone_diaeresis_collation():
    assert _first_author_key("K¨arkk¨ainen, K. and Joo, J. Fairface, 2019.") == (
        "k~arkk~ainen"
    )


def test_neurips_rule_assessment_context_is_compact_and_manifest_scoped():
    config, raw_manifest = load_profile_manifest("neurips_2020")
    payload = json.loads((ROOT / "neurips2020_byol.fused_facts.json").read_text(encoding="utf-8"))
    facts = review_facts_from_fused(payload)
    manifest = _applicable_manifest(raw_manifest, "camera_ready")
    standards = _manifest_standard_evidences(manifest, [config["shared_document_id"]])
    rules_by_id = {item["rule_id"]: item for item in manifest}
    units, _ = _allocate_rules_to_units(
        units=_plan_review_units(facts),
        manifest=manifest,
        facts=facts,
        standard_evidences=standards,
        submission_mode="camera_ready",
        retrieval_coverage={"strategy": "manifest_exact"},
    )

    assessment_facts = [item for item in facts if item.get("role") == "derived_rule_assessment"]
    assert len(assessment_facts) == 10
    assert all("rule_assessments" not in str(item.get("block_id")) for item in facts)

    front_unit = next(unit for unit in units if unit["unit_kind"] == "front_matter")
    context = _unit_context_payload(
        unit=front_unit,
        facts=_facts_for_unit(front_unit, facts),
        standards=[
            item
            for item in standards
            if item["canonical_rule_id"] in set(front_unit["expected_rule_ids"])
        ],
        rules_by_id=rules_by_id,
        submission_mode="camera_ready",
    )
    context_json = json.dumps(context["paper_layout_facts"], ensure_ascii=False)
    assessment_groups = {
        item["measurements"]["parser_rule_group"]
        for item in context["paper_layout_facts"]
        if item.get("role") == "derived_rule_assessment"
    }
    assert assessment_groups == {"NIPS-04"}
    assert "Bootstrap Your Own Latent" not in context_json
    assert all(
        item.get("role", "").startswith("derived_") for item in context["paper_layout_facts"]
    )
    front_assessment = next(
        item["measurements"]
        for item in context["paper_layout_facts"]
        if item.get("role") == "derived_rule_assessment"
    )
    assert front_assessment["evidence"]["title"]["title_between_top_and_bottom_rules"] is True

    font_rule = "neurips-2020-shared-02-9dc4e64e3cb08908"
    font_standard = _model_standard_evidence(
        next(item for item in standards if item["canonical_rule_id"] == font_rule)
    )
    assert "NimbusRomNo9L*" in font_standard["font_whitelist"]
    typography_inventory = next(
        item["measurements"]
        for item in facts
        if item.get("role") == "derived_typography_inventory"
    )
    assert typography_inventory["main_body"]["styles"]["font_names"] == [
        "NimbusRomNo9L-Regu"
    ]
    assert typography_inventory["coverage"]["main_body_excluded_formula_like_count"] >= 1

    reference_unit = next(unit for unit in units if unit["unit_kind"] == "reference")
    reference_context = _unit_context_payload(
        unit=reference_unit,
        facts=_facts_for_unit(reference_unit, facts),
        standards=[
            item
            for item in standards
            if item["canonical_rule_id"] in set(reference_unit["expected_rule_ids"])
        ],
        rules_by_id=rules_by_id,
        submission_mode="camera_ready",
    )
    reference_assessment = next(
        item["measurements"]
        for item in reference_context["paper_layout_facts"]
        if item.get("role") == "derived_rule_assessment"
    )
    assert reference_assessment["evidence"]["references_heading"]["is_unnumbered"] is True


def test_atomic_chunk_embedded_rule_id_wins_over_ambiguous_prose():
    rule_id = "icml-2026-shared-07-ba175c511470365b"
    content = f"规则ID：{rule_id}\n章节：7. 附录\n附录正文字体必须属于字体白名单。"

    assert _explicit_rule_id(content, {rule_id, "another-rule"}) == rule_id


def test_rule_policies_allow_auditable_sufficiency_sampling_and_tolerance():
    sufficiency = _rule_review_policy(
        {"rule_category": "page_layout", "is_global": True, "supported_checks": ["page_size"]}
    )
    inventory_backed = _rule_review_policy(
        {"rule_category": "heading", "supported_checks": ["heading_style"]}
    )
    sampled = _rule_review_policy(
        {
            "rule_category": "abstract",
            "assessment_mode": "sampled",
            "supported_checks": ["heading_style"],
        }
    )
    tolerance = _rule_review_policy(
        {"rule_category": "abstract", "supported_checks": ["font_size"]}
    )

    assert sufficiency["mode"] == "sufficiency"
    assert sufficiency["coverage_scope"] == "derived_aggregate"
    assert inventory_backed["mode"] == "sufficiency"
    assert inventory_backed["coverage_scope"] == "derived_aggregate"
    assert sampled["mode"] == "sampled"
    assert sampled["coverage_scope"] == "representative_sample"
    assert tolerance["mode"] == "tolerance_aware"
    assert tolerance["coverage_scope"] == "localized"
    assert _rule_review_policy({"rule_category": "figure"})["mode"] == "complete_inventory"

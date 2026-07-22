from app.format_review.pdf_layout import bbox_iou, extract_native_pdf_layout, normalize_font_name
from app.format_review.validators import validate_findings
from app.format_review.workflow import (
    MAX_CONTEXT_CHARACTERS,
    _allocate_rules_to_units,
    _applicable_manifest,
    _context_facts_for_category,
    _context_groups,
    _pdf_span_record_values,
    _plan_review_units,
    _refine_units_for_context_budget,
)


def test_non_compliance_without_page_bbox_is_downgraded_to_unverifiable():
    findings = validate_findings(
        [
            {
                "category": "heading",
                "aspect": "一级标题字号",
                "result": "non_compliant",
                "finding": "字号不符合规范。",
                "paper_evidence_ids": ["P1"],
                "standard_evidence_ids": ["S1"],
            }
        ],
        paper_evidences=[{"evidence_id": "P1", "page_number": 2, "bbox": None}],
        standard_evidences=[{"evidence_id": "S1", "quote": "一级标题使用 12pt。"}],
        coverage_report={"missing_categories": []},
    )

    assert findings[0]["result"] == "unverifiable"
    assert findings[0]["evidence_status"] == "incomplete"


def test_missing_rule_coverage_cannot_return_compliant():
    findings = validate_findings(
        [
            {
                "category": "reference",
                "aspect": "参考文献格式",
                "result": "compliant",
                "finding": "符合。",
                "paper_evidence_ids": ["P1"],
                "standard_evidence_ids": ["S1"],
            }
        ],
        paper_evidences=[{"evidence_id": "P1", "page_number": 8, "bbox": [1, 2, 3, 4]}],
        standard_evidences=[{"evidence_id": "S1", "quote": "参考文献要求。"}],
        coverage_report={"missing_categories": ["reference"]},
    )

    assert findings[0]["result"] == "unverifiable"


def test_native_pdf_style_helpers_normalize_subset_fonts_and_measure_overlap():
    assert normalize_font_name("ABCDEE+TimesNewRomanPSMT") == "TimesNewRomanPSMT"
    assert bbox_iou([0, 0, 10, 10], [5, 0, 15, 10]) == 1 / 3


def test_inactive_rules_are_not_added_to_the_frozen_applicable_manifest():
    rules = [
        {
            "canonical_rule_id": "active-rule",
            "description": "Active format rule",
            "submission_mode": "shared",
            "status": "active",
        },
        {
            "canonical_rule_id": "retired-rule",
            "description": "Retired format rule",
            "submission_mode": "initial_submission",
            "status": "retired",
        },
        {
            "canonical_rule_id": "source-only-rule",
            "description": "Needs a LaTeX source file.",
            "submission_mode": "shared",
            "status": "disabled",
            "excluded_reason": "Source evidence is unavailable in PDF-only review.",
        },
    ]

    manifest = _applicable_manifest(rules, "initial_submission")

    assert [item["rule_id"] for item in manifest] == ["active-rule"]


def test_review_plan_groups_many_body_sections_into_three_semantic_units():
    facts = [
        {
            "evidence_id": f"P{index}",
            "block_id": f"b-{index}",
            "page_number": index,
            "quote": f"{index} Main section",
            "role": "body",
            "section_title": f"{index} Main section",
        }
        for index in range(1, 8)
    ]

    units = _plan_review_units(facts)

    body_units = [unit for unit in units if unit["unit_kind"] == "body_section"]
    assert len(body_units) == 3
    assert sum(len(unit["fact_ids"]) for unit in body_units) == len(facts)
    assert all(unit["page_range"][0] <= unit["page_range"][1] for unit in body_units)


def test_oversized_unit_is_split_at_a_page_boundary_before_llm_execution():
    facts = [
        {
            "evidence_id": "P1",
            "block_id": "p1",
            "page_number": 1,
            "quote": "1 Introduction " + "evidence " * 40,
            "role": "body",
            "section_title": "1 Introduction",
        },
        {
            "evidence_id": "P2",
            "block_id": "p2",
            "page_number": 2,
            "quote": "2 Method " + "evidence " * 40,
            "role": "body",
            "section_title": "2 Method",
        },
    ]
    unit = {
        "unit_id": "u-001",
        "unit_position": 0,
        "unit_kind": "body_section",
        "title": "Introduction - Method",
        "page_range": [1, 2],
        "block_ids": ["p1", "p2"],
        "fact_ids": ["P1", "P2"],
        "expected_rule_ids": ["heading-style"],
        "allocated_rule_ids": ["heading-style"],
        "global_rule_ids": [],
        "not_applicable_rule_ids": [],
        "retrieved_rule_ids": ["heading-style"],
        "coverage": {"complete": True},
    }
    rules_by_id = {
        "heading-style": {
            "rule_id": "heading-style",
            "rule_category": "heading",
            "title": "Heading style",
            "description": "Headings must be visible.",
        }
    }
    refined, metrics = _refine_units_for_context_budget(
        units=[unit],
        facts=facts,
        standards=[{"canonical_rule_id": "heading-style", "quote": "Heading rule"}],
        rules_by_id=rules_by_id,
        submission_mode="initial_submission",
        context_budget=0,
    )

    assert metrics["split_parent_count"] == 1
    assert [item["page_range"] for item in refined] == [[1, 1], [2, 2]]
    assert all(item["expected_rule_ids"] == ["heading-style"] for item in refined)


def test_native_pdf_extractor_preserves_span_geometry_and_style(tmp_path):
    import fitz

    path = tmp_path / "native.pdf"
    document = fitz.open()
    page = document.new_page(width=612, height=792)
    page.insert_text((72, 96), "Format Evidence", fontsize=14, fontname="helv")
    document.save(path)
    document.close()

    layout = extract_native_pdf_layout(str(path))

    assert layout.available is True
    assert layout.page_count == 1
    assert layout.spans[0]["text"] == "Format Evidence"
    assert layout.spans[0]["font_size_pt"] == 14.0
    assert len(layout.spans[0]["bbox"]) == 4

    values = _pdf_span_record_values(layout.spans[0])
    assert values["bbox_json"] == layout.spans[0]["bbox"]
    assert "bbox" not in values


def test_context_groups_select_category_relevant_facts_within_budget():
    facts = [
        {
            "evidence_id": f"P{index}",
            "page_number": index % 9 + 1,
            "bbox": [1, 2, 3, 4],
            "quote": "body text " * 120,
            "role": "native_text_span",
            "source": "native_pdf",
            "page_width_pt": 612.0,
        }
        for index in range(500)
    ]
    plan = {
        "venue_id": "neurips",
        "submission_mode": "initial_submission",
        "target_categories": ["page_layout"],
    }
    standards = [
        {"category": "page_layout", "evidence_id": "S1", "quote": "Use US Letter paper size."}
    ]

    groups = _context_groups(plan, standards, facts)

    assert len(groups) == 1
    category, payload = groups[0]
    assert category == "all"
    assert payload is not None
    assert len(payload["paper_layout_facts"]) < len(facts)
    assert len(str(payload)) < MAX_CONTEXT_CHARACTERS


def test_author_identity_context_keeps_author_and_affiliation_evidence():
    facts = [
        {
            "evidence_id": "P-title",
            "page_number": 1,
            "bbox": [1, 2, 3, 4],
            "quote": "A paper title",
            "role": "heading",
            "source": "native_pdf",
        },
        {
            "evidence_id": "P-author",
            "page_number": 1,
            "bbox": [1, 5, 3, 7],
            "quote": "Ada Lovelace ada@example.test",
            "role": "paragraph",
            "source": "native_pdf+mineru",
        },
        {
            "evidence_id": "P-affiliation",
            "page_number": 1,
            "bbox": [1, 8, 3, 10],
            "quote": "School of Computing, Example University",
            "role": "paragraph",
            "source": "native_pdf+mineru",
        },
    ]

    selected = _context_facts_for_category(facts, "author_identity", limit=8)

    assert {"P-author", "P-affiliation"} <= {item["evidence_id"] for item in selected}


def test_structured_scope_allocation_keeps_global_rules_single_and_records_not_applicable():
    facts = [
        {
            "evidence_id": "P1",
            "block_id": "p1-title",
            "page_number": 1,
            "page_width_pt": 612.0,
            "page_height_pt": 792.0,
            "quote": "Abstract\nA projection algorithm.",
            "role": "abstract",
            "section_title": "Abstract",
        },
        {
            "evidence_id": "P2",
            "block_id": "p2-body",
            "page_number": 2,
            "page_width_pt": 612.0,
            "page_height_pt": 792.0,
            "quote": "1 Introduction",
            "role": "body",
            "section_title": "Introduction",
        },
    ]
    manifest = [
        {
            "rule_id": "page-size",
            "title": "页面尺寸",
            "rule_category": "page",
            "applicable_unit_kinds": ["global"],
            "is_global": True,
            "requires_cross_unit": True,
            "cross_unit_kinds": ["global"],
            "applicability_conditions": {},
        },
        {
            "rule_id": "abstract-style",
            "title": "摘要格式",
            "rule_category": "abstract",
            "applicable_unit_kinds": ["abstract"],
            "is_global": False,
            "requires_cross_unit": False,
            "cross_unit_kinds": [],
            "applicability_conditions": {"requires_section_roles": ["abstract"]},
        },
        {
            "rule_id": "figure-caption",
            "title": "图题格式",
            "rule_category": "figure",
            "applicable_unit_kinds": ["figure_table"],
            "is_global": False,
            "requires_cross_unit": False,
            "cross_unit_kinds": [],
            "applicability_conditions": {"requires_object_types": ["figure"]},
        },
    ]
    standards = [
        {"canonical_rule_id": "page-size", "category": "page"},
        {"canonical_rule_id": "abstract-style", "category": "abstract"},
        {"canonical_rule_id": "figure-caption", "category": "figure"},
    ]

    units, coverage = _allocate_rules_to_units(
        units=_plan_review_units(facts),
        manifest=manifest,
        facts=facts,
        standard_evidences=standards,
        submission_mode="initial_submission",
        retrieval_coverage={},
    )

    global_units = [unit for unit in units if unit["unit_kind"] == "global"]
    assert len(global_units) == 1
    assert global_units[0]["global_rule_ids"] == ["page-size"]
    assert sum("page-size" in unit["allocated_rule_ids"] for unit in units) == 1
    assert global_units[0]["not_applicable_rule_ids"] == [
        {
            "rule_id": "figure-caption",
            "condition_evidence_ids": [],
            "reason": "论文不包含该规则要求的对象类型。",
        }
    ]
    assert coverage["missing_rule_ids"] == []
    assert coverage["unallocated_rule_ids"] == []

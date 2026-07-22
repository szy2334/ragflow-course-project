from app.format_review.pdf_layout import bbox_iou, extract_native_pdf_layout, normalize_font_name
from app.format_review.validators import validate_findings
from app.format_review.workflow import (
    MAX_CONTEXT_CHARACTERS,
    _allocate_rules_to_units,
    _appendix_pages,
    _applicable_manifest,
    _coarse_body_groups,
    _context_facts_for_category,
    _context_groups,
    _deduplicate_unit_findings,
    _has_complete_actionable_finding,
    _item_record,
    _model_unit_plan,
    _pdf_span_record_values,
    _plan_review_units,
    _refine_units_for_context_budget,
    _unverifiable_findings_for_rules,
)
from tools.backfill_format_rule_scopes import scope_for_v2
from tools.reconcile_format_profiles import _icml_scope, _reference_scope


def test_format_review_state_keeps_synthesized_findings():
    from app.format_review.schemas import FormatReviewState

    assert "final_findings" in FormatReviewState.__annotations__


def test_final_findings_are_consolidated_by_canonical_rule():
    units = [
        {
            "unit_id": "u-1",
            "unit_position": 1,
            "findings": [
                {
                    "rule_ids": ["heading-rule"],
                    "category": "heading",
                    "aspect": "Heading size",
                    "result": "unverifiable",
                    "severity": "info",
                    "finding": "One heading lacks evidence.",
                    "paper_evidences": [],
                    "standard_evidences": [
                        {"evidence_id": "S1", "canonical_rule_id": "heading-rule"}
                    ],
                    "evidence_status": "incomplete",
                }
            ],
        },
        {
            "unit_id": "u-2",
            "unit_position": 2,
            "findings": [
                {
                    "rule_ids": ["heading-rule"],
                    "category": "heading",
                    "aspect": "Heading size",
                    "result": "non_compliant",
                    "severity": "medium",
                    "finding": "Another heading is too small.",
                    "paper_evidences": [
                        {"evidence_id": "P2", "page_number": 2, "bbox": [1, 2, 3, 4]}
                    ],
                    "standard_evidences": [
                        {"evidence_id": "S1", "canonical_rule_id": "heading-rule"}
                    ],
                    "evidence_status": "complete",
                }
            ],
        },
    ]

    findings = _deduplicate_unit_findings(units)

    assert len(findings) == 1
    assert findings[0]["rule_ids"] == ["heading-rule"]
    assert findings[0]["result"] == "non_compliant"
    assert [item["evidence_id"] for item in findings[0]["paper_evidences"]] == ["P2"]


def test_complete_rule_conclusion_survives_another_rule_being_unverifiable():
    findings = [
        {"result": "unverifiable", "evidence_status": "incomplete"},
        {"result": "non_compliant", "evidence_status": "complete"},
    ]

    assert _has_complete_actionable_finding(findings) is True


def test_confirmed_compliance_is_not_downgraded_by_an_unrelated_missing_unit():
    units = [
        {
            "unit_id": "u-1",
            "unit_position": 1,
            "findings": [
                {
                    "rule_ids": ["organization"],
                    "category": "heading",
                    "aspect": "Organization",
                    "result": "unverifiable",
                    "severity": "info",
                    "finding": "One page is unavailable.",
                    "paper_evidences": [],
                    "standard_evidences": [
                        {"evidence_id": "S1", "canonical_rule_id": "organization"}
                    ],
                    "evidence_status": "incomplete",
                }
            ],
        },
        {
            "unit_id": "u-2",
            "unit_position": 2,
            "findings": [
                {
                    "rule_ids": ["organization"],
                    "category": "heading",
                    "aspect": "Organization",
                    "result": "compliant",
                    "severity": "info",
                    "finding": "The document has numbered sections and paragraphs.",
                    "paper_evidences": [
                        {"evidence_id": "P2", "page_number": 2, "bbox": [1, 2, 3, 4]}
                    ],
                    "standard_evidences": [
                        {"evidence_id": "S1", "canonical_rule_id": "organization"}
                    ],
                    "evidence_status": "complete",
                }
            ],
        },
    ]

    assert _deduplicate_unit_findings(units)[0]["result"] == "compliant"


def test_only_final_items_store_the_bare_canonical_rule_id():
    finding = {
        "rule_ids": ["canonical-heading-rule"],
        "category": "heading",
        "aspect": "Heading size",
        "result": "non_compliant",
        "severity": "medium",
        "finding": "Heading is too small.",
        "paper_evidences": [],
        "standard_evidences": [],
        "evidence_status": "incomplete",
    }

    first_unit = _item_record(
        "review-1", finding, unit_id="u-001", unit_position=1, source_stage="unit"
    )
    second_unit = _item_record(
        "review-1", finding, unit_id="u-002", unit_position=2, source_stage="unit"
    )
    final = _item_record("review-1", finding, source_stage="final")

    assert first_unit.rule_id.startswith("canonical-heading-rule:")
    assert second_unit.rule_id.startswith("canonical-heading-rule:")
    assert first_unit.rule_id != second_unit.rule_id
    assert final.rule_id == "canonical-heading-rule"


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


def test_sampled_conclusion_keeps_a_located_primary_anchor_with_unlocated_auxiliary_evidence():
    findings = validate_findings(
        [
            {
                "category": "heading",
                "aspect": "章节组织",
                "result": "compliant",
                "finding": "基于跨章节抽样审查，标题与段落组织满足要求。",
                "paper_evidence_ids": ["P1", "P2"],
                "standard_evidence_ids": ["S1"],
            }
        ],
        paper_evidences=[
            {"evidence_id": "P1", "page_number": 1, "bbox": [1, 2, 3, 4]},
            {"evidence_id": "P2", "page_number": 6, "bbox": None},
        ],
        standard_evidences=[{"evidence_id": "S1", "canonical_rule_id": "organization"}],
        coverage_report={"missing_categories": []},
    )

    assert findings[0]["result"] == "compliant"
    assert findings[0]["evidence_status"] == "complete"


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


def test_review_plan_keeps_top_level_sections_as_independent_units():
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
    assert len(body_units) == 7
    assert sum(len(unit["fact_ids"]) for unit in body_units) == len(facts)
    assert [unit["title"] for unit in body_units] == [
        f"{index} Main section" for index in range(1, 8)
    ]
    assert all(unit["page_range"][0] <= unit["page_range"][1] for unit in body_units)


def test_review_plan_does_not_merge_long_second_and_third_sections():
    facts = []
    evidence_index = 0
    sections = (
        ("1 Introduction", [1]),
        ("2 Related Work", [2, 3]),
        ("3 Method", [4, 5, 6]),
    )
    for section, pages in sections:
        for page in pages:
            evidence_index += 1
            facts.append(
                {
                    "evidence_id": f"P{evidence_index}",
                    "block_id": f"b-{evidence_index}",
                    "page_number": page,
                    "quote": f"Content for {section}",
                    "role": "body",
                    "section_title": section,
                }
            )

    units = _plan_review_units(facts)
    body_units = [unit for unit in units if unit["unit_kind"] == "body_section"]

    assert [unit["title"] for unit in body_units] == [
        "1 Introduction",
        "2 Related Work",
        "3 Method",
    ]
    assert [unit["page_range"] for unit in body_units] == [[1, 1], [2, 3], [4, 6]]


def test_coarse_body_group_does_not_use_a_subsection_as_the_chapter_title():
    facts = [
        {"page_number": 5, "section_title": "4.1 Datasets and Features"},
        {"page_number": 5, "section_title": "4 Experiments"},
        {"page_number": 6, "section_title": "4.2 Evaluation"},
    ]

    groups = _coarse_body_groups(facts)

    assert [title for title, _ in groups] == ["4 Experiments"]


def test_review_plan_keeps_native_graphic_geometry_in_figure_table_unit():
    facts = [
        {
            "evidence_id": "P1",
            "block_id": "caption-1",
            "page_number": 2,
            "quote": "Figure 1: Model overview",
            "role": "paragraph",
            "section_title": "1 Introduction",
        },
        {
            "evidence_id": "P2",
            "block_id": "native-vector-1",
            "page_number": 2,
            "bbox": [72, 100, 280, 260],
            "quote": "vector_graphic object; visual content not inspected",
            "role": "native_vector_graphic_object",
            "page_width_pt": 612.0,
            "page_height_pt": 792.0,
        },
    ]

    units = _plan_review_units(facts)

    figure_table = next(unit for unit in units if unit["unit_kind"] == "figure_table")
    assert set(figure_table["fact_ids"]) == {"P1", "P2"}


def test_figure_and_table_contexts_include_native_graphic_geometry():
    facts = [
        {
            "evidence_id": "P1",
            "page_number": 2,
            "bbox": [72, 100, 280, 260],
            "quote": "vector_graphic object; visual content not inspected",
            "role": "native_vector_graphic_object",
        },
        {
            "evidence_id": "P2",
            "page_number": 2,
            "bbox": [72, 270, 280, 282],
            "quote": "Table 1: Results",
            "role": "paragraph",
        },
        {
            "evidence_id": "P3",
            "page_number": 2,
            "bbox": [72, 284, 280, 296],
            "quote": "Figure 1: Overview",
            "role": "paragraph",
        },
    ]

    figure_ids = {item["evidence_id"] for item in _context_facts_for_category(facts, "figure")}
    table_ids = {item["evidence_id"] for item in _context_facts_for_category(facts, "table")}

    assert figure_ids == {"P1", "P3"}
    assert table_ids == {"P1", "P2"}


def test_reference_rules_are_pdf_observable_and_target_reference_unit():
    scope = scope_for_v2(
        {
            "title": "References formatting",
            "description": (
                "References follow the acknowledgments. Use an unnumbered first-level heading "
                "and a consistent citation style."
            ),
        }
    )

    assert scope["status"] == "active"
    assert scope["rule_category"] == "reference"
    assert scope["applicable_unit_kinds"] == ["reference"]
    assert scope["evidence_selector"] == ["reference_entry", "font_style", "text_content"]


def test_reference_scope_is_not_silently_skipped_when_section_roles_are_missing():
    assert _reference_scope()["applicability_conditions"] == {}


def test_icml_scope_activates_pdf_observable_table_and_layout_rules():
    table = _icml_scope("2.8 Tables", "All tables must be centered and have a title.")
    layout = _icml_scope("2.1 Dimensions", "Use US letter size and two columns.")
    abstract = _icml_scope("2.4 Abstract", "The abstract is in a single paragraph.")

    assert table["status"] == "active"
    assert table["rule_category"] == "table"
    assert table["applicable_unit_kinds"] == ["figure_table"]
    assert layout["status"] == "active"
    assert layout["is_global"] is True
    assert abstract["status"] == "active"
    assert abstract["applicable_unit_kinds"] == ["abstract"]


def test_icml_scope_excludes_rules_without_supported_pdf_facts():
    paragraph = _icml_scope(
        "2.5.2 Paragraphs and footnotes",
        "Do not indent paragraphs and place footnotes at the bottom of each column.",
    )
    algorithm = _icml_scope("2.7 Algorithms", "Use the algorithmic environment.")
    impact = _icml_scope("Impact statement", "Include societal consequences.")
    self_citation = _icml_scope("2.3.1 Self-citations", "Refer to yourself in the third person.")

    assert paragraph["status"] == "disabled"
    assert algorithm["status"] == "disabled"
    assert impact["status"] == "disabled"
    assert self_citation["status"] == "disabled"


def test_icml_scope_marks_complex_visible_rules_as_sampled_and_bounds_strict_checks():
    title = _icml_scope("2.2 Title", "The title should be centered in 14 point bold type.")
    figure = _icml_scope("2.6 Figures", "Captions should follow figures.")

    assert title["assessment_mode"] == "strict"
    assert title["supported_checks"] == ["title_font_size", "title_weight", "title_alignment"]
    assert figure["assessment_mode"] == "sampled"
    assert "caption_font" in figure["supported_checks"]


def test_figure_rules_request_object_geometry_without_visual_content():
    scope = scope_for_v2(
        {
            "title": "Figure formatting",
            "description": "All artwork must be centered. The figure number and caption follow it.",
        }
    )

    assert scope["status"] == "active"
    assert scope["rule_category"] == "figure"
    assert "object_geometry" in scope["evidence_selector"]


def test_exact_rule_evidence_survives_partial_category_retrieval():
    findings = validate_findings(
        [
            {
                "rule_ids": ["reference-heading"],
                "category": "reference",
                "aspect": "Reference heading",
                "result": "compliant",
                "finding": "The heading is present.",
                "paper_evidence_ids": ["P1"],
                "standard_evidence_ids": ["S1"],
            }
        ],
        paper_evidences=[{"evidence_id": "P1", "page_number": 9, "bbox": [1, 2, 3, 4]}],
        standard_evidences=[
            {
                "evidence_id": "S1",
                "canonical_rule_id": "reference-heading",
                "quote": "Use an unnumbered first-level heading for references.",
            }
        ],
        coverage_report={
            "missing_categories": ["reference"],
            "missing_rule_ids": ["reference-font-size"],
        },
    )

    assert findings[0]["result"] == "compliant"
    assert findings[0]["rule_ids"] == ["reference-heading"]


def test_declared_rule_drops_standard_evidence_from_other_rules():
    findings = validate_findings(
        [
            {
                "rule_ids": ["table-rule"],
                "category": "table",
                "aspect": "Table placement",
                "result": "unverifiable",
                "finding": "Spacing cannot be confirmed.",
                "paper_evidence_ids": ["P1"],
                "standard_evidence_ids": ["S1", "S2"],
            }
        ],
        paper_evidences=[{"evidence_id": "P1", "page_number": 2, "bbox": [1, 2, 3, 4]}],
        standard_evidences=[
            {"evidence_id": "S1", "canonical_rule_id": "table-rule", "quote": "Table rule"},
            {"evidence_id": "S2", "canonical_rule_id": "figure-rule", "quote": "Figure rule"},
        ],
        coverage_report={"missing_categories": []},
    )

    assert [item["evidence_id"] for item in findings[0]["standard_evidences"]] == ["S1"]


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


def test_oversized_figure_table_unit_remains_a_single_review_block():
    facts = [
        {
            "evidence_id": "P1",
            "block_id": "figure-1",
            "page_number": 2,
            "quote": "Figure 1 " + "evidence " * 40,
            "role": "paragraph",
        },
        {
            "evidence_id": "P2",
            "block_id": "table-1",
            "page_number": 3,
            "quote": "Table 1 " + "evidence " * 40,
            "role": "paragraph",
        },
    ]
    unit = {
        "unit_id": "u-001",
        "unit_position": 0,
        "unit_kind": "figure_table",
        "title": "Figures and tables",
        "page_range": [2, 3],
        "block_ids": ["figure-1", "table-1"],
        "fact_ids": ["P1", "P2"],
        "expected_rule_ids": ["figure-rule"],
        "allocated_rule_ids": ["figure-rule"],
        "global_rule_ids": [],
        "not_applicable_rule_ids": [],
        "retrieved_rule_ids": ["figure-rule"],
        "coverage": {"complete": True},
    }
    rules_by_id = {
        "figure-rule": {
            "rule_id": "figure-rule",
            "rule_category": "figure",
            "title": "Figure placement",
            "description": "Figures are inspected together.",
        }
    }

    refined, metrics = _refine_units_for_context_budget(
        units=[unit],
        facts=facts,
        standards=[{"canonical_rule_id": "figure-rule", "quote": "Figure rule"}],
        rules_by_id=rules_by_id,
        submission_mode="initial_submission",
        context_budget=0,
    )

    assert metrics["split_parent_count"] == 0
    assert len(refined) == 1
    assert refined[0]["context_refinement"]["status"] == "aggregate_figure_table_exceeds_budget"


def test_native_pdf_extractor_preserves_span_geometry_and_style(tmp_path):
    import fitz

    path = tmp_path / "native.pdf"
    document = fitz.open()
    page = document.new_page(width=612, height=792)
    page.insert_text((72, 96), "Format Evidence", fontsize=14, fontname="helv")
    page.draw_rect(fitz.Rect(72, 120, 280, 240))
    document.save(path)
    document.close()

    layout = extract_native_pdf_layout(str(path))

    assert layout.available is True
    assert layout.page_count == 1
    assert layout.spans[0]["text"] == "Format Evidence"
    assert layout.spans[0]["font_size_pt"] == 14.0
    assert len(layout.spans[0]["bbox"]) == 4
    assert any(item["object_type"] == "vector_graphic" for item in layout.objects)

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


def test_appendix_planning_requires_a_heading_and_bounds_references_before_it():
    facts = [
        {
            "evidence_id": "P1",
            "block_id": "body-mention",
            "page_number": 4,
            "quote": "More details are available in the appendix.",
            "role": "text",
            "section_title": "4 Experiments",
        },
        {
            "evidence_id": "P2",
            "block_id": "references",
            "page_number": 9,
            "quote": "References",
            "role": "heading",
            "section_title": "References",
        },
        {
            "evidence_id": "P3",
            "block_id": "reference-entry",
            "page_number": 10,
            "quote": "Ada, A. An example reference.",
            "role": "text",
            "section_title": "References",
        },
        {
            "evidence_id": "P3a",
            "block_id": "native-span-reference-author",
            "page_number": 9,
            "quote": "A. Smith",
            "role": "native_text_span",
            "section_title": None,
        },
        {
            "evidence_id": "P4",
            "block_id": "appendix-heading",
            "page_number": 12,
            "quote": "A. Additional experiments",
            "role": "heading",
            "section_title": "A. Additional experiments",
        },
        {
            "evidence_id": "P5",
            "block_id": "appendix-content",
            "page_number": 13,
            "quote": "Appendix experiment detail.",
            "role": "text",
            "section_title": "A. Additional experiments",
        },
    ]

    assert _appendix_pages(facts) == {12, 13}
    units = _plan_review_units(facts)
    reference = next(unit for unit in units if unit["unit_kind"] == "reference")
    appendix = next(unit for unit in units if unit["unit_kind"] == "appendix")
    assert reference["page_range"] == [9, 10]
    assert appendix["page_range"] == [12, 13]


def test_model_context_omits_large_fact_and_block_id_lists():
    unit = {
        "unit_id": "u-global",
        "unit_kind": "global",
        "page_range": [1, 13],
        "fact_ids": [f"P{index}" for index in range(5000)],
        "block_ids": [f"b-{index}" for index in range(5000)],
        "expected_rule_ids": ["page-size"],
    }

    compact = _model_unit_plan(unit)

    assert "fact_ids" not in compact
    assert "block_ids" not in compact
    assert compact["expected_rule_ids"] == ["page-size"]


def test_unverifiable_rule_fallback_retains_inspected_pdf_and_standard_evidence():
    facts = [
        {
            "evidence_id": "P1",
            "page_number": 1,
            "bbox": [72.0, 100.0, 320.0, 114.0],
            "quote": "1. Introduction",
            "role": "heading",
            "section_title": "1. Introduction",
            "font_size_pt": 12.0,
        }
    ]
    standards = [{"evidence_id": "S1", "canonical_rule_id": "heading-rule"}]
    rules = {"heading-rule": {"rule_category": "heading", "title": "Heading"}}

    findings = _unverifiable_findings_for_rules(
        ["heading-rule"],
        rules,
        "模型调用失败。",
        facts=facts,
        standards=standards,
    )

    assert findings[0]["result"] == "unverifiable"
    assert findings[0]["evidence_status"] == "incomplete"
    assert [item["evidence_id"] for item in findings[0]["paper_evidences"]] == ["P1"]
    assert [item["evidence_id"] for item in findings[0]["standard_evidences"]] == ["S1"]

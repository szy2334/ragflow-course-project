"""Standalone search test for the public ICML-content standards dataset."""

from public_search_common import DatasetProfile, run_dataset_cli


PROFILE = DatasetProfile(
    key="icml_content",
    dataset_name="public_research_content_icml_2026",
    display_name="ICML 2026 内容评审标准库",
    indexed_document_name="icml_2026_reviewer_instructions.html",
    questions=(
        "ICML如何评价论文的技术可靠性？",
        "如何判断实验是否充分支持论文的核心贡献？",
        "Soundness、Presentation、Significance和Originality如何评分？",
    ),
)


if __name__ == "__main__":
    raise SystemExit(run_dataset_cli(PROFILE))

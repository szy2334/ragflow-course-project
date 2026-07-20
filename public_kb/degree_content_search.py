"""Standalone search test for the public degree-content standards dataset."""

from public_search_common import DatasetProfile, run_dataset_cli


PROFILE = DatasetProfile(
    key="degree_content",
    dataset_name="public_degree_content_2026",
    display_name="学位论文内容标准库",
    indexed_document_name="degree_content_source_2026.pdf",
    questions=(
        "学位论文摘要需要包含哪些内容？",
        "如何评价学位论文的论点是否明确、论证是否充分？",
        "学位论文综合评价分为哪些等级？",
    ),
)


if __name__ == "__main__":
    raise SystemExit(run_dataset_cli(PROFILE))

"""Standalone search test for the public NeurIPS-format standards dataset."""

from public_search_common import DatasetProfile, run_dataset_cli


PROFILE = DatasetProfile(
    key="neurips_format",
    dataset_name="public_research_format_neurips_2026",
    display_name="NeurIPS 2026 格式标准库",
    indexed_document_name="neurips_2026_format_source.txt",
    questions=(
        "NeurIPS论文主PDF中的内容顺序是什么？",
        "NeurIPS论文是否需要包含论文检查清单？",
        "请只检查NeurIPS格式，不评价论文内容。",
    ),
)


if __name__ == "__main__":
    raise SystemExit(run_dataset_cli(PROFILE))

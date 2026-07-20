"""Standalone search test for the public degree-format standards dataset."""

from public_search_common import DatasetProfile, run_dataset_cli


PROFILE = DatasetProfile(
    key="degree_format",
    dataset_name="public_degree_format_2026",
    display_name="学位论文格式标准库",
    indexed_document_name="degree_format_document.txt",
    questions=(
        "学位论文中的英文缩写第一次出现时应该怎么写？",
        "学位论文的标点符号和专业术语有哪些要求？",
        "请只检查学位论文格式，不评价论文研究内容。",
    ),
)


if __name__ == "__main__":
    raise SystemExit(run_dataset_cli(PROFILE))

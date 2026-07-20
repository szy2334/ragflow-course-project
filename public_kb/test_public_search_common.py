import unittest

from public_kb.public_search_common import (
    build_answer,
    filter_document_chunks,
    rank_chunks,
)


class PublicSearchCommonTest(unittest.TestCase):
    def test_rank_chunks_filters_threshold_and_sorts_descending(self):
        chunks = [
            {"content": "middle", "similarity": 0.52},
            {"content": "low", "similarity": 0.19},
            {"content": "highest", "rerank_score": 0.91, "similarity": 0.33},
            {"content": "second", "score": 0.71},
        ]

        ranked = rank_chunks(chunks)

        self.assertEqual(
            [chunk["content"] for chunk in ranked],
            ["highest", "second", "middle"],
        )

    def test_build_answer_returns_ranked_evidence(self):
        ranked = [
            {
                "content": "摘要应包含研究方法和主要结论。",
                "similarity": 0.86,
                "standard_id": "DEGREE-CONTENT-ABSTRACT-001",
            }
        ]

        answer, evidences = build_answer("测试标准库", ranked)

        self.assertIn("摘要应包含研究方法和主要结论", answer)
        self.assertEqual(evidences[0]["rank"], 1)
        self.assertEqual(evidences[0]["score"], 0.86)

    def test_ragflow_content_with_weight_is_supported(self):
        ranked = rank_chunks(
            [
                {
                    "content_with_weight": (
                        "标准编号：STD-001\n标准描述：有效规则\n标准版本：2026.1"
                    ),
                    "docnm_kwd": "standards.jsonl",
                    "similarity": 0.75,
                }
            ]
        )

        _, evidences = build_answer("测试标准库", ranked)

        self.assertEqual(evidences[0]["standard_id"], "STD-001")
        self.assertEqual(evidences[0]["standard_version"], "2026.1")
        self.assertEqual(evidences[0]["document_name"], "standards.jsonl")

    def test_build_answer_reports_no_evidence(self):
        answer, evidences = build_answer("测试标准库", [])

        self.assertIn("未检索到", answer)
        self.assertEqual(evidences, [])

    def test_document_scope_is_checked_after_retrieval(self):
        chunks = [
            {"doc_id": "allowed", "content_with_weight": "允许", "similarity": 0.8},
            {"doc_id": "other", "content_with_weight": "其他", "similarity": 0.9},
        ]

        scoped = filter_document_chunks(chunks, "allowed")

        self.assertEqual([chunk["doc_id"] for chunk in scoped], ["allowed"])

    def test_document_name_handles_manual_chunk_internal_id(self):
        chunks = [
            {
                "doc_id": "internal-id",
                "docnm_kwd": "degree_format_document.txt",
                "content_with_weight": "允许",
                "similarity": 0.8,
            }
        ]

        scoped = filter_document_chunks(
            chunks,
            "manifest-id",
            "degree_format_document.txt",
        )

        self.assertEqual(len(scoped), 1)


if __name__ == "__main__":
    unittest.main()

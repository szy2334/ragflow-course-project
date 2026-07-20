import unittest

from public_kb.public_evidence_policy import evaluate_public_evidence


class PublicEvidencePolicyTests(unittest.TestCase):
    def test_empty_public_results_are_explicitly_no_evidence(self):
        decision = evaluate_public_evidence([], min_score=0.5)
        self.assertEqual(decision.status, "no_evidence")
        self.assertEqual(decision.warning_code, "PUBLIC_KB_NO_EVIDENCE")
        self.assertFalse(decision.score_allowed)


    def test_low_score_or_wrong_scope_is_not_valid_evidence(self):
        chunks = [
            {
                "content": "低相关性标准",
                "standard_id": "S-low",
                "standard_version": "2026.1",
                "paper_type": "research",
                "rule_type": "content",
                "similarity": 0.2,
            },
            {
                "content": "其他类型标准",
                "standard_id": "S-wrong-scope",
                "standard_version": "2026.1",
                "paper_type": "degree",
                "rule_type": "content",
                "similarity": 0.9,
            },
        ]
        decision = evaluate_public_evidence(
            chunks,
            min_score=0.5,
            filters={"paper_type": "research", "rule_type": "format"},
        )
        self.assertEqual(decision.status, "no_evidence")


    def test_matching_standard_allows_score(self):
        decision = evaluate_public_evidence(
            [
                {
                    "content": "实验设计标准",
                    "standard_id": "S1",
                    "standard_version": "2026.1",
                    "source_type": "standard",
                    "paper_type": "research",
                    "rule_type": "content",
                    "dimension": "experimental_design",
                    "similarity": 0.86,
                }
            ],
            min_score=0.5,
            filters={"paper_type": "research", "rule_type": "content"},
        )
        self.assertEqual(decision.status, "found")
        self.assertTrue(decision.score_allowed)
        self.assertEqual(len(decision.items), 1)

    def test_matching_evidence_is_sorted_by_score_descending(self):
        chunks = [
            {
                "standard_id": f"STD-{score}",
                "standard_version": "2026.1",
                "source_type": "standard",
                "content": "有效标准",
                "similarity": score,
            }
            for score in (0.31, 0.88, 0.57)
        ]

        decision = evaluate_public_evidence(chunks, min_score=0.2)

        self.assertEqual(
            [item["similarity"] for item in decision.items],
            [0.88, 0.57, 0.31],
        )


if __name__ == "__main__":
    unittest.main()

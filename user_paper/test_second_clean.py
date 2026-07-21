import unittest

from pipeline_common import estimate_tokens
from second_clean import (
    group_text_blocks,
    normalize_formula_spacing,
    split_text_at_token_limit,
)


class TruncationMergeTests(unittest.TestCase):
    def test_oversized_block_is_split_without_losing_words(self) -> None:
        text = " ".join(f"word{index}" for index in range(1, 26))

        fragments = split_text_at_token_limit(text, max_tokens=6)

        self.assertGreater(len(fragments), 1)
        self.assertTrue(all(estimate_tokens(fragment) <= 6 for fragment in fragments))
        self.assertEqual(" ".join(fragments), text)

    def test_truncated_tail_merges_with_following_text(self) -> None:
        document = {
            "paper_id": "paper-1",
            "paper_version_id": "version-1",
            "parser_name": "test",
            "parser_version": "test",
        }
        blocks = [
            {
                "indexable": True,
                "content_role": "paragraph",
                "normalized_text": " ".join(
                    f"alpha{index}" for index in range(1, 13)
                ),
                "section_path": ["Introduction"],
                "page_start": 1,
                "page_end": 1,
                "bbox": None,
                "source_ref": "paper://paper-1/page/1/block/1",
                "block_id": "block-1",
            },
            {
                "indexable": True,
                "content_role": "paragraph",
                "normalized_text": "omega1 omega2 omega3",
                "section_path": ["Introduction"],
                "page_start": 1,
                "page_end": 1,
                "bbox": None,
                "source_ref": "paper://paper-1/page/1/block/2",
                "block_id": "block-2",
            },
        ]

        chunks = group_text_blocks(document, blocks, target_tokens=6, max_tokens=10)
        visible = [chunk for chunk in chunks if chunk["indexable"]]

        self.assertEqual(len(visible), 2)
        self.assertIn("omega1 omega2 omega3", visible[-1]["raw_content"])
        self.assertIn("block-1", visible[-1]["source_block_ids"])
        self.assertIn("block-2", visible[-1]["source_block_ids"])
        self.assertTrue(
            visible[-1]["provenance"]["truncation_merge"]["source_block_fragments"]
        )
        self.assertTrue(
            all(estimate_tokens(chunk["raw_content"]) <= 10 for chunk in visible)
        )

    def test_character_spaced_formula_is_normalized(self) -> None:
        formula = "l r a t e = d _ {\\mathrm{model}} ^ {- 0. 5} \\cdot w a r m u p\\_s t e p s"

        repaired = normalize_formula_spacing(formula)

        self.assertEqual(
            repaired, "lrate = d_{\\mathrm{model}} ^ {-0.5} \\cdot warmup\\_steps"
        )


if __name__ == "__main__":
    unittest.main()

import pytest

from app.ai.agents import fallback_route


@pytest.mark.parametrize(
    ("question", "summary", "initial", "effective"),
    [
        ("论文使用了什么数据集？", "", "fact", "fact"),
        ("作者为什么选择这个损失函数？", "", "explain", "explain"),
        ("实验设计是否充分？", "", "fact", "fact"),
        ("请为实验充分性评分。", "", "fact", "fact"),
        ("那实验部分呢？", "上一轮讨论了方法。", "follow_up", "fact"),
        ("今天天气怎么样？", "", "general_chat", "general_chat"),
        ("hi", "", "general_chat", "general_chat"),
    ],
)
def test_deterministic_fallback_covers_all_routes(question, summary, initial, effective):
    route = fallback_route(question, summary)
    assert route.initial_route_type == initial
    assert route.effective_route_type == effective


def test_evaluation_fallback_has_non_evaluative_warning():
    route = fallback_route("请评价实验是否充分。", "")
    assert route.review_dimensions == []
    assert not route.needs_public_kb
    assert any("non-evaluative paper reading" in warning for warning in route.warnings)

"""Workflow execution policies kept separate from business state."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WorkflowPolicy:
    context_timeout_seconds: float = 10.0
    retrieval_timeout_seconds: float = 20.0
    delta_chunk_size: int = 240
    max_semantic_repairs: int = 1


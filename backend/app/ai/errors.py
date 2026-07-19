"""Domain errors raised by the AI package."""


class AiWorkflowError(Exception):
    """Base error carrying a stable cross-boundary error code."""

    code = "AI_WORKFLOW_ERROR"
    retryable = False


class WorkflowCancelled(AiWorkflowError):
    code = "TASK_CANCELLED"


class ModelTransportError(AiWorkflowError):
    code = "MODEL_TRANSPORT_ERROR"
    retryable = True


class ModelOutputInvalid(AiWorkflowError):
    code = "MODEL_OUTPUT_INVALID"


class PaperEvidenceUnavailable(AiWorkflowError):
    code = "RAG_NO_EVIDENCE"


class AnswerPersistenceFailed(AiWorkflowError):
    code = "AI_WORKFLOW_ERROR"
    retryable = True

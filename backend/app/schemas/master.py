from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

from app.schemas.source_understanding import SourceUnderstanding
from app.schemas.content_strategy import ContentStrategy
from app.schemas.content_generation import GeneratedContent
from app.schemas.validation import ValidationResult
from app.schemas.action import ActionResult


class MasterDecision(BaseModel):
    """
    Structured routing decision emitted by the Master Agent.
    Dictates the execution flow through LangGraph nodes.
    """
    intent: str = Field(
        ...,
        description="High-level classification of user intent (e.g., 'source_analysis', 'social_generation', 'publish_content').",
    )
    requires_source_understanding: bool = Field(
        ...,
        description="True if raw source text needs to be analyzed by Agent 1.",
    )
    requires_strategy: bool = Field(
        ...,
        description="True if multi-platform content strategy must be formulated by Agent 2.",
    )
    requires_generation: bool = Field(
        ...,
        description="True if platform content must be produced by Agent 3.",
    )
    requires_validation: bool = Field(
        ...,
        description="True if generated copy must be audited by Agent 4.",
    )
    requires_action: bool = Field(
        ...,
        description="True if an operational action (preview, export, publish) is requested for Agent 5.",
    )
    requires_human_approval: bool = Field(
        ...,
        description="True if the action involves sensitive operations (such as publishing) that mandate human approval.",
    )
    requested_platforms: List[str] = Field(
        default_factory=list,
        description="Platforms requested by the user: 'linkedin', 'instagram', 'advisory'.",
    )
    requested_formats: List[str] = Field(
        default_factory=list,
        description="Formats requested (e.g., 'post', 'carousel', 'executive_brief').",
    )
    requested_action: Optional[str] = Field(
        default=None,
        description="The specific action requested: 'preview', 'export', or 'publish'.",
    )
    user_request: str = Field(
        ...,
        description="The verbatim natural language request from the user.",
    )
    reasoning_summary: str = Field(
        ...,
        description="A concise operational explanation of the routing decision. Contains no secret chain-of-thought.",
    )


class MasterRunRequest(BaseModel):
    """
    Request model for the main Master Agent orchestration endpoint (POST /api/master/run).
    """
    user_request: str = Field(
        ...,
        description="Natural language instruction guiding the agentic workflow.",
        examples=["Take this report and create a professional LinkedIn post and publish it."],
    )
    source_text: Optional[str] = Field(
        default=None,
        description="Raw document or source material to be processed.",
        examples=["Artificial intelligence is transforming modern healthcare by helping doctors analyze medical images."],
    )
    workflow_id: Optional[str] = Field(
        default=None,
        description="Optional pre-assigned thread/workflow identifier. Generated automatically if omitted.",
    )

    @field_validator("user_request")
    @classmethod
    def validate_request(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("user_request must not be empty or whitespace-only.")
        return v.strip()


class MasterRunResponse(BaseModel):
    """
    Structured outcome returned by the Master Agent workflow invocation.
    """
    workflow_id: str = Field(
        ...,
        description="Unique thread identifier for resuming or inspecting workflow state.",
    )
    status: str = Field(
        ...,
        description="Workflow status: 'completed', 'waiting_for_approval', 'validation_failed', or 'failed'.",
    )
    current_stage: str = Field(
        ...,
        description="Current pipeline stage: 'master', 'source', 'strategy', 'generation', 'validation', 'human_approval', 'action'.",
    )
    decision: MasterDecision = Field(
        ...,
        description="The intelligent routing decision produced by the Master Agent.",
    )
    source_understanding: Optional[SourceUnderstanding] = Field(
        default=None,
        description="Extracted facts if Agent 1 executed.",
    )
    content_strategy: Optional[ContentStrategy] = Field(
        default=None,
        description="Strategy if Agent 2 executed.",
    )
    generated_content: Optional[GeneratedContent] = Field(
        default=None,
        description="Generated copy if Agent 3 executed.",
    )
    validation: Optional[ValidationResult] = Field(
        default=None,
        description="Validation report if Agent 4 executed.",
    )
    action_result: Optional[ActionResult] = Field(
        default=None,
        description="Action result if Agent 5 executed.",
    )
    approval_required: bool = Field(
        default=False,
        description="Flag indicating if workflow is currently paused awaiting human approval.",
    )
    revision_count: int = Field(
        default=0,
        description="Number of autonomous revision cycles executed.",
    )
    validation_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Audit log of iterative validation attempts.",
    )
    message: str = Field(
        ...,
        description="User-friendly status or completion message.",
    )


class MasterApproveRequest(BaseModel):
    """
    Request model for human approval (POST /api/master/approve).
    """
    workflow_id: str = Field(
        ...,
        description="Unique workflow identifier to approve and resume.",
    )

    @field_validator("workflow_id")
    @classmethod
    def validate_workflow_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("workflow_id must not be empty.")
        return v.strip()


class MasterRejectRequest(BaseModel):
    """
    Request model for human rejection with feedback (POST /api/master/reject).
    """
    workflow_id: str = Field(
        ...,
        description="Unique workflow identifier to reject.",
    )
    feedback: str = Field(
        ...,
        description="Actionable user feedback detailing why content was rejected and how to revise it.",
    )

    @field_validator("workflow_id", "feedback")
    @classmethod
    def validate_fields(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Fields must not be empty.")
        return v.strip()


class MasterStatusResponse(BaseModel):
    """
    Response model for querying workflow state (GET /api/master/status/{workflow_id}).
    """
    workflow_id: str = Field(...)
    status: str = Field(...)
    current_stage: str = Field(...)
    approval_required: bool = Field(...)
    human_approval_status: str = Field(...)
    content: Optional[GeneratedContent] = Field(default=None)
    validation: Optional[ValidationResult] = Field(default=None)
    action_result: Optional[ActionResult] = Field(default=None)
    revision_count: int = Field(default=0)
    message: str = Field(...)

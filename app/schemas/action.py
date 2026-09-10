from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

from app.schemas.content_generation import GeneratedContent


class ActionRequest(BaseModel):
    """
    Request model for Agent 5 (Action / Publishing Agent).
    Encapsulates the validated generated content, the requested action, and the validation gate status.
    """
    action: str = Field(
        ...,
        description="The action to perform: 'preview', 'export', or 'publish'.",
        examples=["preview", "export", "publish"],
    )
    platform: str = Field(
        default="all",
        description="Target platform: 'linkedin', 'instagram', 'advisory', or 'all'.",
    )
    content: GeneratedContent = Field(
        ...,
        description="The validated generated content from Agent 3 / Agent 4.",
    )
    validation_status: str = Field(
        ...,
        description="Validation outcome: 'PASS' or 'FAIL'.",
    )
    validation_issues: List[str] = Field(
        default_factory=list,
        description="List of issues flagged by Agent 4 if validation did not pass.",
    )

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        cleaned = v.strip().lower()
        allowed = {"preview", "export", "publish"}
        if cleaned not in allowed:
            raise ValueError(f"Unsupported action: '{v}'. Must be one of {allowed}.")
        return cleaned

    @field_validator("platform")
    @classmethod
    def validate_platform(cls, v: str) -> str:
        cleaned = v.strip().lower()
        allowed = {"linkedin", "instagram", "advisory", "all"}
        if cleaned not in allowed:
            raise ValueError(f"Unsupported platform: '{v}'. Must be one of {allowed}.")
        return cleaned


class ActionResult(BaseModel):
    """
    Structured outcome model returned by Agent 5 (Action / Publishing Agent).
    Predictable and typed, preventing random unstructured responses.
    """
    success: bool = Field(
        ...,
        description="Whether the requested action completed successfully.",
    )
    action: str = Field(
        ...,
        description="The requested action that was processed ('preview', 'export', 'publish').",
    )
    platform: str = Field(
        ...,
        description="Target platform.",
    )
    action_allowed: bool = Field(
        ...,
        description="Whether the action was permitted by the validation gate.",
    )
    status: str = Field(
        ...,
        description="Outcome status: 'completed', 'blocked', 'dry_run', or 'failed'.",
    )
    message: str = Field(
        ...,
        description="Human-readable explanation of the action execution or blocking reason.",
    )
    content: Optional[GeneratedContent] = Field(
        default=None,
        description="The content associated with the action.",
    )
    validation_status: str = Field(
        ...,
        description="The validation status assessed prior to action execution.",
    )
    validation_issues: List[str] = Field(
        default_factory=list,
        description="Preserved validation issues if action was blocked.",
    )
    exported_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Exported payload structured for CMS/file export when action is 'export'.",
    )

from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from app.schemas.source_understanding import SourceUnderstanding
from app.schemas.content_strategy import ContentStrategy
from app.schemas.content_generation import GeneratedContent


class ValidationResult(BaseModel):
    """
    Structured validation output model produced by Agent 4 (Validation Agent).
    Captures factual grounding, hallucination checks, score, and actionable issues.
    """
    passed: bool = Field(
        ...,
        description="Whether the generated content satisfies all validation criteria (no hallucinations, accurate numbers, grounded facts).",
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Quality and fidelity score from 0.0 to 1.0. Typically >= 0.8 signifies a pass.",
    )
    issues: List[str] = Field(
        default_factory=list,
        description="Explicit list of identified issues, unsupported claims, or factual inaccuracies.",
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="Actionable suggestions for revising the content to pass validation.",
    )
    checks: Dict[str, bool] = Field(
        default_factory=dict,
        description="Granular check flags: grounding_verified, no_hallucinations, numbers_accurate, format_compliance.",
    )


class ValidationRequest(BaseModel):
    """
    Request model for the standalone Validation Agent endpoint.
    """
    source_understanding: SourceUnderstanding = Field(
        ...,
        description="Structured source facts from Agent 1 against which to validate.",
    )
    generated_content: GeneratedContent = Field(
        ...,
        description="Multi-platform generated copy from Agent 3 to be validated.",
    )
    content_strategy: Optional[ContentStrategy] = Field(
        default=None,
        description="Optional content strategy from Agent 2 to check strategic compliance.",
    )
    user_request: Optional[str] = Field(
        default=None,
        description="Optional original user prompt to verify requirements adherence.",
    )

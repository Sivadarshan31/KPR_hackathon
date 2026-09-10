from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
from app.schemas.source_understanding import SourceUnderstanding
from app.schemas.content_strategy import ContentStrategy
from app.schemas.content_generation import GeneratedContent


class ValidationStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


class ValidationResult(BaseModel):
    """
    Structured validation output model produced by Agent 4 (Validation Agent).
    Captures factual grounding, hallucination checks, score, and actionable issues.
    """
    status: ValidationStatus = Field(
        default=ValidationStatus.PASS,
        description="Validation status outcome: 'PASS' or 'FAIL'.",
    )
    passed: bool = Field(
        default=True,
        description="Whether the generated content satisfies all validation criteria.",
    )
    score: float = Field(
        ...,
        description="Quality and fidelity score. Supported ranges: 0 to 100 or 0.0 to 1.0.",
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

    @field_validator("score")
    @classmethod
    def validate_score_range(cls, v: float) -> float:
        if v < 0.0 or v > 100.0:
            raise ValueError(f"Validation score must be between 0 and 100 (got {v}).")
        return v

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, v: Any) -> ValidationStatus:
        if isinstance(v, bool):
            return ValidationStatus.PASS if v else ValidationStatus.FAIL
        if isinstance(v, str):
            cleaned = v.strip().upper()
            if cleaned == "PASS":
                return ValidationStatus.PASS
            if cleaned == "FAIL":
                return ValidationStatus.FAIL
        return v


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

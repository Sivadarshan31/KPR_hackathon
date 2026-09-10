from typing import Optional
from pydantic import BaseModel, Field, field_validator


class UserRequest(BaseModel):
    """
    Central user input request model for ContentForge content transformation.
    Captures source content and transformation instructions.
    """
    source_id: Optional[str] = Field(
        default=None,
        description="Optional unique identifier for the source content.",
    )
    source_text: str = Field(
        ...,
        description="The raw text of the source document to analyze and transform.",
        examples=[
            "Artificial intelligence is transforming healthcare. A recent pilot program analyzed 50,000 medical images over six months to assist doctors with diagnosis."
        ],
    )
    source_type: Optional[str] = Field(
        default="text",
        description="Type of source material (e.g., 'pdf', 'text', 'docx', 'article', 'report').",
    )
    requested_platform: Optional[str] = Field(
        default="all",
        description="Target platform: 'linkedin', 'instagram', 'advisory', or 'all'.",
    )
    target_audience: Optional[str] = Field(
        default=None,
        description="Optional custom target audience profile.",
    )
    tone: Optional[str] = Field(
        default=None,
        description="Optional desired writing tone (e.g., 'professional', 'casual', 'authoritative').",
    )
    instructions: Optional[str] = Field(
        default=None,
        description="Optional custom instructions or guidance for content generation.",
    )

    @field_validator("source_text")
    @classmethod
    def validate_source_text(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("source_text must not be empty or whitespace-only.")
        return value.strip()

    @field_validator("requested_platform")
    @classmethod
    def validate_platform(cls, v: Optional[str]) -> str:
        if not v:
            return "all"
        cleaned = v.strip().lower()
        allowed = {"linkedin", "instagram", "advisory", "all"}
        if cleaned not in allowed:
            raise ValueError(f"Unsupported platform: '{v}'. Must be one of {allowed}.")
        return cleaned

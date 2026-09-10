from typing import List
from pydantic import BaseModel, Field, field_validator


class SourceUnderstandingRequest(BaseModel):
    """
    Request model for the Source Understanding Agent.
    Accepts raw source text and ensures it is non-empty.
    """

    source_text: str = Field(
        ...,
        description="The raw plain text of the source document to analyze.",
        examples=[
            "The National Institute of Ocean Technology developed an autonomous underwater vehicle to monitor marine pollution. The vehicle uses sensors to collect water-quality measurements and can operate for up to 18 hours. The project began in 2026."
        ],
    )

    @field_validator("source_text")
    @classmethod
    def validate_source_text(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("source_text must not be empty or whitespace-only.")
        return value.strip()


class SourceUnderstanding(BaseModel):
    """
    Structured source understanding output model produced by Agent 1.
    All fields are source-grounded and designed for consumption by downstream agents.
    """

    title: str = Field(
        ...,
        description="A concise title describing the source. Preserves original title if explicit, or synthesizes one strictly from source text.",
    )
    summary: str = Field(
        ...,
        description="A concise summary capturing the central message without outside knowledge.",
    )
    main_topic: str = Field(
        ...,
        description="The central subject or core theme of the source text.",
    )
    key_points: List[str] = Field(
        default_factory=list,
        description="The most important ideas contained in the source.",
    )
    facts: List[str] = Field(
        default_factory=list,
        description="Explicit factual statements verified by the source text.",
    )
    entities: List[str] = Field(
        default_factory=list,
        description="Entities such as people, organizations, companies, technologies, locations, products, or events appearing in the source.",
    )
    important_numbers: List[str] = Field(
        default_factory=list,
        description="Numerical information (percentages, metrics, amounts, durations) preserved exactly as written.",
    )
    dates: List[str] = Field(
        default_factory=list,
        description="Explicit dates and temporal references mentioned in the source.",
    )
    claims: List[str] = Field(
        default_factory=list,
        description="Claims, assertions, or beliefs expressed in the source, clearly separated from objective facts.",
    )
    terminology: List[str] = Field(
        default_factory=list,
        description="Domain-specific terms, jargon, or keywords appearing in the source.",
    )
    target_audience: str = Field(
        default="Not specified",
        description="Intended audience only when supported by source evidence, otherwise 'Not specified'.",
    )
    tone: str = Field(
        default="Neutral",
        description="Writing tone (e.g., Informative, Technical, Professional, Academic, Promotional, News-style), or 'Neutral'.",
    )
    source_type: str = Field(
        default="Unknown",
        description="Identified document type (e.g., Article, Report, Research paper, Press release, Technical document, Case study), or 'Unknown'.",
    )

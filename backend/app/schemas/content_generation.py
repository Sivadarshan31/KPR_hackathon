from typing import List, Optional, Union
from pydantic import BaseModel, Field

from app.schemas.source_understanding import SourceUnderstanding
from app.schemas.content_strategy import ContentStrategy


class LinkedInContent(BaseModel):
    """
    Structured content model for LinkedIn posts.
    """
    hook: str = Field(
        ...,
        description="Attention-grabbing opening line or headline for LinkedIn.",
    )
    body: str = Field(
        ...,
        description="Main body text of the LinkedIn post providing core insights and context.",
    )
    cta: str = Field(
        ...,
        description="Call to action driving professional engagement or discussion.",
    )
    hashtags: List[str] = Field(
        default_factory=list,
        description="List of relevant professional hashtags.",
    )


class InstagramContent(BaseModel):
    """
    Structured content model for Instagram.
    Supports single captions, carousels (slides), hashtags, and calls to action.
    """
    caption: str = Field(
        ...,
        description="The primary narrative caption tailored for Instagram, including hook and body.",
    )
    slides: Optional[List[str]] = Field(
        default=None,
        description="Sequential slide content when a carousel format is specified by strategy; otherwise None.",
    )
    hashtags: List[str] = Field(
        default_factory=list,
        description="Curated list of hashtags aligned with source topics and strategy.",
    )
    call_to_action: Optional[str] = Field(
        default=None,
        description="Call to action directing audience engagement (comments, saves, shares).",
    )


class AdvisoryContent(BaseModel):
    """
    Structured content model for Advisory / Executive Briefings.
    """
    title: str = Field(
        ...,
        description="Executive advisory title or subject line.",
    )
    summary: str = Field(
        ...,
        description="Brief high-level summary of the advisory briefing.",
    )
    important_information: List[str] = Field(
        default_factory=list,
        description="Key findings, critical metrics, or core information points.",
    )
    recommended_actions: List[str] = Field(
        default_factory=list,
        description="Actionable recommendations for leadership or decision-makers.",
    )
    warning: Optional[str] = Field(
        default=None,
        description="Optional warning, risk alert, or time-sensitive notice.",
    )


class GeneratedContent(BaseModel):
    """
    Structured output model for Agent 3 (Content Generation Agent).
    Encapsulates generated content across supported MVP channels (LinkedIn, Instagram, Advisory).
    Fields accept both structured models and formatted strings for maximum flexibility.
    """
    linkedin: Optional[Union[LinkedInContent, str]] = Field(
        default=None,
        description="Professional LinkedIn post (structured model or formatted string).",
    )
    instagram: Optional[InstagramContent] = Field(
        default=None,
        description="Structured Instagram content including caption, optional carousel slides, hashtags, and CTA.",
    )
    advisory: Optional[Union[AdvisoryContent, str]] = Field(
        default=None,
        description="Concise advisory briefing (structured model or formatted string).",
    )


class ContentGenerationRequest(BaseModel):
    """
    Request model for the standalone Content Generation Agent endpoint.
    Receives both the structured SourceUnderstanding (Agent 1) and ContentStrategy (Agent 2).
    """
    source_understanding: SourceUnderstanding = Field(
        ...,
        description="Structured source understanding produced by Agent 1.",
    )
    content_strategy: ContentStrategy = Field(
        ...,
        description="Structured multi-platform content strategy produced by Agent 2.",
    )

from typing import List, Optional
from pydantic import BaseModel, Field

from app.schemas.source_understanding import SourceUnderstanding
from app.schemas.content_strategy import ContentStrategy


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


class GeneratedContent(BaseModel):
    """
    Structured output model for Agent 3 (Content Generation Agent).
    Encapsulates generated content across supported MVP channels (LinkedIn, Instagram, Advisory).
    Fields are optional to support platform selectivity when strategy only requests specific channels.
    """
    linkedin: Optional[str] = Field(
        default=None,
        description="Professional LinkedIn post with hook, body, spacing, and CTA based on strategy.",
    )
    instagram: Optional[InstagramContent] = Field(
        default=None,
        description="Structured Instagram content including caption, optional carousel slides, hashtags, and CTA.",
    )
    advisory: Optional[str] = Field(
        default=None,
        description="Concise advisory or executive briefing prioritizing high-signal actionable points.",
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

from typing import List, Optional
from pydantic import BaseModel, Field
from app.schemas.source_understanding import SourceUnderstanding


class LinkedInStrategy(BaseModel):
    """
    Platform-specific content strategy for LinkedIn.
    Focuses on professional, thought-leadership, or B2B audience orientation.
    """
    objective: str = Field(
        ...,
        description="The strategic communication objective for the LinkedIn post (e.g., Thought leadership, industry awareness, professional discussion).",
    )
    audience: str = Field(
        ...,
        description="Target professional audience profile on LinkedIn.",
    )
    angle: str = Field(
        ...,
        description="The content angle or hook tailored for a professional audience.",
    )
    key_message: str = Field(
        ...,
        description="The core takeaway or primary insight to be communicated.",
    )
    tone: str = Field(
        ...,
        description="Tone of voice for LinkedIn (e.g., Professional, Authoritative, Analytical, Conversational).",
    )
    cta: str = Field(
        ...,
        description="Call to action directing professional engagement or discussion.",
    )
    recommended_structure: List[str] = Field(
        default_factory=list,
        description="Sequential structural components for the post (e.g., ['Hook', 'Problem context', 'Key metric', 'Insight', 'Discussion CTA']).",
    )


class InstagramStrategy(BaseModel):
    """
    Platform-specific content strategy for Instagram.
    Focuses on visual storytelling, digestible carousel points, and engagement.
    """
    objective: str = Field(
        ...,
        description="Strategic communication goal for Instagram (e.g., Visual education, community engagement, brand awareness).",
    )
    audience: str = Field(
        ...,
        description="Target audience demographic or interest group on Instagram.",
    )
    angle: str = Field(
        ...,
        description="Engaging hook or angle suited for social and visual consumption.",
    )
    carousel_direction: List[str] = Field(
        default_factory=list,
        description="Slide-by-slide narrative outline for a carousel or multi-part graphic.",
    )
    visual_direction: str = Field(
        ...,
        description="Guidance on visual style, graphic concepts, imagery, or infographics.",
    )
    tone: str = Field(
        ...,
        description="Tone of voice for Instagram (e.g., Dynamic, Accessible, Inspiring, Educational).",
    )
    cta: str = Field(
        ...,
        description="Call to action prompting comments, saves, or shares.",
    )


class AdvisoryStrategy(BaseModel):
    """
    Platform-specific content strategy for Advisory / Executive Briefings.
    Focuses on high-signal recommendations, critical implications, and priority action items.
    """
    objective: str = Field(
        ...,
        description="Strategic objective of the advisory piece (e.g., Strategic briefing, risk mitigation, operational guidance).",
    )
    audience: str = Field(
        ...,
        description="Executive, technical, or decision-maker audience profile.",
    )
    key_information: List[str] = Field(
        default_factory=list,
        description="Critical factual points, findings, or metrics that inform the advisory.",
    )
    priority: str = Field(
        ...,
        description="Strategic priority level (e.g., High, Medium, Immediate, Informational).",
    )
    tone: str = Field(
        ...,
        description="Tone of voice for advisory (e.g., Executive, Direct, Objective, Pragmatic).",
    )
    recommended_structure: List[str] = Field(
        default_factory=list,
        description="Structured sections for the advisory notice or executive brief.",
    )


class ContentStrategy(BaseModel):
    """
    Structured content strategy produced by Agent 2.
    Synthesizes overall strategic direction and provides platform-specific strategies
    for LinkedIn, Instagram, and Advisory formats without hallucinating facts.
    """
    summary: str = Field(
        ...,
        description="Executive summary of the overall multi-platform content strategy.",
    )
    overall_angle: str = Field(
        ...,
        description="Overarching thematic narrative linking all platform content.",
    )
    target_audience: str = Field(
        ...,
        description="Comprehensive definition of the intended target audiences across channels.",
    )
    key_takeaway: str = Field(
        ...,
        description="The primary factual conclusion or core message that must remain consistent across platforms.",
    )
    linkedin: Optional[LinkedInStrategy] = Field(
        default=None,
        description="Tailored strategy for LinkedIn.",
    )
    instagram: Optional[InstagramStrategy] = Field(
        default=None,
        description="Tailored strategy for Instagram.",
    )
    advisory: Optional[AdvisoryStrategy] = Field(
        default=None,
        description="Tailored strategy for Advisory / Executive format.",
    )

    @property
    def content_angle(self) -> str:
        """Alias for overall_angle."""
        return self.overall_angle

    @property
    def core_message(self) -> str:
        """Alias for key_takeaway."""
        return self.key_takeaway


class ContentStrategyRequest(BaseModel):
    """
    Request model for the standalone Content Strategy Agent endpoint.
    Directly receives the structured SourceUnderstanding output from Agent 1.
    """
    source_understanding: SourceUnderstanding = Field(
        ...,
        description="Structured understanding produced by Agent 1 (Source Understanding Agent).",
    )

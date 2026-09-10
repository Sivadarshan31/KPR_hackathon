import json
import logging
import re
from typing import Optional, Union
from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.groq_manager import GroqManager, groq_manager as default_groq_manager
from app.llm.exceptions import GroqRequestError, sanitize_error_message
from app.schemas.source_understanding import SourceUnderstanding
from app.schemas.content_strategy import ContentStrategy
from app.schemas.content_generation import GeneratedContent

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """You are the ContentForge Content Generation Agent (Agent 3).

Your sole responsibility is to convert the structured source understanding (Agent 1) and structured content strategy (Agent 2) into high-quality, platform-specific content for the requested formats.

You are a CONTENT GENERATOR, not a strategist or validator.
- Do NOT plan or alter the content strategy.
- Do NOT validate or criticize the source.
- Do NOT explain your generation process or return commentary.
- Do NOT fabricate facts, statistics, numbers, dates, quotations, partnerships, revenue, or awards.

Strict Operational Guidelines:
1. Grounding & Anti-Hallucination:
   - Ground all factual assertions STRICTLY in the supplied SourceUnderstanding.
   - You may creatively phrase and adapt content for each platform, but you must NEVER invent facts.
   - If the source states "reduced processing time by 40%", you may say "achieved a 40% reduction in processing time", but you must NEVER say "60%".
   - Never invent: statistics, dates, names, organizations, customer numbers, revenue, funding, technical specs, quotations, awards, partnerships, or locations.
   - If the strategy requests an angle or detail not supported by the source understanding (e.g. "include revenue" when revenue is not in the source), do NOT invent it. Omit the unsupported detail.
2. Strategy Adherence:
   - Follow the target audience, tone, objective, content angle, key message, CTA, and structural outline defined by Agent 2.
3. Platform-Specific Generation:
   a) LinkedIn:
      - Strong opening hook.
      - Professional body with readable paragraphs and clear spacing.
      - Strategic call-to-action (CTA) matching the strategy.
      - Relevant hashtags matching source topics and strategy.
      - Avoid excessive emojis, generic AI fluff, and marketing hype.
   b) Instagram:
      - Engaging caption suited for social media.
      - If the strategy provides carousel direction / outline, structure the slides sequentially into the 'slides' list (e.g. ["Slide 1: Hook ...", "Slide 2: ..."]). If no carousel is requested, set 'slides' to null.
      - Curated, relevant hashtags in the 'hashtags' list.
      - Engaging community call-to-action (e.g. comments, saves, shares).
   c) Advisory:
      - Concise, high-signal advisory or executive briefing.
      - Prioritize clarity, direct actionable insights, target audience relevance, and source-grounded findings.
      - Avoid promotional or social-media language.
4. Platform Selectivity:
   - Generate content for platforms present in the ContentStrategy. If a platform is not requested or is omitted, leave its field as null.
5. Output Format:
   - You MUST output a single valid, raw JSON object matching the schema below.
   - Do NOT wrap in markdown code blocks or backticks.
   - Only return the JSON object.

Target Schema:
{schema}"""
SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE


def _parse_generation_json(raw_text: str) -> GeneratedContent:
    """Helper to parse and validate GeneratedContent from raw LLM output."""
    cleaned = raw_text.strip()
    # Strip <think> tags if present
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", cleaned).strip()
    # Strip markdown code fencing if present
    if cleaned.startswith("```"):
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if match:
            cleaned = match.group(1).strip()
    try:
        return GeneratedContent.model_validate_json(cleaned)
    except Exception:
        pass

    # Extract JSON object using raw_decode
    fb = cleaned.find("{")
    if fb != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(cleaned[fb:])
            return GeneratedContent.model_validate(obj)
        except Exception:
            pass

    data = json.loads(cleaned)
    return GeneratedContent.model_validate(data)


class ContentGenerationAgent:
    """
    Agent 3: Content Generation Agent.
    Transforms structured SourceUnderstanding and structured ContentStrategy into
    platform-ready content for LinkedIn, Instagram, and Advisory formats.
    Decoupled from HTTP frameworks to facilitate direct pipeline composition.
    """

    def __init__(
        self,
        manager: Optional[GroqManager] = None,
        temperature: float = 0.2,
    ):
        self.groq_manager = manager or default_groq_manager
        self.temperature = temperature
        self._schema_json = json.dumps(GeneratedContent.model_json_schema(), indent=2)

    def _get_system_prompt(self) -> str:
        return SYSTEM_PROMPT_TEMPLATE.format(schema=self._schema_json)

    def _format_input_prompt(
        self,
        source_understanding: Union[SourceUnderstanding, dict],
        content_strategy: Union[ContentStrategy, dict],
    ) -> str:
        """Serializes structured inputs from Agent 1 and Agent 2 into a clear prompt."""
        if isinstance(source_understanding, SourceUnderstanding):
            source_data = source_understanding.model_dump()
        elif isinstance(source_understanding, dict):
            source_data = SourceUnderstanding.model_validate(source_understanding).model_dump()
        else:
            raise ValueError("source_understanding must be an instance of SourceUnderstanding or a valid dict.")

        if isinstance(content_strategy, ContentStrategy):
            strategy_data = content_strategy.model_dump()
        elif isinstance(content_strategy, dict):
            strategy_data = ContentStrategy.model_validate(content_strategy).model_dump()
        else:
            raise ValueError("content_strategy must be an instance of ContentStrategy or a valid dict.")

        prompt_payload = {
            "STRUCTURED_SOURCE_UNDERSTANDING_AGENT_1": source_data,
            "STRUCTURED_CONTENT_STRATEGY_AGENT_2": strategy_data,
        }
        return (
            "GENERATE PLATFORM CONTENT FROM THE FOLLOWING INPUTS:\n\n"
            f"{json.dumps(prompt_payload, indent=2)}"
        )

    def generate(
        self,
        source_understanding: Union[SourceUnderstanding, dict],
        content_strategy: Union[ContentStrategy, dict],
    ) -> GeneratedContent:
        """
        Synchronously generates multi-platform content from structured source and strategy.
        """
        logger.info("Content Generation Agent synchronous execution started.")
        input_text = self._format_input_prompt(source_understanding, content_strategy)
        messages = [
            SystemMessage(content=self._get_system_prompt()),
            HumanMessage(content=input_text),
        ]
        try:
            response = self.groq_manager.invoke(
                messages,
                temperature=self.temperature,
            )
            raw_content = response.content if hasattr(response, "content") else str(response)
            result = _parse_generation_json(raw_content)
        except Exception as exc:
            sanitized = sanitize_error_message(str(exc))
            logger.error("Groq provider error during content generation: %s", sanitized)
            raise GroqRequestError(f"Groq request failed: {sanitized}", original_error=exc) from exc

        logger.info("Content Generation Agent successfully generated content.")
        return result

    async def agenerate(
        self,
        source_understanding: Union[SourceUnderstanding, dict],
        content_strategy: Union[ContentStrategy, dict],
    ) -> GeneratedContent:
        """
        Asynchronously generates multi-platform content from structured source and strategy.
        Used by async FastAPI endpoints and sequential pipeline.
        """
        logger.info("Content Generation Agent asynchronous execution started.")
        input_text = self._format_input_prompt(source_understanding, content_strategy)
        messages = [
            SystemMessage(content=self._get_system_prompt()),
            HumanMessage(content=input_text),
        ]
        try:
            response = await self.groq_manager.ainvoke(
                messages,
                temperature=self.temperature,
            )
            raw_content = response.content if hasattr(response, "content") else str(response)
            result = _parse_generation_json(raw_content)
        except Exception as exc:
            sanitized = sanitize_error_message(str(exc))
            logger.error("Groq provider error during content generation: %s", sanitized)
            raise GroqRequestError(f"Groq request failed: {sanitized}", original_error=exc) from exc
            sanitized = sanitize_error_message(str(exc))
            logger.error("Groq provider error during content generation: %s", sanitized)
            raise GroqRequestError(f"Groq request failed: {sanitized}", original_error=exc) from exc

        logger.info("Content Generation Agent successfully generated content.")
        return result


# Reusable singleton instance
content_generation_agent = ContentGenerationAgent()

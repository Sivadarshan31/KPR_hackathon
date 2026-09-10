import json
import logging
import re
from typing import Optional, Union
from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.groq_manager import GroqManager, groq_manager as default_groq_manager
from app.llm.exceptions import GroqRequestError, sanitize_error_message
from app.schemas.source_understanding import SourceUnderstanding
from app.schemas.content_strategy import ContentStrategy

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """You are the Content Strategy Agent (Agent 2) for ContentForge.

Your sole responsibility is to formulate a high-impact, multi-platform content strategy based STRICTLY on the structured source understanding provided by Agent 1.

You are planning the content strategy, NOT generating the final published posts.

Strict Operational Guidelines:
1. Grounding & Anti-Hallucination:
   - Base your strategy entirely on the structured information provided in the SourceUnderstanding input.
   - Do NOT introduce unsupported factual claims, fake metrics, unstated dates, or imaginary company achievements.
   - Exact Preservation: If the source understanding states "50,000 images" or "1,250 households" or "2026", preserve these exact figures when referencing them in key messages or angles. Never inflate, round, or alter numbers (e.g., do NOT turn 50,000 into 500,000).
2. Strategic Scope:
   - Define strategic angles, audience segmentation, key messaging, tone, CTAs, and structural outlines.
   - Tailor strategies specifically for the 3 MVP formats:
     a) LinkedIn: Professional thought-leadership, B2B industry context, discussion CTA.
     b) Instagram: Engaging visual storytelling, slide-by-slide carousel concepts, visual direction, community CTA.
     c) Advisory: Executive briefing, high-signal actionable implications, priority rating, clear briefing structure.
3. Strategic Conciseness & Completeness:
   - Keep all field values concise, crisp, and high-signal (1-2 sentences max per field, 3-4 items max per list) so that all platform sections (LinkedIn, Instagram, Advisory) are completely populated.
4. Output Format:
   - You MUST output a single valid, raw JSON object matching the schema below.
   - Do NOT wrap in markdown code blocks or backticks.
   - All fields must be populated.

Target Schema:
{schema}"""
SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE


def _parse_strategy_json(raw_text: str) -> ContentStrategy:
    """Helper to parse and validate ContentStrategy from raw LLM output."""
    cleaned = raw_text.strip()
    # Strip <think> tags if present
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", cleaned).strip()
    # Strip markdown code fencing if present
    if cleaned.startswith("```"):
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if match:
            cleaned = match.group(1).strip()
    try:
        return ContentStrategy.model_validate_json(cleaned)
    except Exception:
        pass

    # Extract JSON object using raw_decode
    fb = cleaned.find("{")
    if fb != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(cleaned[fb:])
            return ContentStrategy.model_validate(obj)
        except Exception:
            pass

    data = json.loads(cleaned)
    return ContentStrategy.model_validate(data)


class ContentStrategyAgent:
    """
    Agent 2: Content Strategy Agent.
    Transforms structured SourceUnderstanding into multi-format content strategies
    (LinkedIn, Instagram, Advisory).
    Decoupled from HTTP frameworks to facilitate direct pipeline composition.
    """

    def __init__(
        self,
        manager: Optional[GroqManager] = None,
        temperature: float = 0.1,
    ):
        self.groq_manager = manager or default_groq_manager
        self.temperature = temperature
        self._schema_json = json.dumps(ContentStrategy.model_json_schema(), indent=2)

    def _get_system_prompt(self) -> str:
        return SYSTEM_PROMPT_TEMPLATE.format(schema=self._schema_json)

    def _format_input_prompt(self, source_understanding: Union[SourceUnderstanding, dict]) -> str:
        """Serializes the structured source understanding into a clear structured prompt."""
        if isinstance(source_understanding, SourceUnderstanding):
            data = source_understanding.model_dump()
        elif isinstance(source_understanding, dict):
            # Validate into schema first to ensure contract adherence
            data = SourceUnderstanding.model_validate(source_understanding).model_dump()
        else:
            raise ValueError("Input must be an instance of SourceUnderstanding or a valid dict.")

        return f"STRUCTURED SOURCE UNDERSTANDING FROM AGENT 1:\n\n{json.dumps(data, indent=2)}"

    def strategize(self, source_understanding: Union[SourceUnderstanding, dict]) -> ContentStrategy:
        """
        Synchronously processes structured source understanding and returns a ContentStrategy.
        """
        logger.info("Content Strategy Agent synchronous execution started.")
        input_text = self._format_input_prompt(source_understanding)
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
            result = _parse_strategy_json(raw_content)
        except Exception as exc:
            sanitized = sanitize_error_message(str(exc))
            logger.error("Groq provider error during content strategy: %s", sanitized)
            raise GroqRequestError(f"Groq request failed: {sanitized}", original_error=exc) from exc

        logger.info("Content Strategy Agent successfully created content strategy.")
        return result

    async def astrategize(self, source_understanding: Union[SourceUnderstanding, dict]) -> ContentStrategy:
        """
        Asynchronously processes structured source understanding and returns a ContentStrategy.
        Used by async FastAPI endpoints and sequential pipeline.
        """
        logger.info("Content Strategy Agent asynchronous execution started.")
        input_text = self._format_input_prompt(source_understanding)
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
            result = _parse_strategy_json(raw_content)
        except Exception as exc:
            sanitized = sanitize_error_message(str(exc))
            logger.error("Groq provider error during content strategy: %s", sanitized)
            raise GroqRequestError(f"Groq request failed: {sanitized}", original_error=exc) from exc

        logger.info("Content Strategy Agent successfully created content strategy.")
        return result


# Reusable singleton instance
content_strategy_agent = ContentStrategyAgent()

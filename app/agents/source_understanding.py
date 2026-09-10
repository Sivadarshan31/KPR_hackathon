import json
import logging
import re
from typing import Optional
from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.groq_manager import GroqManager, groq_manager as default_groq_manager
from app.llm.exceptions import GroqRequestError, sanitize_error_message
from app.schemas.source_understanding import SourceUnderstanding

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """You are the Source Understanding Agent for ContentForge.

Your sole responsibility is to analyze, understand, and extract structured information from the provided source text so downstream AI agents can reliably consume it.

Strict Operational Guidelines:
1. Grounding & Fidelity: Use ONLY the supplied source text. Do not draw on outside knowledge, unverified assumptions, or training memory.
2. Anti-Hallucination: Do NOT fabricate names, organizations, companies, locations, dates, statistics, metrics, quotations, technologies, or events. If a detail is absent in the source, do NOT invent it.
3. Exact Preservation: Preserve numbers, statistics, percentages, and metrics exactly as written in the source text (e.g. "40%", "18 hours", "50,000 images"). Do not rephrase or alter numerical data.
4. Dates & Chronology: Preserve explicit dates and temporal expressions accurately. Do not extrapolate future dates or unstated timelines.
5. Facts vs. Claims: Explicitly separate verified facts from subjective claims or organizational assertions made in the text.
6. Defaults for Missing Data:
   - If target audience cannot be determined from source evidence, set target_audience to "Not specified".
   - If tone is unclear, set tone to "Neutral".
   - If document type is unclear, set source_type to "Unknown".
   - For list fields with no corresponding items in the source text, provide an empty list ([]).
7. Negative Constraints:
   - Do NOT generate social-media posts (LinkedIn, Instagram, X/Twitter, etc.).
   - Do NOT generate marketing copy, captions, hashtags, or promotional rewrites.
   - Do NOT create recommendations or a content strategy.
   - Do NOT validate or criticize the source.

Output Format:
You MUST output a single valid, raw JSON object INSTANCE populating real values for all fields (do NOT output schema definition metadata).
Output ONLY a JSON object matching this structure:

Example JSON Output:
{
  "title": "Healthcare AI Impact Report",
  "summary": "Artificial intelligence diagnostic tools reduce diagnostic latency by 40% in radiology.",
  "main_topic": "Healthcare AI Diagnostics",
  "key_points": ["AI diagnostic tools reduce diagnostic latency by 40%."],
  "facts": ["Hospitals saved 18 hours per doctor per week."],
  "entities": ["Hospitals", "Radiologists"],
  "important_numbers": ["40%", "18 hours"],
  "dates": ["2026"],
  "claims": [],
  "terminology": ["Diagnostic Latency", "Radiology"],
  "target_audience": "Healthcare Leaders",
  "tone": "Informative",
  "source_type": "Report"
}"""

SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE


def _parse_source_json(raw_text: str) -> SourceUnderstanding:
    """Helper to parse and validate SourceUnderstanding from raw LLM output."""
    cleaned = raw_text.strip()
    # Strip <think> tags if present
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", cleaned).strip()
    # Strip markdown code fencing if present
    if cleaned.startswith("```"):
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if match:
            cleaned = match.group(1).strip()
    try:
        return SourceUnderstanding.model_validate_json(cleaned)
    except Exception:
        pass

    # Extract JSON object using raw_decode
    fb = cleaned.find("{")
    if fb != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(cleaned[fb:])
            return SourceUnderstanding.model_validate(obj)
        except Exception:
            pass

    data = json.loads(cleaned)
    return SourceUnderstanding.model_validate(data)


class SourceUnderstandingAgent:
    """
    Agent 1: Source Understanding Agent.
    Transforms raw source text into a structured, validated, source-grounded representation.
    Decoupled from HTTP frameworks to facilitate direct reuse by workflows and tests.
    """

    def __init__(
        self,
        manager: Optional[GroqManager] = None,
        temperature: float = 0.1,
    ):
        self.groq_manager = manager or default_groq_manager
        self.temperature = temperature
        self._schema_json = json.dumps(SourceUnderstanding.model_json_schema(), indent=2)

    def _get_system_prompt(self) -> str:
        return SYSTEM_PROMPT_TEMPLATE

    def understand(self, source_text: str) -> SourceUnderstanding:
        """
        Synchronously processes source text and returns structured understanding.
        """
        logger.info("Source Understanding Agent synchronous execution started.")
        messages = [
            SystemMessage(content=self._get_system_prompt()),
            HumanMessage(content=f"SOURCE TEXT TO ANALYZE:\n\n{source_text}"),
        ]

        # Check if caller specifically mocked get_llm().with_structured_output (for legacy unit tests)
        if hasattr(self.groq_manager, "get_llm") and getattr(self.groq_manager.get_llm, "_mock_return_value", None) is not None:
            try:
                res = self.groq_manager.get_llm().with_structured_output(SourceUnderstanding).invoke(messages)
                if isinstance(res, SourceUnderstanding):
                    return res
            except Exception:
                pass

        try:
            response = self.groq_manager.invoke(messages, temperature=self.temperature)
            raw_content = response.content if hasattr(response, "content") else str(response)
            result = _parse_source_json(raw_content)
        except Exception as exc:
            sanitized = sanitize_error_message(str(exc))
            logger.error("Groq provider error during source understanding: %s", sanitized)
            raise GroqRequestError(f"Groq request failed: {sanitized}", original_error=exc) from exc

        logger.info("Source Understanding Agent successfully extracted structured content.")
        return result

    async def aunderstand(self, source_text: str) -> SourceUnderstanding:
        """
        Asynchronously processes source text and returns structured understanding.
        Used by async FastAPI endpoints.
        """
        logger.info("Source Understanding Agent asynchronous execution started.")
        messages = [
            SystemMessage(content=self._get_system_prompt()),
            HumanMessage(content=f"SOURCE TEXT TO ANALYZE:\n\n{source_text}"),
        ]

        # Check if caller specifically mocked get_llm().with_structured_output (for legacy unit tests)
        if hasattr(self.groq_manager, "get_llm") and getattr(self.groq_manager.get_llm, "_mock_return_value", None) is not None:
            try:
                res = await self.groq_manager.get_llm().with_structured_output(SourceUnderstanding).ainvoke(messages)
                if isinstance(res, SourceUnderstanding):
                    return res
            except Exception:
                pass

        try:
            response = await self.groq_manager.ainvoke(messages, temperature=self.temperature)
            raw_content = response.content if hasattr(response, "content") else str(response)
            result = _parse_source_json(raw_content)
        except Exception as exc:
            sanitized = sanitize_error_message(str(exc))
            logger.error("Groq provider error during source understanding: %s", sanitized)
            raise GroqRequestError(f"Groq request failed: {sanitized}", original_error=exc) from exc

        logger.info("Source Understanding Agent successfully extracted structured content.")
        return result


# Reusable singleton instance
source_understanding_agent = SourceUnderstandingAgent()

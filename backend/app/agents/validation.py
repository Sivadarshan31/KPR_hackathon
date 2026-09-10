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
from app.schemas.validation import ValidationResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """You are the ContentForge Validation Agent (Agent 4).

Your sole responsibility is to evaluate generated content against the source facts and determine whether it is strictly grounded, factually accurate, and safe for publication.

You are a CRITICAL VALIDATOR, not a content generator or marketer.
- Do NOT generate new posts.
- Do NOT alter the facts.
- Evaluate objectively with zero tolerance for factual hallucination.

Strict Operational Guidelines:
1. Grounding & Anti-Hallucination:
   - Check every factual statement in the generated content against the supplied SourceUnderstanding.
   - Flag ANY invented metric, date, organization, person, customer count, revenue, funding, partnership, or award.
   - If the content mentions a figure not in the source (e.g., claiming "$50,000 revenue" when source has no revenue), flag it as a critical issue.
2. Exact Preservation:
   - Ensure numbers, percentages, durations, and dates from the source are preserved accurately (e.g. "40%", "150 students", "2026", "18 hours").
3. Scoring & Pass/Fail Criteria:
   - passed = true ONLY if no unsupported factual claims are present and the score is >= 0.8.
   - If ANY hallucination or altered metric is detected, set passed = false and score <= 0.6.
   - In checks dictionary:
     - grounding_verified: true/false
     - no_hallucinations: true/false
     - numbers_accurate: true/false
     - format_compliance: true/false
4. Actionable Feedback:
   - List explicit, concise points under 'issues'.
   - Provide concrete revision guidance under 'suggestions' so the content generator can fix the issue in the next loop.
5. Output Format:
   - You MUST output a single valid, raw JSON object matching the schema below.
   - Do NOT wrap in markdown code blocks or backticks.

Target Schema:
{schema}"""
SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE


def _parse_validation_json(raw_text: str) -> ValidationResult:
    """Helper to parse and validate ValidationResult from raw LLM output."""
    cleaned = raw_text.strip()
    # Strip <think> tags if present
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", cleaned).strip()
    # Strip markdown code fencing if present
    if cleaned.startswith("```"):
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if match:
            cleaned = match.group(1).strip()
    try:
        return ValidationResult.model_validate_json(cleaned)
    except Exception:
        pass

    # Extract JSON object using raw_decode
    fb = cleaned.find("{")
    if fb != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(cleaned[fb:])
            return ValidationResult.model_validate(obj)
        except Exception:
            pass

    data = json.loads(cleaned)
    return ValidationResult.model_validate(data)


class ValidationAgent:
    """
    Agent 4: Validation Agent.
    Audits generated content for factual fidelity, hallucination freedom, and quality.
    """

    def __init__(
        self,
        manager: Optional[GroqManager] = None,
        temperature: float = 0.1,
    ):
        self.groq_manager = manager or default_groq_manager
        self.temperature = temperature
        self._schema_json = json.dumps(ValidationResult.model_json_schema(), indent=2)

    def _get_system_prompt(self) -> str:
        return SYSTEM_PROMPT_TEMPLATE.format(schema=self._schema_json)

    def _format_input_prompt(
        self,
        source_understanding: Union[SourceUnderstanding, dict],
        generated_content: Union[GeneratedContent, dict],
        content_strategy: Optional[Union[ContentStrategy, dict]] = None,
        user_request: Optional[str] = None,
    ) -> str:
        """Serializes input contexts into a structured validation audit prompt."""
        if isinstance(source_understanding, SourceUnderstanding):
            source_data = source_understanding.model_dump()
        elif isinstance(source_understanding, dict):
            source_data = SourceUnderstanding.model_validate(source_understanding).model_dump()
        else:
            raise ValueError("source_understanding must be an instance of SourceUnderstanding or dict.")

        if isinstance(generated_content, GeneratedContent):
            content_data = generated_content.model_dump()
        elif isinstance(generated_content, dict):
            content_data = GeneratedContent.model_validate(generated_content).model_dump()
        else:
            raise ValueError("generated_content must be an instance of GeneratedContent or dict.")

        strategy_data = None
        if content_strategy:
            if isinstance(content_strategy, ContentStrategy):
                strategy_data = content_strategy.model_dump()
            elif isinstance(content_strategy, dict):
                strategy_data = ContentStrategy.model_validate(content_strategy).model_dump()

        payload = {
            "SOURCE_FACTS_GROUND_TRUTH": source_data,
            "GENERATED_CONTENT_TO_AUDIT": content_data,
            "CONTENT_STRATEGY": strategy_data,
            "USER_REQUEST": user_request,
        }
        return (
            "AUDIT THE GENERATED CONTENT FOR FACTUAL FIDELITY AND QUALITY:\n\n"
            f"{json.dumps(payload, indent=2)}"
        )

    def validate(
        self,
        source_understanding: Union[SourceUnderstanding, dict],
        generated_content: Union[GeneratedContent, dict],
        content_strategy: Optional[Union[ContentStrategy, dict]] = None,
        user_request: Optional[str] = None,
    ) -> ValidationResult:
        """
        Synchronously validates generated content against source ground truth.
        """
        logger.info("Validation Agent synchronous execution started.")
        input_text = self._format_input_prompt(
            source_understanding, generated_content, content_strategy, user_request
        )
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
            result = _parse_validation_json(raw_content)
        except Exception as exc:
            sanitized = sanitize_error_message(str(exc))
            logger.error("Groq provider error during validation: %s", sanitized)
            raise GroqRequestError(f"Groq request failed: {sanitized}", original_error=exc) from exc

        logger.info("Validation completed. Passed: %s, Score: %s", result.passed, result.score)
        return result

    async def avalidate(
        self,
        source_understanding: Union[SourceUnderstanding, dict],
        generated_content: Union[GeneratedContent, dict],
        content_strategy: Optional[Union[ContentStrategy, dict]] = None,
        user_request: Optional[str] = None,
    ) -> ValidationResult:
        """
        Asynchronously validates generated content against source ground truth.
        """
        logger.info("Validation Agent asynchronous execution started.")
        input_text = self._format_input_prompt(
            source_understanding, generated_content, content_strategy, user_request
        )
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
            result = _parse_validation_json(raw_content)
        except Exception as exc:
            sanitized = sanitize_error_message(str(exc))
            logger.error("Groq provider error during validation: %s", sanitized)
            raise GroqRequestError(f"Groq request failed: {sanitized}", original_error=exc) from exc

        logger.info("Validation completed. Passed: %s, Score: %s", result.passed, result.score)
        return result


# Reusable singleton instance
validation_agent = ValidationAgent()

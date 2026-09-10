import json
import logging
import re
from typing import Any, Dict, Optional
from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.groq_manager import GroqManager, groq_manager as default_groq_manager
from app.llm.exceptions import GroqRequestError, sanitize_error_message
from app.schemas.master import MasterDecision

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """You are the ContentForge Master Agent.

Your sole responsibility is to analyze the user's natural language request and current workflow state to determine an intelligent, optimal execution plan for the specialized agents.

Specialized Agents Available:
- Agent 1 (Source Understanding): Analyzes raw source text to extract verified facts, numbers, entities, dates.
- Agent 2 (Content Strategy): Formulates multi-platform strategy (LinkedIn, Instagram, Advisory).
- Agent 3 (Content Generation): Produces platform copy (LinkedIn post, Instagram caption/carousel, Advisory brief).
- Agent 4 (Validation Agent): Audits generated content for factual fidelity and hallucinations.
- Agent 5 (Action / Publishing Agent): Executes preview, export, or publishing (with human approval gate).

Operational Routing Rules:
1. Source Understanding: Set requires_source_understanding = true if new raw source text must be processed and structured facts are not already present in state.
2. Strategy & Generation: Set requires_strategy = true and requires_generation = true if the user requests creating posts, carousels, or advisories.
3. Validation: Whenever new content is generated, set requires_validation = true.
4. Action:
   - If user asks to 'publish', 'post', or 'stage for publication': set requires_action = true, requested_action = 'publish', and requires_human_approval = true.
   - If user asks to 'export' or 'download': set requires_action = true, requested_action = 'export', and requires_human_approval = false.
   - If user asks to 'preview': set requires_action = true, requested_action = 'preview', and requires_human_approval = false.
   - If user only asks to generate without publishing/exporting/previewing: set requires_action = false, requested_action = null, requires_human_approval = false.
5. Platforms: Identify all platforms explicitly mentioned or implied: 'linkedin', 'instagram', 'advisory'.
6. State Awareness:
   - If structured source facts already exist in state, do not re-run Agent 1.
   - If human approval is already granted, reflect this in the decision.
7. Concise Reasoning:
   - In reasoning_summary, provide a single brief operational sentence (e.g., "User requested LinkedIn post generation and validation from source text.").
   - Do NOT engage in lengthy internal deliberations or verbose thinking. Output the target JSON directly.
8. Output Format:
   - You MUST output a single valid, raw JSON object INSTANCE populating real values for all fields.
   - Do NOT output the schema or meta-properties. Output ONLY a JSON object matching this structure:

Example JSON Output:
{
  "intent": "publish_content",
  "requires_source_understanding": true,
  "requires_strategy": true,
  "requires_generation": true,
  "requires_validation": true,
  "requires_action": true,
  "requires_human_approval": true,
  "requested_platforms": ["linkedin"],
  "requested_formats": ["post"],
  "requested_action": "publish",
  "user_request": "Take this report and create a professional LinkedIn post and publish it.",
  "reasoning_summary": "User requested LinkedIn post generation and publishing from raw source text."
}"""


def _parse_decision_json(raw_text: str) -> MasterDecision:
    """Helper to parse and validate MasterDecision from raw LLM output."""
    cleaned = raw_text.strip()
    # Strip <think> tags if present
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", cleaned).strip()
    # Strip markdown code fencing if present
    if cleaned.startswith("```"):
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if match:
            cleaned = match.group(1).strip()
    try:
        return MasterDecision.model_validate_json(cleaned)
    except Exception:
        pass

    # Extract JSON object using raw_decode
    fb = cleaned.find("{")
    if fb != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(cleaned[fb:])
            return MasterDecision.model_validate(obj)
        except Exception:
            pass

    try:
        data = json.loads(cleaned)
        return MasterDecision.model_validate(data)
    except Exception as exc:
        logger.error("Master Agent JSON parse error: %s. Raw text was: %s", exc, raw_text[:500])
        raise exc


class MasterAgent:
    """
    Phase 7: Master Agent.
    Orchestrates user intent parsing, workflow routing, and agent delegation.
    """

    def __init__(
        self,
        manager: Optional[GroqManager] = None,
        temperature: float = 0.1,
    ):
        self.groq_manager = manager or default_groq_manager
        self.temperature = temperature
        self._schema_json = json.dumps(MasterDecision.model_json_schema(), indent=2)

    def _get_system_prompt(self) -> str:
        return SYSTEM_PROMPT_TEMPLATE

    def _format_input_prompt(
        self,
        user_request: str,
        state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Serializes user request and current state into routing analysis prompt."""
        current_state_summary = {}
        if state:
            current_state_summary = {
                "has_source_text": bool(state.get("source_text")),
                "has_source_understanding": bool(state.get("source_understanding")),
                "has_content_strategy": bool(state.get("content_strategy")),
                "has_generated_content": bool(state.get("generated_content")),
                "has_validation_result": bool(state.get("validation_result")),
                "human_approval_status": state.get("human_approval_status", "none"),
                "revision_count": state.get("revision_count", 0),
            }

        payload = {
            "USER_REQUEST": user_request,
            "CURRENT_WORKFLOW_STATE": current_state_summary,
        }
        return (
            "DETERMINE THE OPTIMAL AGENTIC WORKFLOW DECISION FOR THE FOLLOWING REQUEST:\n\n"
            f"{json.dumps(payload, indent=2)}"
        )

    def decide(
        self,
        user_request: str,
        state: Optional[Dict[str, Any]] = None,
    ) -> MasterDecision:
        """
        Synchronously analyzes user request and state to formulate a MasterDecision.
        """
        logger.info("Master Agent synchronous execution started for request: %s", user_request[:60])
        input_text = self._format_input_prompt(user_request, state)
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
            result = _parse_decision_json(raw_content)
        except Exception as exc:
            sanitized = sanitize_error_message(str(exc))
            logger.error("Groq provider error during Master Agent execution: %s", sanitized)
            raise GroqRequestError(f"Groq request failed: {sanitized}", original_error=exc) from exc

        logger.info("Master Agent decision reached: Intent=%s, RequiresAction=%s", result.intent, result.requires_action)
        return result

    async def adecide(
        self,
        user_request: str,
        state: Optional[Dict[str, Any]] = None,
    ) -> MasterDecision:
        """
        Asynchronously analyzes user request and state to formulate a MasterDecision.
        """
        logger.info("Master Agent asynchronous execution started for request: %s", user_request[:60])
        input_text = self._format_input_prompt(user_request, state)
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
            result = _parse_decision_json(raw_content)
        except Exception as exc:
            sanitized = sanitize_error_message(str(exc))
            logger.error("Groq provider error during Master Agent execution: %s", sanitized)
            raise GroqRequestError(f"Groq request failed: {sanitized}", original_error=exc) from exc
        logger.info("Master Agent decision reached: Intent=%s, RequiresAction=%s", result.intent, result.requires_action)
        return result


# Reusable singleton instance
master_agent = MasterAgent()

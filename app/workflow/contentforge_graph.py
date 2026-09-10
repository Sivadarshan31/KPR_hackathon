import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.schemas.source_understanding import SourceUnderstanding
from app.schemas.content_strategy import ContentStrategy
from app.schemas.content_generation import GeneratedContent
from app.schemas.validation import ValidationResult
from app.schemas.action import ActionResult
from app.schemas.master import MasterDecision
from app.workflow.state import ContentForgeState

from app.agents.master_agent import master_agent, MasterAgent
from app.agents.source_understanding import source_understanding_agent, SourceUnderstandingAgent
from app.agents.content_strategy import content_strategy_agent, ContentStrategyAgent
from app.agents.content_generation import (
    content_generation_agent,
    ContentGenerationAgent,
    _parse_generation_json,
)
from app.agents.validation import validation_agent, ValidationAgent
from app.agents.action import action_agent, ActionAgent

from app.llm.groq_manager import GroqManager, groq_manager as default_groq_manager
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

MAX_VALIDATION_ATTEMPTS = 3


# -----------------------------------------------------------------------------
# Node Implementations
# -----------------------------------------------------------------------------

async def master_node(state: ContentForgeState) -> Dict[str, Any]:
    """Phase 7: Analyzes user intent and current state to determine routing."""
    logger.info("[Workflow Node: Master] Evaluating user request.")
    user_request = state.get("user_request", "")
    decision = await master_agent.adecide(user_request, state)
    return {
        "decision": decision,
        "current_stage": "master",
        "status": "processing",
    }


async def source_node(state: ContentForgeState) -> Dict[str, Any]:
    """Agent 1 Node: Extracts grounded facts from source text."""
    logger.info("[Workflow Node: Source Understanding] Processing source material.")
    source_text = state.get("source_text", "")
    if not source_text or not source_text.strip():
        return {
            "status": "failed",
            "error": "Source understanding required but source_text is empty.",
            "current_stage": "source",
        }
    understanding = await source_understanding_agent.aunderstand(source_text)
    dec = state.get("decision")
    is_completed = bool(dec and not (dec.requires_strategy or dec.requires_generation or dec.requires_validation or dec.requires_action))
    return {
        "source_understanding": understanding,
        "current_stage": "source",
        "status": "completed" if is_completed else "processing",
    }


async def strategy_node(state: ContentForgeState) -> Dict[str, Any]:
    """Agent 2 Node: Formulates multi-platform strategy."""
    logger.info("[Workflow Node: Content Strategy] Formulating strategy.")
    su = state.get("source_understanding")
    if not su:
        return {
            "status": "failed",
            "error": "Content strategy required but source_understanding is missing.",
            "current_stage": "strategy",
        }
    strategy = await content_strategy_agent.astrategize(su)
    dec = state.get("decision")
    is_completed = bool(dec and not (dec.requires_generation or dec.requires_validation or dec.requires_action))
    return {
        "content_strategy": strategy,
        "current_stage": "strategy",
        "status": "completed" if is_completed else "processing",
    }


async def generation_node(state: ContentForgeState) -> Dict[str, Any]:
    """Agent 3 Node: Generates platform-tailored copy."""
    logger.info("[Workflow Node: Content Generation] Generating copy.")
    su = state.get("source_understanding")
    cs = state.get("content_strategy")
    if not su:
        return {
            "status": "failed",
            "error": "Content generation required but source_understanding is missing.",
            "current_stage": "generation",
        }
    # If strategy wasn't explicitly generated, formulate a minimal strategy on the fly or pass what exists
    if not cs:
        cs = await content_strategy_agent.astrategize(su)

    content = await content_generation_agent.agenerate(su, cs)
    dec = state.get("decision")
    is_completed = bool(dec and not (dec.requires_validation or dec.requires_action))
    return {
        "content_strategy": cs,
        "generated_content": content,
        "current_stage": "generation",
        "status": "completed" if is_completed else "processing",
    }


async def validation_node(state: ContentForgeState) -> Dict[str, Any]:
    """Agent 4 Node: Audits generated content for fidelity and hallucinations."""
    logger.info("[Workflow Node: Validation] Auditing content.")
    su = state.get("source_understanding")
    gc = state.get("generated_content")
    cs = state.get("content_strategy")
    req = state.get("user_request")

    if not su or not gc:
        return {
            "status": "failed",
            "error": "Validation requires both source_understanding and generated_content.",
            "current_stage": "validation",
        }

    val_result = await validation_agent.avalidate(su, gc, cs, req)
    
    # Update validation history
    history = list(state.get("validation_history", []))
    rev_count = state.get("revision_count", 0)
    history.append({
        "attempt": rev_count + 1,
        "passed": val_result.passed,
        "score": val_result.score,
        "issues": list(val_result.issues),
        "suggestions": list(val_result.suggestions),
    })

    dec = state.get("decision")
    if val_result.passed:
        status = "completed" if not (dec and dec.requires_action) else "processing"
    else:
        status = "validation_failed" if rev_count >= MAX_VALIDATION_ATTEMPTS else "processing"

    return {
        "validation_result": val_result,
        "validation_history": history,
        "current_stage": "validation",
        "status": status,
    }


async def revision_node(state: ContentForgeState) -> Dict[str, Any]:
    """Phase 9 Autonomous Revision Node: Refines copy addressing validator/human feedback."""
    logger.info("[Workflow Node: Revision] Correcting issues detected by validator or user feedback.")
    su = state.get("source_understanding")
    cs = state.get("content_strategy")
    gc = state.get("generated_content")
    val = state.get("validation_result")
    feedback = state.get("human_feedback")

    rev_count = state.get("revision_count", 0) + 1

    # Construct targeted revision prompt
    issues_str = "\n".join(f"- {iss}" for iss in (val.issues if val else []))
    suggestions_str = "\n".join(f"- {sug}" for sug in (val.suggestions if val else []))

    revision_instructions = f"""You are the ContentForge Content Generation Agent performing an autonomous revision.
PREVIOUS ISSUES DETECTED:
{issues_str if issues_str else 'None'}

VALIDATOR SUGGESTIONS:
{suggestions_str if suggestions_str else 'None'}

USER FEEDBACK:
{feedback if feedback else 'None'}

CRITICAL: Fix all identified issues. Ensure strict grounding in source facts. Never invent unstated metrics, dates, revenue, or claims.
Output the complete corrected GeneratedContent JSON."""

    schema_json = json.dumps(GeneratedContent.model_json_schema(), indent=2)
    messages = [
        SystemMessage(content=f"Revise the content matching the target JSON schema below:\n\n{schema_json}"),
        HumanMessage(content=f"{revision_instructions}\n\nORIGINAL FACTS:\n{json.dumps(su.model_dump() if su else {}, indent=2)}\n\nPREVIOUS CONTENT:\n{json.dumps(gc.model_dump() if gc else {}, indent=2)}"),
    ]

    try:
        response = await default_groq_manager.ainvoke(messages)
        raw_text = response.content if hasattr(response, "content") else str(response)
        new_content = _parse_generation_json(raw_text)
    except Exception as exc:
        logger.warning("Revision generation fallback: %s", exc)
        new_content = gc  # keep previous if revision call failed

    return {
        "generated_content": new_content,
        "revision_count": rev_count,
        "current_stage": "revision",
        "status": "processing",
        "human_feedback": "",  # clear feedback once consumed
    }


async def approval_checkpoint_node(state: ContentForgeState) -> Dict[str, Any]:
    """Phase 9 Human Approval Checkpoint Node: Pauses execution for human approval before publishing."""
    logger.info("[Workflow Node: Human Approval Checkpoint] Setting state to waiting_for_approval.")
    return {
        "status": "waiting_for_approval",
        "current_stage": "human_approval",
        "human_approval_required": True,
        "human_approval_status": "pending",
    }


async def action_node(state: ContentForgeState) -> Dict[str, Any]:
    """Agent 5 Node: Executes preview, export, or publish action on validated content."""
    logger.info("[Workflow Node: Action Agent] Executing action.")
    gc = state.get("generated_content")
    val = state.get("validation_result")
    dec = state.get("decision")

    action_name = (dec.requested_action if dec and dec.requested_action else "preview").lower()
    platform = dec.requested_platforms[0] if (dec and dec.requested_platforms) else "all"

    val_status = "PASS" if (val and val.passed) else "FAIL"
    val_issues = list(val.issues) if val else []

    action_result = await action_agent.aexecute(
        action=action_name,
        platform=platform,
        content=gc,
        validation_status=val_status,
        validation_issues=val_issues,
    )

    return {
        "action_result": action_result,
        "current_stage": "action",
        "status": "completed",
    }


# -----------------------------------------------------------------------------
# Conditional Routing Functions
# -----------------------------------------------------------------------------

def route_after_master(state: ContentForgeState) -> str:
    dec: Optional[MasterDecision] = state.get("decision")
    if not dec:
        return "end"

    if dec.requires_source_understanding and not state.get("source_understanding"):
        return "source"
    elif dec.requires_strategy and not state.get("content_strategy"):
        return "strategy"
    elif dec.requires_generation:
        return "generation"
    elif dec.requires_action:
        return "action"
    return "end"


def route_after_source(state: ContentForgeState) -> str:
    if state.get("status") == "failed":
        return "end"
    dec: Optional[MasterDecision] = state.get("decision")
    if dec and dec.requires_strategy:
        return "strategy"
    return "end"


def route_after_strategy(state: ContentForgeState) -> str:
    if state.get("status") == "failed":
        return "end"
    dec: Optional[MasterDecision] = state.get("decision")
    if dec and dec.requires_generation:
        return "generation"
    return "end"


def route_after_generation(state: ContentForgeState) -> str:
    if state.get("status") == "failed":
        return "end"
    dec: Optional[MasterDecision] = state.get("decision")
    if dec and dec.requires_validation:
        return "validation"
    return "end"


def route_after_validation(state: ContentForgeState) -> str:
    if state.get("status") == "failed":
        return "end"

    val: Optional[ValidationResult] = state.get("validation_result")
    dec: Optional[MasterDecision] = state.get("decision")
    rev_count = state.get("revision_count", 0)

    # If validation passed:
    if val and val.passed:
        if dec and dec.requires_action:
            if dec.requires_human_approval:
                return "approval_checkpoint"
            return "action"
        return "end"

    # If validation failed: Check max attempts
    if rev_count < MAX_VALIDATION_ATTEMPTS:
        logger.info("Validation failed (attempt %d/%d). Routing to revision.", rev_count + 1, MAX_VALIDATION_ATTEMPTS)
        return "revision"

    logger.warning("Validation failed after reaching max attempts (%d). Halting.", MAX_VALIDATION_ATTEMPTS)
    state["status"] = "validation_failed"
    return "end"


def route_after_approval(state: ContentForgeState) -> str:
    approval_status = state.get("human_approval_status", "none")
    if approval_status == "approved":
        return "action"
    elif approval_status == "rejected":
        return "revision"
    return "wait"


# -----------------------------------------------------------------------------
# Graph Construction & Workflow Controller
# -----------------------------------------------------------------------------

def build_contentforge_graph(checkpointer: Optional[MemorySaver] = None) -> StateGraph:
    """Builds and compiles the full ContentForge LangGraph workflow."""
    builder = StateGraph(ContentForgeState)

    # Register all nodes
    builder.add_node("master_node", master_node)
    builder.add_node("source_node", source_node)
    builder.add_node("strategy_node", strategy_node)
    builder.add_node("generation_node", generation_node)
    builder.add_node("validation_node", validation_node)
    builder.add_node("revision_node", revision_node)
    builder.add_node("approval_checkpoint_node", approval_checkpoint_node)
    builder.add_node("action_node", action_node)

    # Connect edges
    builder.add_edge(START, "master_node")

    builder.add_conditional_edges(
        "master_node",
        route_after_master,
        {
            "source": "source_node",
            "strategy": "strategy_node",
            "generation": "generation_node",
            "action": "action_node",
            "end": END,
        },
    )

    builder.add_conditional_edges(
        "source_node",
        route_after_source,
        {
            "strategy": "strategy_node",
            "end": END,
        },
    )

    builder.add_conditional_edges(
        "strategy_node",
        route_after_strategy,
        {
            "generation": "generation_node",
            "end": END,
        },
    )

    builder.add_conditional_edges(
        "generation_node",
        route_after_generation,
        {
            "validation": "validation_node",
            "end": END,
        },
    )

    builder.add_conditional_edges(
        "validation_node",
        route_after_validation,
        {
            "approval_checkpoint": "approval_checkpoint_node",
            "action": "action_node",
            "revision": "revision_node",
            "end": END,
        },
    )

    # Revision feeds directly back to validation for iterative verification
    builder.add_edge("revision_node", "validation_node")

    # Approval checkpoint conditionally routes on human approval / rejection
    builder.add_conditional_edges(
        "approval_checkpoint_node",
        route_after_approval,
        {
            "action": "action_node",
            "revision": "revision_node",
            "wait": END,
        },
    )

    builder.add_edge("action_node", END)

    memory = checkpointer or MemorySaver()
    compiled_graph = builder.compile(checkpointer=memory)
    return compiled_graph


class ContentForgeWorkflow:
    """
    High-level orchestration manager controlling LangGraph execution,
    thread persistence, and human approval resumption.
    """

    def __init__(self, checkpointer: Optional[MemorySaver] = None):
        self.checkpointer = checkpointer or MemorySaver()
        self.graph = build_contentforge_graph(self.checkpointer)

    async def arun(
        self,
        user_request: str,
        source_text: Optional[str] = None,
        workflow_id: Optional[str] = None,
    ) -> ContentForgeState:
        """Runs the workflow for a new request."""
        w_id = workflow_id or f"cf-workflow-{uuid.uuid4().hex[:10]}"
        config = {"configurable": {"thread_id": w_id}}

        initial_state: ContentForgeState = {
            "workflow_id": w_id,
            "user_request": user_request,
            "source_text": source_text or "",
            "revision_count": 0,
            "validation_history": [],
            "human_approval_required": False,
            "human_approval_status": "none",
            "status": "processing",
            "current_stage": "master",
        }

        logger.info("Executing ContentForge workflow [ID: %s] async.", w_id)
        final_state = await self.graph.ainvoke(initial_state, config)
        return final_state

    def run(
        self,
        user_request: str,
        source_text: Optional[str] = None,
        workflow_id: Optional[str] = None,
    ) -> ContentForgeState:
        """Synchronously runs the workflow for a new request."""
        import asyncio
        return asyncio.run(self.arun(user_request, source_text, workflow_id))

    async def aapprove(self, workflow_id: str) -> ContentForgeState:
        """Approves a paused publishing workflow and resumes execution."""
        config = {"configurable": {"thread_id": workflow_id}}
        saved = self.graph.get_state(config)
        if not saved or not saved.values:
            raise ValueError(f"Workflow ID '{workflow_id}' not found.")

        current_status = saved.values.get("status")
        if current_status == "completed":
            logger.info("Workflow '%s' is already completed. Idempotent return.", workflow_id)
            return saved.values

        if current_status != "waiting_for_approval":
            raise ValueError(f"Workflow '{workflow_id}' is in status '{current_status}', not 'waiting_for_approval'.")

        # Update state with approval
        self.graph.update_state(
            config,
            {
                "human_approval_status": "approved",
                "status": "processing",
                "current_stage": "action",
            },
            as_node="approval_checkpoint_node",
        )

        # Resume execution
        logger.info("Resuming approved workflow '%s'.", workflow_id)
        resumed_state = await self.graph.ainvoke(None, config)
        return resumed_state

    async def areject(self, workflow_id: str, feedback: str) -> ContentForgeState:
        """Rejects content with user feedback, triggering revision."""
        config = {"configurable": {"thread_id": workflow_id}}
        saved = self.graph.get_state(config)
        if not saved or not saved.values:
            raise ValueError(f"Workflow ID '{workflow_id}' not found.")

        current_status = saved.values.get("status")
        if current_status != "waiting_for_approval":
            raise ValueError(f"Workflow '{workflow_id}' is in status '{current_status}', not 'waiting_for_approval'.")

        # Update state with rejection and feedback
        self.graph.update_state(
            config,
            {
                "human_approval_status": "rejected",
                "human_feedback": feedback,
                "status": "processing",
                "current_stage": "revision",
            },
            as_node="approval_checkpoint_node",
        )

        # Resume execution into revision node
        logger.info("Resuming rejected workflow '%s' with feedback: %s", workflow_id, feedback[:60])
        resumed_state = await self.graph.ainvoke(None, config)
        return resumed_state

    def get_state(self, workflow_id: str) -> Optional[ContentForgeState]:
        """Retrieves stored state for a given workflow ID."""
        config = {"configurable": {"thread_id": workflow_id}}
        saved = self.graph.get_state(config)
        if not saved or not saved.values:
            return None
        return saved.values


# Reusable singleton instance
contentforge_workflow = ContentForgeWorkflow()

import logging
from fastapi import APIRouter, HTTPException, status

from app.schemas.master import (
    MasterRunRequest,
    MasterRunResponse,
    MasterApproveRequest,
    MasterRejectRequest,
    MasterStatusResponse,
)
from app.workflow.contentforge_graph import contentforge_workflow
from app.llm.exceptions import GroqConfigurationError, GroqRequestError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/master", tags=["Master Orchestration"])


def _format_master_response(state: dict) -> MasterRunResponse:
    """Helper to convert workflow state dict into typed MasterRunResponse."""
    status_str = state.get("status", "completed")
    stage_str = state.get("current_stage", "master")
    approval_req = bool(state.get("human_approval_required")) and state.get("human_approval_status") == "pending"

    if status_str == "waiting_for_approval":
        msg = "Content validated and staged. Paused waiting for explicit human approval before publishing."
    elif status_str == "validation_failed":
        msg = "Content validation failed after reaching maximum revision attempts."
    elif status_str == "failed":
        msg = f"Workflow failed: {state.get('error', 'Unknown error')}"
    else:
        msg = "Workflow execution completed successfully."

    return MasterRunResponse(
        workflow_id=state.get("workflow_id", "unknown"),
        status=status_str,
        current_stage=stage_str,
        decision=state["decision"],
        source_understanding=state.get("source_understanding"),
        content_strategy=state.get("content_strategy"),
        generated_content=state.get("generated_content"),
        validation=state.get("validation_result"),
        action_result=state.get("action_result"),
        approval_required=approval_req,
        revision_count=state.get("revision_count", 0),
        validation_history=state.get("validation_history", []),
        message=msg,
    )


@router.post(
    "/run",
    response_model=MasterRunResponse,
    status_code=status.HTTP_200_OK,
    summary="Master Agent: Run Orchestrated Workflow",
    description="Understands user intent, dynamically routes through specialized agents (1-5), manages validation loops, and pauses for human approval when publishing.",
)
async def run_master_workflow(
    request: MasterRunRequest,
) -> MasterRunResponse:
    """
    Main orchestration entry point:
    1. Master Agent determines routing decision.
    2. LangGraph state engine orchestrates Agent 1 -> Agent 2 -> Agent 3 -> Agent 4.
    3. If validation fails, automatically enters revision loop (up to 3 times).
    4. If publishing is requested, pauses at waiting_for_approval for human gate.
    5. Returns unified workflow response.
    """
    try:
        final_state = await contentforge_workflow.arun(
            user_request=request.user_request,
            source_text=request.source_text,
            workflow_id=request.workflow_id,
        )
        return _format_master_response(final_state)
    except GroqConfigurationError as cfg_err:
        logger.error("Master workflow failed due to Groq configuration: %s", cfg_err)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Groq service unavailable: No Groq API keys configured. Please configure GROQ_API_KEY_1 in .env.",
        )
    except GroqRequestError as req_err:
        err_str = str(req_err).lower()
        logger.error("Master workflow provider error: %s", req_err)
        if "429" in err_str or "rate limit" in err_str or "quota" in err_str:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Groq rate limit exceeded across configured keys. Please retry later.",
            )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Groq provider error encountered during master workflow execution.",
        )
    except ValueError as val_err:
        logger.warning("Master workflow validation error: %s", val_err)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err),
        )
    except Exception as exc:
        logger.error("Unexpected error in master workflow: %s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error executing master workflow.",
        )


@router.get(
    "/status/{workflow_id}",
    response_model=MasterStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Master Agent: Query Workflow Status",
    description="Inspects the persisted state, current stage, and approval requirement for a given workflow ID.",
)
def get_workflow_status(
    workflow_id: str,
) -> MasterStatusResponse:
    """
    Returns current status and checkpoints for a given workflow thread.
    """
    state = contentforge_workflow.get_state(workflow_id)
    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow ID '{workflow_id}' not found.",
        )

    status_str = state.get("status", "unknown")
    approval_req = bool(state.get("human_approval_required")) and state.get("human_approval_status") == "pending"

    return MasterStatusResponse(
        workflow_id=workflow_id,
        status=status_str,
        current_stage=state.get("current_stage", "unknown"),
        approval_required=approval_req,
        human_approval_status=state.get("human_approval_status", "none"),
        content=state.get("generated_content"),
        validation=state.get("validation_result"),
        action_result=state.get("action_result"),
        revision_count=state.get("revision_count", 0),
        message=f"Workflow is currently in '{status_str}' status at stage '{state.get('current_stage')}'.",
    )


@router.post(
    "/approve",
    response_model=MasterRunResponse,
    status_code=status.HTTP_200_OK,
    summary="Master Agent: Approve Publishing Action",
    description="Explicit human approval for a paused workflow. Resumes execution and invokes Agent 5 (Action Agent).",
)
async def approve_workflow(
    request: MasterApproveRequest,
) -> MasterRunResponse:
    """
    Approves and resumes a publishing workflow currently paused at 'waiting_for_approval'.
    """
    try:
        resumed_state = await contentforge_workflow.aapprove(request.workflow_id)
        return _format_master_response(resumed_state)
    except ValueError as val_err:
        err_msg = str(val_err)
        logger.warning("Approval error: %s", err_msg)
        if "not found" in err_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=err_msg,
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=err_msg,
        )
    except Exception as exc:
        logger.error("Unexpected error approving workflow: %s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error processing workflow approval.",
        )


@router.post(
    "/reject",
    response_model=MasterRunResponse,
    status_code=status.HTTP_200_OK,
    summary="Master Agent: Reject Content with Feedback",
    description="Rejects content with user feedback. Prevents publishing and routes into revision and re-validation.",
)
async def reject_workflow(
    request: MasterRejectRequest,
) -> MasterRunResponse:
    """
    Rejects content for a paused publishing workflow and triggers targeted revision.
    """
    try:
        resumed_state = await contentforge_workflow.areject(
            workflow_id=request.workflow_id,
            feedback=request.feedback,
        )
        return _format_master_response(resumed_state)
    except ValueError as val_err:
        err_msg = str(val_err)
        logger.warning("Rejection error: %s", err_msg)
        if "not found" in err_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=err_msg,
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=err_msg,
        )
    except Exception as exc:
        logger.error("Unexpected error rejecting workflow: %s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error processing workflow rejection.",
        )

import logging
from fastapi import APIRouter, HTTPException, status

from app.schemas.action import ActionRequest, ActionResult
from app.agents.action import action_agent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agents", tags=["Agents"])


@router.post(
    "/action",
    response_model=ActionResult,
    status_code=status.HTTP_200_OK,
    summary="Agent 5: Action / Publishing Agent",
    description="Executes requested content action ('preview', 'export', 'publish'). Enforces strict server-side validation gating.",
)
async def process_action(
    request: ActionRequest,
) -> ActionResult:
    """
    HTTP endpoint for Agent 5 (Action Agent).
    Executes preview, export, or publish workflows on validated content.
    """
    try:
        result = await action_agent.aexecute(
            action=request.action,
            platform=request.platform,
            content=request.content,
            validation_status=request.validation_status,
            validation_issues=request.validation_issues,
        )
        return result
    except ValueError as val_err:
        logger.warning("Action validation error: %s", val_err)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err),
        )
    except Exception as exc:
        logger.error("Unexpected error during action execution: %s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error executing action.",
        )

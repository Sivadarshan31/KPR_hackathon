import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, status

from app.schemas.pipeline import (
    PipelineRequest,
    PipelineResponse,
    PipelineGenerationResponse,
)
from app.schemas.master import (
    MasterRunRequest,
    MasterRunResponse,
    MasterStatusResponse,
)
from app.services.content_pipeline import content_pipeline
from app.api.master import _format_master_response
from app.llm.exceptions import GroqConfigurationError, GroqRequestError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/pipeline", tags=["Pipeline"])


@router.post(
    "/run",
    response_model=MasterRunResponse,
    status_code=status.HTTP_200_OK,
    summary="Unified Pipeline: Run In-Memory Workflow",
    description="Executes the unified in-memory ContentForge workflow. Returns workflow_id, current_stage, status, and generated outputs.",
)
async def run_unified_pipeline(
    request: MasterRunRequest,
) -> MasterRunResponse:
    """
    Unified high-level entry point for executing the in-memory ContentForge workflow.
    """
    try:
        final_state = await content_pipeline.execute_workflow(
            user_request=request.user_request,
            source_text=request.source_text,
            workflow_id=request.workflow_id,
        )
        return _format_master_response(final_state)
    except GroqConfigurationError as cfg_err:
        logger.error("Pipeline execution failed due to Groq configuration: %s", cfg_err)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Groq service unavailable: No Groq API keys configured.",
        )
    except GroqRequestError as req_err:
        err_str = str(req_err).lower()
        logger.error("Pipeline execution failed during provider call: %s", req_err)
        if "429" in err_str or "rate limit" in err_str or "quota" in err_str:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Groq rate limit exceeded across configured keys.",
            )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Groq provider error encountered during pipeline execution.",
        )
    except ValueError as val_err:
        logger.warning("Pipeline validation error: %s", val_err)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err),
        )
    except Exception as exc:
        logger.error("Unexpected error during pipeline execution: %s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error executing content pipeline.",
        )


@router.get(
    "/{workflow_id}",
    response_model=MasterStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Query Workflow Status",
    description="Queries the thread-isolated in-memory status and outputs for a given workflow_id.",
)
def get_pipeline_status(
    workflow_id: str,
) -> MasterStatusResponse:
    """
    Retrieves current state and status for a given workflow_id.
    """
    state = content_pipeline.get_workflow_status(workflow_id)
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
    "/source-to-strategy",
    response_model=PipelineResponse,
    status_code=status.HTTP_200_OK,
    summary="Sequential Pipeline: Agent 1 -> Agent 2",
    description="Connects Agent 1 (Source Understanding) directly to Agent 2 (Content Strategy) in a sequential pipeline.",
)
async def run_source_to_strategy_pipeline(
    request: PipelineRequest,
) -> PipelineResponse:
    try:
        response = await content_pipeline.arun(request.source_text)
        return response
    except GroqConfigurationError as cfg_err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Groq service unavailable: No Groq API keys configured.",
        )
    except GroqRequestError as req_err:
        err_str = str(req_err).lower()
        if "429" in err_str or "rate limit" in err_str or "quota" in err_str:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Groq rate limit exceeded across configured keys.",
            )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Groq provider error encountered during pipeline execution.",
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err),
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error executing content pipeline.",
        )


@router.post(
    "/content-generation",
    response_model=PipelineGenerationResponse,
    status_code=status.HTTP_200_OK,
    summary="Full Pipeline: Agent 1 -> Agent 2 -> Agent 3",
    description="Executes the end-to-end pipeline: Source Understanding -> Strategy -> Generation.",
)
async def run_full_content_generation_pipeline(
    request: PipelineRequest,
) -> PipelineGenerationResponse:
    try:
        response = await content_pipeline.arun_full(request.source_text)
        return response
    except GroqConfigurationError as cfg_err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Groq service unavailable: No Groq API keys configured.",
        )
    except GroqRequestError as req_err:
        err_str = str(req_err).lower()
        if "429" in err_str or "rate limit" in err_str or "quota" in err_str:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Groq rate limit exceeded across configured keys.",
            )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Groq provider error encountered during full pipeline execution.",
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err),
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error executing full content pipeline.",
        )

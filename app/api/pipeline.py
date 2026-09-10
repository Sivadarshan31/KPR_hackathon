import logging
from fastapi import APIRouter, HTTPException, status

from app.schemas.pipeline import (
    PipelineRequest,
    PipelineResponse,
    PipelineGenerationResponse,
)
from app.services.content_pipeline import content_pipeline
from app.llm.exceptions import GroqConfigurationError, GroqRequestError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/pipeline", tags=["Pipeline"])


@router.post(
    "/source-to-strategy",
    response_model=PipelineResponse,
    status_code=status.HTTP_200_OK,
    summary="Sequential Pipeline: Agent 1 -> Agent 2",
    description="Connects Agent 1 (Source Understanding) directly to Agent 2 (Content Strategy) in a sequential pipeline, returning both structured outputs.",
)
async def run_source_to_strategy_pipeline(
    request: PipelineRequest,
) -> PipelineResponse:
    """
    Executes the connected pipeline:
    1. Agent 1 extracts structured facts from raw source text.
    2. Agent 2 formulates multi-platform strategy based strictly on Agent 1's structured output.
    3. Returns both structured results in a single response.
    """
    try:
        response = await content_pipeline.arun(request.source_text)
        return response
    except GroqConfigurationError as cfg_err:
        logger.error("Pipeline execution failed due to missing Groq configuration: %s", cfg_err)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Groq service unavailable: No Groq API keys configured. Please configure GROQ_API_KEY_1 in .env.",
        )
    except GroqRequestError as req_err:
        err_str = str(req_err).lower()
        logger.error("Pipeline execution failed during agent execution: %s", req_err)
        if "429" in err_str or "rate limit" in err_str or "quota" in err_str:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Groq rate limit exceeded across configured keys. Please retry later.",
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


@router.post(
    "/content-generation",
    response_model=PipelineGenerationResponse,
    status_code=status.HTTP_200_OK,
    summary="Full Pipeline: Agent 1 -> Agent 2 -> Agent 3",
    description="Executes the end-to-end pipeline: Source Understanding (Agent 1) -> Content Strategy (Agent 2) -> Multi-Format Content Generation (Agent 3).",
)
async def run_full_content_generation_pipeline(
    request: PipelineRequest,
) -> PipelineGenerationResponse:
    """
    Executes the complete ContentForge generation flow:
    1. Agent 1 analyzes source text and extracts structured facts, numbers, entities, and dates.
    2. Agent 2 constructs multi-platform content strategy tailored for LinkedIn, Instagram, and Advisory.
    3. Agent 3 generates platform-ready content using Agent 1 facts and Agent 2 strategy.
    4. Returns all three structured outputs in a single unified response.
    """
    try:
        response = await content_pipeline.arun_full(request.source_text)
        return response
    except GroqConfigurationError as cfg_err:
        logger.error("Full pipeline execution failed due to missing Groq configuration: %s", cfg_err)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Groq service unavailable: No Groq API keys configured. Please configure GROQ_API_KEY_1 in .env.",
        )
    except GroqRequestError as req_err:
        err_str = str(req_err).lower()
        logger.error("Full pipeline execution failed during agent execution: %s", req_err)
        if "429" in err_str or "rate limit" in err_str or "quota" in err_str:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Groq rate limit exceeded across configured keys. Please retry later.",
            )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Groq provider error encountered during full pipeline execution.",
        )
    except ValueError as val_err:
        logger.warning("Full pipeline validation error: %s", val_err)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err),
        )
    except Exception as exc:
        logger.error("Unexpected error during full pipeline execution: %s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error executing full content pipeline.",
        )

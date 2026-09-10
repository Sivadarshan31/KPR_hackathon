import logging
from fastapi import APIRouter, HTTPException, status

from app.schemas.source_understanding import (
    SourceUnderstandingRequest,
    SourceUnderstanding,
)
from app.agents.source_understanding import source_understanding_agent
from app.llm.exceptions import GroqConfigurationError, GroqRequestError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agents", tags=["Agents"])


@router.post(
    "/source-understanding",
    response_model=SourceUnderstanding,
    status_code=status.HTTP_200_OK,
    summary="Agent 1: Source Understanding",
    description="Analyzes raw source text, extracts facts, entities, numbers, dates, claims, and returns a structured representation.",
)
async def process_source_understanding(
    request: SourceUnderstandingRequest,
) -> SourceUnderstanding:
    """
    HTTP endpoint for Agent 1 (Source Understanding Agent).
    Accepts plain source text and returns structured, source-grounded information.
    """
    try:
        result = await source_understanding_agent.aunderstand(request.source_text)
        return result
    except GroqConfigurationError as cfg_err:
        logger.error("Source understanding failed due to missing Groq configuration.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Groq service unavailable: No Groq API keys configured. Please configure GROQ_API_KEY_1 in .env.",
        )
    except GroqRequestError as req_err:
        err_str = str(req_err).lower()
        logger.error("Source understanding failed during Groq provider execution.")
        if "429" in err_str or "rate limit" in err_str or "quota" in err_str:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Groq rate limit exceeded across configured keys. Please retry later.",
            )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Groq provider error encountered during source understanding.",
        )
    except Exception as exc:
        logger.error("Unexpected error during source understanding: %s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error processing source text.",
        )

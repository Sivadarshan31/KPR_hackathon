import logging
from fastapi import APIRouter, HTTPException, status

from app.schemas.content_generation import (
    ContentGenerationRequest,
    GeneratedContent,
)
from app.agents.content_generation import content_generation_agent
from app.llm.exceptions import GroqConfigurationError, GroqRequestError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agents", tags=["Agents"])


@router.post(
    "/content-generation",
    response_model=GeneratedContent,
    status_code=status.HTTP_200_OK,
    summary="Agent 3: Content Generation",
    description="Receives structured source understanding (Agent 1) and content strategy (Agent 2) to generate multi-format content (LinkedIn, Instagram, Advisory).",
)
async def process_content_generation(
    request: ContentGenerationRequest,
) -> GeneratedContent:
    """
    HTTP endpoint for Agent 3 (Content Generation Agent).
    Accepts structured source understanding and structured content strategy,
    generating grounded, platform-tailored copy.
    """
    try:
        result = await content_generation_agent.agenerate(
            request.source_understanding,
            request.content_strategy,
        )
        return result
    except GroqConfigurationError as cfg_err:
        logger.error("Content generation failed due to missing Groq configuration: %s", cfg_err)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Groq service unavailable: No Groq API keys configured. Please configure GROQ_API_KEY_1 in .env.",
        )
    except GroqRequestError as req_err:
        err_str = str(req_err).lower()
        logger.error("Content generation failed during Groq provider execution: %s", req_err)
        if "429" in err_str or "rate limit" in err_str or "quota" in err_str:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Groq rate limit exceeded across configured keys. Please retry later.",
            )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Groq provider error encountered during content generation.",
        )
    except ValueError as val_err:
        logger.warning("Content generation validation error: %s", val_err)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err),
        )
    except Exception as exc:
        logger.error("Unexpected error during content generation: %s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error processing content generation.",
        )

import logging
from fastapi import APIRouter, HTTPException, status

from app.schemas.content_strategy import (
    ContentStrategyRequest,
    ContentStrategy,
)
from app.agents.content_strategy import content_strategy_agent
from app.llm.exceptions import GroqConfigurationError, GroqRequestError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agents", tags=["Agents"])


@router.post(
    "/content-strategy",
    response_model=ContentStrategy,
    status_code=status.HTTP_200_OK,
    summary="Agent 2: Content Strategy",
    description="Receives structured source understanding from Agent 1 and generates a multi-format content strategy (LinkedIn, Instagram, Advisory).",
)
async def process_content_strategy(
    request: ContentStrategyRequest,
) -> ContentStrategy:
    """
    HTTP endpoint for Agent 2 (Content Strategy Agent).
    Accepts structured source understanding and returns a platform-tailored content strategy.
    """
    try:
        result = await content_strategy_agent.astrategize(request.source_understanding)
        return result
    except GroqConfigurationError as cfg_err:
        logger.error("Content strategy failed due to missing Groq configuration.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Groq service unavailable: No Groq API keys configured. Please configure GROQ_API_KEY_1 in .env.",
        )
    except GroqRequestError as req_err:
        err_str = str(req_err).lower()
        logger.error("Content strategy failed during Groq provider execution.")
        if "429" in err_str or "rate limit" in err_str or "quota" in err_str:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Groq rate limit exceeded across configured keys. Please retry later.",
            )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Groq provider error encountered during content strategy generation.",
        )
    except Exception as exc:
        logger.error("Unexpected error during content strategy: %s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error processing content strategy.",
        )

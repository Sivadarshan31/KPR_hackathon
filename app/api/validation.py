import logging
from fastapi import APIRouter, HTTPException, status

from app.schemas.validation import ValidationRequest, ValidationResult
from app.agents.validation import validation_agent
from app.llm.exceptions import GroqConfigurationError, GroqRequestError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agents", tags=["Agents"])


@router.post(
    "/validation",
    response_model=ValidationResult,
    status_code=status.HTTP_200_OK,
    summary="Agent 4: Validation Agent",
    description="Audits generated content against source ground truth. Flags hallucinations, verifies numbers, and computes quality score.",
)
async def process_validation(
    request: ValidationRequest,
) -> ValidationResult:
    """
    HTTP endpoint for Agent 4 (Validation Agent).
    Accepts source understanding and generated copy, returning an audit report.
    """
    try:
        result = await validation_agent.avalidate(
            source_understanding=request.source_understanding,
            generated_content=request.generated_content,
            content_strategy=request.content_strategy,
            user_request=request.user_request,
        )
        return result
    except GroqConfigurationError as cfg_err:
        logger.error("Validation failed due to missing Groq configuration: %s", cfg_err)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Groq service unavailable: No Groq API keys configured. Please configure GROQ_API_KEY_1 in .env.",
        )
    except GroqRequestError as req_err:
        err_str = str(req_err).lower()
        logger.error("Validation failed during Groq provider execution: %s", req_err)
        if "429" in err_str or "rate limit" in err_str or "quota" in err_str:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Groq rate limit exceeded across configured keys. Please retry later.",
            )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Groq provider error encountered during content validation.",
        )
    except ValueError as val_err:
        logger.warning("Validation input error: %s", val_err)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err),
        )
    except Exception as exc:
        logger.error("Unexpected error during validation: %s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error processing validation.",
        )

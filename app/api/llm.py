import logging
from fastapi import APIRouter, HTTPException, status
from app.schemas.llm import LLMTestRequest, LLMTestResponse
from app.llm.groq_manager import groq_manager
from app.llm.exceptions import GroqConfigurationError, GroqRequestError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/llm", tags=["LLM"])


@router.post(
    "/test",
    response_model=LLMTestResponse,
    status_code=status.HTTP_200_OK,
    summary="Test Groq LLM Connectivity",
    description="Validates Groq and LangChain integration. Accepts a prompt message and returns the LLM response.",
)
async def test_llm_connection(request: LLMTestRequest) -> LLMTestResponse:
    """
    Test endpoint verifying connectivity to Groq API via LangChain ChatGroq.
    Does not expose internal keys or credentials.
    """
    if not groq_manager.is_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Groq service unavailable: No Groq API keys configured. Please configure GROQ_API_KEY_1 in .env.",
        )

    try:
        response_text = await groq_manager.generate_response(request.message)
        return LLMTestResponse(response=response_text)
    except GroqConfigurationError as cfg_err:
        logger.error("Groq configuration error during LLM invocation")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Groq configuration error: No valid Groq API keys available.",
        )
    except ValueError as val_err:
        logger.error("Value error during LLM invocation")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Groq configuration error: No valid Groq API keys available.",
        )
    except Exception as exc:
        err_str = str(exc).lower()
        logger.error("Groq request failed with an external exception")

        if "401" in err_str or "unauthorized" in err_str or "invalid api key" in err_str:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Groq authentication failed.",
            )
        elif "429" in err_str or "rate limit" in err_str or "quota" in err_str:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Groq rate limit exceeded. Please try again later.",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Groq API request failed.",
            )

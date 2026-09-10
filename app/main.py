from fastapi import FastAPI
from app.config import APP_TITLE, APP_DESCRIPTION, APP_VERSION
from app.api.llm import router as llm_router
from app.api.source_understanding import router as source_understanding_router
from app.api.content_strategy import router as content_strategy_router
from app.api.content_generation import router as content_generation_router
from app.api.validation import router as validation_router
from app.api.action import router as action_router
from app.api.master import router as master_router
from app.api.pipeline import router as pipeline_router

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    docs_url="/docs",
    openapi_url="/openapi.json",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root():
    """Redirects root URL to interactive Swagger documentation."""
    return RedirectResponse(url="/docs")


@app.get(
    "/health",
    tags=["Health"],
    summary="Health Check",
    description="Verifies the backend server is operational. Does not perform external LLM calls.",
)
def health_check():
    """
    Health check endpoint to verify that the ContentForge AI Backend is running.
    """
    return {
        "status": "ok",
        "service": APP_TITLE,
        "version": APP_VERSION,
    }


# Register routes
app.include_router(llm_router)
app.include_router(source_understanding_router)
app.include_router(content_strategy_router)
app.include_router(content_generation_router)
app.include_router(validation_router)
app.include_router(action_router)
app.include_router(master_router)
app.include_router(pipeline_router)

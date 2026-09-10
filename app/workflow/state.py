from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

from app.schemas.source_understanding import SourceUnderstanding
from app.schemas.content_strategy import ContentStrategy
from app.schemas.content_generation import GeneratedContent
from app.schemas.validation import ValidationResult
from app.schemas.action import ActionResult
from app.schemas.master import MasterDecision


class ContentForgeState(TypedDict, total=False):
    """
    Unified LangGraph state schema for ContentForge agentic workflow.
    Serves as the single source of truth passed and updated across nodes.
    """
    workflow_id: str
    user_request: str
    source_text: str

    decision: MasterDecision

    source_understanding: SourceUnderstanding
    content_strategy: ContentStrategy
    generated_content: GeneratedContent

    validation_result: ValidationResult
    revision_count: int
    validation_history: List[Dict[str, Any]]

    human_approval_required: bool
    human_approval_status: str  # 'none', 'pending', 'approved', 'rejected'
    human_feedback: str

    action_result: ActionResult

    status: str  # 'processing', 'waiting_for_approval', 'validation_failed', 'completed', 'failed'
    current_stage: str  # 'master', 'source', 'strategy', 'generation', 'validation', 'revision', 'human_approval', 'action'
    error: Optional[str]

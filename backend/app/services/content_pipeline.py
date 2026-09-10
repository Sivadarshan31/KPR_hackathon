import logging
from typing import Dict, Any, Optional

from app.agents.source_understanding import (
    SourceUnderstandingAgent,
    source_understanding_agent as default_source_agent,
)
from app.agents.content_strategy import (
    ContentStrategyAgent,
    content_strategy_agent as default_strategy_agent,
)
from app.agents.content_generation import (
    ContentGenerationAgent,
    content_generation_agent as default_generation_agent,
)
from app.schemas.source_understanding import SourceUnderstanding
from app.schemas.content_strategy import ContentStrategy
from app.schemas.content_generation import GeneratedContent
from app.schemas.pipeline import PipelineResponse, PipelineGenerationResponse
from app.workflow.contentforge_graph import contentforge_workflow, ContentForgeWorkflow
from app.workflow.state import ContentForgeState
from app.llm.exceptions import GroqRequestError

logger = logging.getLogger(__name__)


class ContentPipeline:
    """
    Unified execution pipeline service connecting specialized agents and the LangGraph workflow.
    Executes in-memory, thread-isolated content transformation workflows.
    """

    def __init__(
        self,
        source_agent: Optional[SourceUnderstandingAgent] = None,
        strategy_agent: Optional[ContentStrategyAgent] = None,
        generation_agent: Optional[ContentGenerationAgent] = None,
        workflow: Optional[ContentForgeWorkflow] = None,
    ):
        self.source_agent = source_agent or default_source_agent
        self.strategy_agent = strategy_agent or default_strategy_agent
        self.generation_agent = generation_agent or default_generation_agent
        self.workflow = workflow or contentforge_workflow

    async def execute_workflow(
        self,
        user_request: str,
        source_text: Optional[str] = None,
        workflow_id: Optional[str] = None,
    ) -> ContentForgeState:
        """
        Asynchronously runs the full in-memory agentic workflow via LangGraph.
        """
        return await self.workflow.arun(
            user_request=user_request,
            source_text=source_text,
            workflow_id=workflow_id,
        )

    def get_workflow_status(self, workflow_id: str) -> Optional[ContentForgeState]:
        """
        Queries thread-isolated state for a given workflow ID.
        """
        return self.workflow.get_state(workflow_id)

    async def approve_workflow(self, workflow_id: str) -> ContentForgeState:
        """
        Resumes a workflow currently paused at 'waiting_for_approval'.
        """
        return await self.workflow.aapprove(workflow_id)

    async def reject_workflow(self, workflow_id: str, feedback: str) -> ContentForgeState:
        """
        Rejects content with user feedback, routing workflow back into revision.
        """
        return await self.workflow.areject(workflow_id, feedback)

    def run(self, source_text: str) -> PipelineResponse:
        """
        Synchronously executes Agent 1 -> Agent 2 pipeline.
        """
        logger.info("Starting sequential pipeline execution (Agent 1 -> Agent 2 sync).")
        if not source_text or not source_text.strip():
            raise ValueError("source_text must not be empty or whitespace-only.")

        source_understanding = self.source_agent.understand(source_text)
        if not isinstance(source_understanding, SourceUnderstanding):
            raise GroqRequestError("Agent 1 produced invalid structured output; pipeline halted.")

        content_strategy = self.strategy_agent.strategize(source_understanding)
        if not isinstance(content_strategy, ContentStrategy):
            raise GroqRequestError("Agent 2 produced invalid structured output; pipeline halted.")

        return PipelineResponse(
            source_understanding=source_understanding,
            content_strategy=content_strategy,
        )

    async def arun(self, source_text: str) -> PipelineResponse:
        """
        Asynchronously executes Agent 1 -> Agent 2 pipeline.
        """
        logger.info("Starting sequential pipeline execution (Agent 1 -> Agent 2 async).")
        if not source_text or not source_text.strip():
            raise ValueError("source_text must not be empty or whitespace-only.")

        source_understanding = await self.source_agent.aunderstand(source_text)
        if not isinstance(source_understanding, SourceUnderstanding):
            raise GroqRequestError("Agent 1 produced invalid structured output; pipeline halted.")

        content_strategy = await self.strategy_agent.astrategize(source_understanding)
        if not isinstance(content_strategy, ContentStrategy):
            raise GroqRequestError("Agent 2 produced invalid structured output; pipeline halted.")

        return PipelineResponse(
            source_understanding=source_understanding,
            content_strategy=content_strategy,
        )

    def run_full(self, source_text: str) -> PipelineGenerationResponse:
        """
        Synchronously executes Agent 1 -> Agent 2 -> Agent 3 pipeline.
        """
        logger.info("Starting full sequential pipeline execution (Agent 1 -> 2 -> 3 sync).")
        if not source_text or not source_text.strip():
            raise ValueError("source_text must not be empty or whitespace-only.")

        source_understanding = self.source_agent.understand(source_text)
        content_strategy = self.strategy_agent.strategize(source_understanding)
        generated_content = self.generation_agent.generate(
            source_understanding=source_understanding,
            content_strategy=content_strategy,
        )

        return PipelineGenerationResponse(
            source_understanding=source_understanding,
            content_strategy=content_strategy,
            generated_content=generated_content,
        )

    async def arun_full(self, source_text: str) -> PipelineGenerationResponse:
        """
        Asynchronously executes Agent 1 -> Agent 2 -> Agent 3 pipeline.
        """
        logger.info("Starting full sequential pipeline execution (Agent 1 -> 2 -> 3 async).")
        if not source_text or not source_text.strip():
            raise ValueError("source_text must not be empty or whitespace-only.")

        source_understanding = await self.source_agent.aunderstand(source_text)
        content_strategy = await self.strategy_agent.astrategize(source_understanding)
        generated_content = await self.generation_agent.agenerate(
            source_understanding=source_understanding,
            content_strategy=content_strategy,
        )

        return PipelineGenerationResponse(
            source_understanding=source_understanding,
            content_strategy=content_strategy,
            generated_content=generated_content,
        )


# Reusable singleton instance
content_pipeline = ContentPipeline()

import logging
from typing import Optional

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
from app.llm.exceptions import GroqRequestError

logger = logging.getLogger(__name__)


class ContentPipeline:
    """
    Sequential orchestration pipeline connecting:
    Agent 1 (Source Understanding) -> Agent 2 (Content Strategy) -> Agent 3 (Content Generation).
    
    Data flow:
    Source Text
        ↓
    Agent 1 (Source Understanding)
        ↓
    Structured SourceUnderstanding
        ↓
    Agent 2 (Content Strategy)
        ↓
    Structured ContentStrategy
        ↓
    Agent 3 (Content Generation)
        ↓
    Structured GeneratedContent
    """

    def __init__(
        self,
        source_agent: Optional[SourceUnderstandingAgent] = None,
        strategy_agent: Optional[ContentStrategyAgent] = None,
        generation_agent: Optional[ContentGenerationAgent] = None,
    ):
        self.source_agent = source_agent or default_source_agent
        self.strategy_agent = strategy_agent or default_strategy_agent
        self.generation_agent = generation_agent or default_generation_agent

    def run(self, source_text: str) -> PipelineResponse:
        """
        Synchronously executes Agent 1 -> Agent 2 pipeline.
        Maintains backward compatibility with /api/pipeline/source-to-strategy.
        """
        logger.info("Starting sequential pipeline execution (Agent 1 -> Agent 2 sync).")
        if not source_text or not source_text.strip():
            raise ValueError("source_text must not be empty or whitespace-only.")

        # Step 1: Agent 1 - Source Understanding
        logger.info("Executing Agent 1: Source Understanding.")
        source_understanding = self.source_agent.understand(source_text)

        if not isinstance(source_understanding, SourceUnderstanding):
            logger.error("Agent 1 failed to produce a valid SourceUnderstanding object.")
            raise GroqRequestError("Agent 1 produced invalid structured output; pipeline halted.")

        # Step 2: Agent 2 - Content Strategy
        logger.info("Executing Agent 2: Content Strategy.")
        content_strategy = self.strategy_agent.strategize(source_understanding)

        if not isinstance(content_strategy, ContentStrategy):
            logger.error("Agent 2 failed to produce a valid ContentStrategy object.")
            raise GroqRequestError("Agent 2 produced invalid structured output; pipeline halted.")

        logger.info("Sequential pipeline (Agent 1 -> Agent 2) completed successfully.")
        return PipelineResponse(
            source_understanding=source_understanding,
            content_strategy=content_strategy,
        )

    async def arun(self, source_text: str) -> PipelineResponse:
        """
        Asynchronously executes Agent 1 -> Agent 2 pipeline.
        Maintains backward compatibility with /api/pipeline/source-to-strategy.
        """
        logger.info("Starting sequential pipeline execution (Agent 1 -> Agent 2 async).")
        if not source_text or not source_text.strip():
            raise ValueError("source_text must not be empty or whitespace-only.")

        # Step 1: Agent 1 - Source Understanding
        logger.info("Executing Agent 1: Source Understanding.")
        source_understanding = await self.source_agent.aunderstand(source_text)

        if not isinstance(source_understanding, SourceUnderstanding):
            logger.error("Agent 1 failed to produce a valid SourceUnderstanding object.")
            raise GroqRequestError("Agent 1 produced invalid structured output; pipeline halted.")

        # Step 2: Agent 2 - Content Strategy
        logger.info("Executing Agent 2: Content Strategy.")
        content_strategy = await self.strategy_agent.astrategize(source_understanding)

        if not isinstance(content_strategy, ContentStrategy):
            logger.error("Agent 2 failed to produce a valid ContentStrategy object.")
            raise GroqRequestError("Agent 2 produced invalid structured output; pipeline halted.")

        logger.info("Sequential pipeline (Agent 1 -> Agent 2) completed successfully.")
        return PipelineResponse(
            source_understanding=source_understanding,
            content_strategy=content_strategy,
        )

    def run_full(self, source_text: str) -> PipelineGenerationResponse:
        """
        Synchronously executes the complete Agent 1 -> Agent 2 -> Agent 3 pipeline.
        Stops immediately if any agent fails or produces invalid output.
        """
        logger.info("Starting full sequential pipeline execution (Agent 1 -> 2 -> 3 sync).")
        if not source_text or not source_text.strip():
            raise ValueError("source_text must not be empty or whitespace-only.")

        # Step 1: Agent 1 - Source Understanding
        logger.info("Executing Agent 1: Source Understanding.")
        source_understanding = self.source_agent.understand(source_text)

        if not isinstance(source_understanding, SourceUnderstanding):
            logger.error("Agent 1 failed to produce a valid SourceUnderstanding object.")
            raise GroqRequestError("Agent 1 produced invalid structured output; pipeline halted.")

        # Step 2: Agent 2 - Content Strategy
        logger.info("Executing Agent 2: Content Strategy.")
        content_strategy = self.strategy_agent.strategize(source_understanding)

        if not isinstance(content_strategy, ContentStrategy):
            logger.error("Agent 2 failed to produce a valid ContentStrategy object.")
            raise GroqRequestError("Agent 2 produced invalid structured output; pipeline halted.")

        # Step 3: Agent 3 - Content Generation
        logger.info("Executing Agent 3: Content Generation.")
        generated_content = self.generation_agent.generate(
            source_understanding=source_understanding,
            content_strategy=content_strategy,
        )

        if not isinstance(generated_content, GeneratedContent):
            logger.error("Agent 3 failed to produce a valid GeneratedContent object.")
            raise GroqRequestError("Agent 3 produced invalid structured output; pipeline halted.")

        logger.info("Full sequential pipeline (Agent 1 -> 2 -> 3) completed successfully.")
        return PipelineGenerationResponse(
            source_understanding=source_understanding,
            content_strategy=content_strategy,
            generated_content=generated_content,
        )

    async def arun_full(self, source_text: str) -> PipelineGenerationResponse:
        """
        Asynchronously executes the complete Agent 1 -> Agent 2 -> Agent 3 pipeline.
        Stops immediately if any agent fails or produces invalid output.
        """
        logger.info("Starting full sequential pipeline execution (Agent 1 -> 2 -> 3 async).")
        if not source_text or not source_text.strip():
            raise ValueError("source_text must not be empty or whitespace-only.")

        # Step 1: Agent 1 - Source Understanding
        logger.info("Executing Agent 1: Source Understanding.")
        source_understanding = await self.source_agent.aunderstand(source_text)

        if not isinstance(source_understanding, SourceUnderstanding):
            logger.error("Agent 1 failed to produce a valid SourceUnderstanding object.")
            raise GroqRequestError("Agent 1 produced invalid structured output; pipeline halted.")

        # Step 2: Agent 2 - Content Strategy
        logger.info("Executing Agent 2: Content Strategy.")
        content_strategy = await self.strategy_agent.astrategize(source_understanding)

        if not isinstance(content_strategy, ContentStrategy):
            logger.error("Agent 2 failed to produce a valid ContentStrategy object.")
            raise GroqRequestError("Agent 2 produced invalid structured output; pipeline halted.")

        # Step 3: Agent 3 - Content Generation
        logger.info("Executing Agent 3: Content Generation.")
        generated_content = await self.generation_agent.agenerate(
            source_understanding=source_understanding,
            content_strategy=content_strategy,
        )

        if not isinstance(generated_content, GeneratedContent):
            logger.error("Agent 3 failed to produce a valid GeneratedContent object.")
            raise GroqRequestError("Agent 3 produced invalid structured output; pipeline halted.")

        logger.info("Full sequential pipeline (Agent 1 -> 2 -> 3) completed successfully.")
        return PipelineGenerationResponse(
            source_understanding=source_understanding,
            content_strategy=content_strategy,
            generated_content=generated_content,
        )


# Reusable singleton instance
content_pipeline = ContentPipeline()

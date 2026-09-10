import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.schemas import (
    UserRequest,
    SourceUnderstanding,
    ContentStrategy,
    LinkedInStrategy,
    GeneratedContent,
    LinkedInContent,
    ValidationResult,
    ValidationStatus,
    ActionResult,
    MasterDecision,
)
from app.workflow.state import ContentForgeState
from app.workflow.contentforge_graph import (
    MAX_VALIDATION_ATTEMPTS,
    build_contentforge_graph,
    ContentForgeWorkflow,
    master_node,
    source_node,
    strategy_node,
    generation_node,
    validation_node,
    revision_node,
    action_node,
    route_after_master,
    route_after_source,
    route_after_strategy,
    route_after_generation,
    route_after_validation,
)


class TestModule7Module8Workflow(unittest.IsolatedAsyncioTestCase):
    """Tests for Module 7 (LangGraph Workflow Integration) and Module 8 (Validation & Regeneration Loop)."""

    def setUp(self):
        self.mock_su = SourceUnderstanding(
            title="Title",
            summary="Summary",
            main_topic="Topic",
            key_points=["Point 1"],
            facts=["Fact 1"],
            entities=["Entity 1"],
            important_numbers=["100"],
            dates=["2026"],
            claims=["Claim 1"],
            terminology=["Term 1"],
        )
        self.mock_cs = ContentStrategy(
            summary="Strategy summary",
            overall_angle="Tech angle",
            target_audience="Devs",
            key_takeaway="Key takeaway",
        )
        self.mock_gc = GeneratedContent(
            linkedin=LinkedInContent(
                hook="Hook",
                body="Body",
                cta="CTA",
                hashtags=["#Tech"],
            )
        )
        self.mock_val_pass = ValidationResult(
            status=ValidationStatus.PASS,
            passed=True,
            score=95.0,
            issues=[],
            suggestions=[],
        )
        self.mock_val_fail = ValidationResult(
            status=ValidationStatus.FAIL,
            passed=False,
            score=40.0,
            issues=["Incorrect figure"],
            suggestions=["Fix number to match source"],
        )

    def test_01_graph_compilation_and_isolation(self):
        graph1 = build_contentforge_graph()
        graph2 = build_contentforge_graph()
        self.assertIsNotNone(graph1)
        self.assertIsNotNone(graph2)

        wf1 = ContentForgeWorkflow()
        wf2 = ContentForgeWorkflow()
        self.assertIsNot(wf1.checkpointer, wf2.checkpointer)

    @patch("app.workflow.contentforge_graph.master_agent.adecide")
    async def test_02_initial_workflow_state(self, mock_adecide):
        mock_adecide.return_value = MasterDecision(
            user_request="Summarize article",
            intent="summarize",
            requires_source_understanding=True,
            requires_strategy=False,
            requires_generation=False,
            requires_validation=False,
            requires_action=False,
            requires_human_approval=False,
            reasoning_summary="Summarize source",
        )
        wf = ContentForgeWorkflow()
        state = await wf.arun(
            user_request="Summarize article",
            source_text="Sample text content for analysis.",
            workflow_id="test-wf-101",
        )
        self.assertEqual(state["workflow_id"], "test-wf-101")
        self.assertEqual(state["user_request"], "Summarize article")
        self.assertEqual(state["revision_count"], 0)

    @patch("app.workflow.contentforge_graph.master_agent.adecide")
    async def test_03_master_node_execution(self, mock_adecide):
        mock_adecide.return_value = MasterDecision(
            user_request="Test req",
            intent="generate_content",
            requires_source_understanding=True,
            requires_strategy=True,
            requires_generation=True,
            requires_validation=True,
            requires_action=False,
            requires_human_approval=False,
            reasoning_summary="Test routing",
        )
        state: ContentForgeState = {"user_request": "Test req"}
        res = await master_node(state)
        self.assertEqual(res["current_stage"], "master")
        self.assertTrue(res["decision"].requires_source_understanding)

    @patch("app.workflow.contentforge_graph.source_understanding_agent.aunderstand")
    async def test_04_source_node_execution(self, mock_aunderstand):
        mock_aunderstand.return_value = self.mock_su
        state: ContentForgeState = {
            "source_text": "Raw source text",
            "decision": MasterDecision(
                user_request="req",
                intent="intent",
                requires_source_understanding=True,
                requires_strategy=True,
                requires_generation=True,
                requires_validation=True,
                requires_action=False,
                requires_human_approval=False,
                reasoning_summary="routing",
            ),
        }
        res = await source_node(state)
        self.assertEqual(res["source_understanding"].title, "Title")
        self.assertEqual(res["current_stage"], "source")

    async def test_05_source_node_empty_source_text_error_handling(self):
        state: ContentForgeState = {"source_text": "   "}
        res = await source_node(state)
        self.assertEqual(res["status"], "failed")
        self.assertIn("empty", res["error"].lower())

    @patch("app.workflow.contentforge_graph.content_strategy_agent.astrategize")
    async def test_06_strategy_node_execution(self, mock_astrategize):
        mock_astrategize.return_value = self.mock_cs
        state: ContentForgeState = {"source_understanding": self.mock_su}
        res = await strategy_node(state)
        self.assertEqual(res["content_strategy"].overall_angle, "Tech angle")

    @patch("app.workflow.contentforge_graph.content_generation_agent.agenerate")
    async def test_07_generation_node_execution(self, mock_agenerate):
        mock_agenerate.return_value = self.mock_gc
        state: ContentForgeState = {
            "source_understanding": self.mock_su,
            "content_strategy": self.mock_cs,
        }
        res = await generation_node(state)
        self.assertIsNotNone(res["generated_content"])
        self.assertEqual(res["current_stage"], "generation")

    @patch("app.workflow.contentforge_graph.validation_agent.avalidate")
    async def test_08_validation_node_pass_path(self, mock_avalidate):
        mock_avalidate.return_value = self.mock_val_pass
        state: ContentForgeState = {
            "source_understanding": self.mock_su,
            "generated_content": self.mock_gc,
            "revision_count": 0,
            "validation_history": [],
        }
        res = await validation_node(state)
        self.assertTrue(res["validation_result"].passed)
        self.assertEqual(len(res["validation_history"]), 1)

    @patch("app.workflow.contentforge_graph.validation_agent.avalidate")
    async def test_09_validation_node_fail_path(self, mock_avalidate):
        mock_avalidate.return_value = self.mock_val_fail
        state: ContentForgeState = {
            "source_understanding": self.mock_su,
            "generated_content": self.mock_gc,
            "revision_count": 0,
            "validation_history": [],
        }
        res = await validation_node(state)
        self.assertFalse(res["validation_result"].passed)

    def test_10_routing_after_validation_pass_and_fail(self):
        # PASS -> END
        state_pass: ContentForgeState = {
            "validation_result": self.mock_val_pass,
            "decision": MasterDecision(
                user_request="u", intent="i",
                requires_source_understanding=True, requires_strategy=True,
                requires_generation=True, requires_validation=True,
                requires_action=False, requires_human_approval=False,
                reasoning_summary="r"
            ),
            "revision_count": 0,
        }
        self.assertEqual(route_after_validation(state_pass), "end")

        # FAIL (attempt 1/3) -> revision
        state_fail_1: ContentForgeState = {
            "validation_result": self.mock_val_fail,
            "decision": MasterDecision(
                user_request="u", intent="i",
                requires_source_understanding=True, requires_strategy=True,
                requires_generation=True, requires_validation=True,
                requires_action=False, requires_human_approval=False,
                reasoning_summary="r"
            ),
            "revision_count": 0,
        }
        self.assertEqual(route_after_validation(state_fail_1), "revision")

        # FAIL (attempt 3/3 max reached) -> end with validation_failed
        state_fail_max: ContentForgeState = {
            "validation_result": self.mock_val_fail,
            "decision": MasterDecision(
                user_request="u", intent="i",
                requires_source_understanding=True, requires_strategy=True,
                requires_generation=True, requires_validation=True,
                requires_action=False, requires_human_approval=False,
                reasoning_summary="r"
            ),
            "revision_count": MAX_VALIDATION_ATTEMPTS,
        }
        self.assertEqual(route_after_validation(state_fail_max), "end")
        self.assertEqual(state_fail_max["status"], "validation_failed")

    @patch("app.workflow.contentforge_graph.default_groq_manager.ainvoke")
    @patch("app.workflow.contentforge_graph._parse_generation_json")
    async def test_11_revision_node_increments_count_and_uses_feedback(self, mock_parse, mock_ainvoke):
        mock_revised_gc = GeneratedContent(linkedin="Revised copy addressing issues")
        mock_parse.return_value = mock_revised_gc
        mock_ainvoke.return_value = MagicMock(content="{}")

        state: ContentForgeState = {
            "source_understanding": self.mock_su,
            "content_strategy": self.mock_cs,
            "generated_content": self.mock_gc,
            "validation_result": self.mock_val_fail,
            "revision_count": 1,
        }
        res = await revision_node(state)
        self.assertEqual(res["revision_count"], 2)
        self.assertEqual(res["generated_content"].linkedin, "Revised copy addressing issues")

    @patch("app.workflow.contentforge_graph.action_agent.aexecute")
    async def test_12_action_node_execution(self, mock_aexecute):
        mock_aexecute.return_value = ActionResult(
            success=True,
            action="preview",
            platform="all",
            action_allowed=True,
            status="completed",
            message="Action completed",
            validation_status="PASS",
        )
        state: ContentForgeState = {
            "generated_content": self.mock_gc,
            "validation_result": self.mock_val_pass,
            "decision": MasterDecision(
                user_request="u", intent="i",
                requires_source_understanding=True, requires_strategy=True,
                requires_generation=True, requires_validation=True,
                requires_action=True, requested_action="preview",
                requires_human_approval=False,
                reasoning_summary="r"
            ),
        }
        res = await action_node(state)
        self.assertEqual(res["action_result"].action, "preview")
        self.assertEqual(res["status"], "completed")


if __name__ == "__main__":
    unittest.main()

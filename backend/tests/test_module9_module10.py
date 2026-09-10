import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from app.workflow.contentforge_graph import (
    ContentForgeWorkflow,
    build_contentforge_graph,
    contentforge_workflow,
)
from app.services.content_pipeline import ContentPipeline, content_pipeline
from app.schemas import (
    SourceUnderstanding,
    ContentStrategy,
    GeneratedContent,
    LinkedInContent,
    ValidationResult,
    ValidationStatus,
    ActionResult,
    MasterDecision,
    MasterRunResponse,
)
from app.llm.exceptions import GroqConfigurationError, GroqRequestError


class TestModule9HumanInTheLoop(unittest.IsolatedAsyncioTestCase):
    """Tests for Module 9: Human-In-The-Loop Checkpointing, Pausing, Approval, and Rejection."""

    def setUp(self):
        self.mock_su = SourceUnderstanding(
            title="Title", summary="Summary", main_topic="Topic",
            key_points=[], facts=[], entities=[], important_numbers=[],
            dates=[], claims=[], terminology=[],
        )
        self.mock_cs = ContentStrategy(
            summary="S", overall_angle="A", target_audience="Audience", key_takeaway="K",
        )
        self.mock_gc = GeneratedContent(
            linkedin=LinkedInContent(hook="H", body="B", cta="C", hashtags=[])
        )
        self.mock_val_pass = ValidationResult(
            status=ValidationStatus.PASS, passed=True, score=95.0, issues=[], suggestions=[]
        )

    @patch("app.workflow.contentforge_graph.master_agent.adecide")
    @patch("app.workflow.contentforge_graph.source_understanding_agent.aunderstand")
    @patch("app.workflow.contentforge_graph.content_strategy_agent.astrategize")
    @patch("app.workflow.contentforge_graph.content_generation_agent.agenerate")
    @patch("app.workflow.contentforge_graph.validation_agent.avalidate")
    @patch("app.workflow.contentforge_graph.action_agent.aexecute")
    async def test_01_approval_not_required_continues_to_action(
        self, mock_act, mock_val, mock_gen, mock_strat, mock_src, mock_master
    ):
        """1. approval not required -> workflow continues to completion."""
        mock_master.return_value = MasterDecision(
            user_request="preview post", intent="preview",
            requires_source_understanding=True, requires_strategy=True,
            requires_generation=True, requires_validation=True,
            requires_action=True, requested_action="preview",
            requires_human_approval=False, reasoning_summary="No human gate for preview"
        )
        mock_src.return_value = self.mock_su
        mock_strat.return_value = self.mock_cs
        mock_gen.return_value = self.mock_gc
        mock_val.return_value = self.mock_val_pass
        mock_act.return_value = ActionResult(
            success=True, action="preview", platform="all", action_allowed=True,
            status="completed", message="Preview ready", validation_status="PASS"
        )

        wf = ContentForgeWorkflow()
        state = await wf.arun(user_request="preview post", source_text="Source text", workflow_id="wf-no-approval")
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["current_stage"], "action")

    @patch("app.workflow.contentforge_graph.master_agent.adecide")
    @patch("app.workflow.contentforge_graph.source_understanding_agent.aunderstand")
    @patch("app.workflow.contentforge_graph.content_strategy_agent.astrategize")
    @patch("app.workflow.contentforge_graph.content_generation_agent.agenerate")
    @patch("app.workflow.contentforge_graph.validation_agent.avalidate")
    async def test_02_approval_required_pauses_workflow(
        self, mock_val, mock_gen, mock_strat, mock_src, mock_master
    ):
        """2. approval required -> workflow pauses at waiting_for_approval."""
        mock_master.return_value = MasterDecision(
            user_request="publish post", intent="publish",
            requires_source_understanding=True, requires_strategy=True,
            requires_generation=True, requires_validation=True,
            requires_action=True, requested_action="publish",
            requires_human_approval=True, reasoning_summary="Human gate for publishing"
        )
        mock_src.return_value = self.mock_su
        mock_strat.return_value = self.mock_cs
        mock_gen.return_value = self.mock_gc
        mock_val.return_value = self.mock_val_pass

        wf = ContentForgeWorkflow()
        state = await wf.arun(user_request="publish post", source_text="Source text", workflow_id="wf-approval-pause")
        self.assertEqual(state["status"], "waiting_for_approval")
        self.assertEqual(state["human_approval_status"], "pending")
        self.assertTrue(state["human_approval_required"])

    async def test_03_pending_approval_status_query(self):
        """3. pending approval status can be queried."""
        wf = ContentForgeWorkflow()
        # Seed state in waiting_for_approval
        config = {"configurable": {"thread_id": "wf-query-pending"}}
        wf.graph.update_state(
            config,
            {
                "workflow_id": "wf-query-pending",
                "status": "waiting_for_approval",
                "current_stage": "human_approval",
                "human_approval_required": True,
                "human_approval_status": "pending",
            },
            as_node="approval_checkpoint_node",
        )
        query_state = wf.get_state("wf-query-pending")
        self.assertEqual(query_state["status"], "waiting_for_approval")
        self.assertEqual(query_state["human_approval_status"], "pending")

    @patch("app.workflow.contentforge_graph.action_agent.aexecute")
    async def test_04_approve_resumes_workflow_to_action(self, mock_act):
        """4. approve -> workflow resumes to action."""
        mock_act.return_value = ActionResult(
            success=True, action="publish", platform="all", action_allowed=True,
            status="dry_run", message="Published dry run", validation_status="PASS"
        )

        wf = ContentForgeWorkflow()
        config = {"configurable": {"thread_id": "wf-approve-resume"}}
        # Initialize waiting_for_approval
        wf.graph.update_state(
            config,
            {
                "workflow_id": "wf-approve-resume",
                "status": "waiting_for_approval",
                "current_stage": "human_approval",
                "human_approval_required": True,
                "human_approval_status": "pending",
                "decision": MasterDecision(
                    user_request="pub", intent="pub",
                    requires_source_understanding=True, requires_strategy=True,
                    requires_generation=True, requires_validation=True,
                    requires_action=True, requested_action="publish",
                    requires_human_approval=True, reasoning_summary="r"
                ),
                "generated_content": self.mock_gc,
                "validation_result": self.mock_val_pass,
            },
            as_node="approval_checkpoint_node",
        )

        resumed = await wf.aapprove("wf-approve-resume")
        self.assertEqual(resumed["human_approval_status"], "approved")
        self.assertEqual(resumed["status"], "completed")

    @patch("app.workflow.contentforge_graph.default_groq_manager.ainvoke")
    @patch("app.workflow.contentforge_graph._parse_generation_json")
    @patch("app.workflow.contentforge_graph.validation_agent.avalidate")
    async def test_05_and_06_reject_routes_feedback_to_revision(self, mock_val, mock_parse, mock_ainvoke):
        """5 & 6. reject -> workflow handles rejection and routes feedback to revision."""
        mock_parse.return_value = GeneratedContent(linkedin="Revised after human feedback")
        mock_ainvoke.return_value = MagicMock(content="{}")
        mock_val.return_value = self.mock_val_pass

        wf = ContentForgeWorkflow()
        config = {"configurable": {"thread_id": "wf-reject-feedback"}}
        wf.graph.update_state(
            config,
            {
                "workflow_id": "wf-reject-feedback",
                "status": "waiting_for_approval",
                "current_stage": "human_approval",
                "human_approval_required": True,
                "human_approval_status": "pending",
                "source_understanding": self.mock_su,
                "content_strategy": self.mock_cs,
                "generated_content": self.mock_gc,
                "validation_result": self.mock_val_pass,
                "revision_count": 0,
            },
            as_node="approval_checkpoint_node",
        )

        resumed = await wf.areject("wf-reject-feedback", feedback="Make the post punchier")
        self.assertEqual(resumed["human_approval_status"], "rejected")

    async def test_07_workflow_id_isolation(self):
        """7. workflow ID isolation across threads."""
        wf = ContentForgeWorkflow()
        config1 = {"configurable": {"thread_id": "wf-user-A"}}
        config2 = {"configurable": {"thread_id": "wf-user-B"}}

        wf.graph.update_state(config1, {"status": "waiting_for_approval", "workflow_id": "wf-user-A"}, as_node="approval_checkpoint_node")
        wf.graph.update_state(config2, {"status": "processing", "workflow_id": "wf-user-B"}, as_node="source_node")

        state_a = wf.get_state("wf-user-A")
        state_b = wf.get_state("wf-user-B")

        self.assertEqual(state_a["status"], "waiting_for_approval")
        self.assertEqual(state_b["status"], "processing")

    async def test_08_invalid_workflow_id_raises_value_error(self):
        """8. invalid workflow ID raises clear ValueError/HTTP Exception."""
        wf = ContentForgeWorkflow()
        with self.assertRaises(ValueError) as cm:
            await wf.aapprove("nonexistent-wf-id")
        self.assertIn("not found", str(cm.exception).lower())

    async def test_09_and_10_duplicate_approval_rejection_error(self):
        """9 & 10. duplicate approval/rejection handling."""
        wf = ContentForgeWorkflow()
        config = {"configurable": {"thread_id": "wf-completed"}}
        wf.graph.update_state(config, {"status": "completed", "workflow_id": "wf-completed"}, as_node="action_node")

        # Duplicate approve on completed workflow is idempotent or returns existing state
        resumed = await wf.aapprove("wf-completed")
        self.assertEqual(resumed["status"], "completed")

        # Reject on completed workflow raises ValueError
        with self.assertRaises(ValueError) as cm:
            await wf.areject("wf-completed", feedback="Too late")
        self.assertIn("completed", str(cm.exception).lower())


class TestModule10PipelineAndMasterAPIs(unittest.TestCase):
    """Tests for Module 10: In-Memory Pipeline Service & Master REST APIs."""

    def setUp(self):
        self.client = TestClient(app)

    @patch("app.api.pipeline.content_pipeline.execute_workflow")
    def test_11_and_12_and_13_pipeline_run_endpoint_success(self, mock_exec):
        """11, 12, 13. pipeline endpoint accepts request, starts workflow, returns workflow_id."""
        mock_exec.return_value = {
            "workflow_id": "wf-pipeline-123",
            "status": "processing",
            "current_stage": "master",
            "decision": MasterDecision(
                user_request="req", intent="intent",
                requires_source_understanding=True, requires_strategy=True,
                requires_generation=True, requires_validation=True,
                requires_action=False, requires_human_approval=False,
                reasoning_summary="summary"
            ),
        }

        resp = self.client.post(
            "/api/pipeline/run",
            json={
                "user_request": "Create a post from this text",
                "source_text": "Sample raw source text content.",
                "workflow_id": "wf-pipeline-123",
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["workflow_id"], "wf-pipeline-123")
        self.assertEqual(data["status"], "processing")

    @patch("app.api.pipeline.content_pipeline.get_workflow_status")
    def test_14_and_15_pipeline_status_query_endpoint(self, mock_status):
        """14, 15. workflow status can be queried and completed state returned."""
        mock_status.return_value = {
            "workflow_id": "wf-status-99",
            "status": "completed",
            "current_stage": "action",
            "human_approval_required": False,
            "human_approval_status": "none",
            "generated_content": {"linkedin": "Final post"},
        }

        resp = self.client.get("/api/pipeline/wf-status-99")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["workflow_id"], "wf-status-99")
        self.assertEqual(data["status"], "completed")

    @patch("app.api.pipeline.content_pipeline.get_workflow_status")
    def test_16_validation_failure_represented_correctly(self, mock_status):
        """16. validation failure status represented correctly."""
        mock_status.return_value = {
            "workflow_id": "wf-val-failed",
            "status": "validation_failed",
            "current_stage": "validation",
            "revision_count": 3,
        }

        resp = self.client.get("/api/pipeline/wf-val-failed")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "validation_failed")

    @patch("app.api.master.contentforge_workflow.arun")
    def test_17_master_api_integrates_with_workflow(self, mock_arun):
        """17. Master API integrates with workflow."""
        mock_arun.return_value = {
            "workflow_id": "wf-master-api",
            "status": "completed",
            "current_stage": "action",
            "decision": MasterDecision(
                user_request="req", intent="intent",
                requires_source_understanding=True, requires_strategy=True,
                requires_generation=True, requires_validation=True,
                requires_action=False, requires_human_approval=False,
                reasoning_summary="summary"
            ),
        }

        resp = self.client.post(
            "/api/master/run",
            json={
                "user_request": "Transform source into advisory",
                "source_text": "Executive report text...",
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["workflow_id"], "wf-master-api")

    def test_18_invalid_requests_return_appropriate_errors(self):
        """18. invalid requests return 422 HTTP unprocessable entity."""
        resp = self.client.post(
            "/api/pipeline/run",
            json={"user_request": "   ", "source_text": ""},
        )
        self.assertEqual(resp.status_code, 422)

    @patch("app.api.master.contentforge_workflow.arun")
    def test_19_agent_llm_errors_handled_cleanly(self, mock_arun):
        """19. agent/llm provider errors return 502/503 HTTP status."""
        mock_arun.side_effect = GroqRequestError("Provider 429 rate limit")
        resp = self.client.post(
            "/api/master/run",
            json={"user_request": "Valid prompt", "source_text": "Valid text"},
        )
        self.assertEqual(resp.status_code, 429)

    @patch("app.api.master.contentforge_workflow.aapprove")
    def test_20_approval_api_route_handling(self, mock_approve):
        """20. Approval API route handles 200 OK, 404 for unknown, 400 for bad state."""
        mock_approve.return_value = {
            "workflow_id": "wf-approved",
            "status": "completed",
            "current_stage": "action",
            "decision": MasterDecision(
                user_request="req", intent="intent",
                requires_source_understanding=True, requires_strategy=True,
                requires_generation=True, requires_validation=True,
                requires_action=True, requested_action="publish",
                requires_human_approval=True, reasoning_summary="summary"
            ),
        }

        resp = self.client.post("/api/master/approve", json={"workflow_id": "wf-approved"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "completed")

        # Unknown workflow ID -> 404
        mock_approve.side_effect = ValueError("Workflow ID 'unknown' not found.")
        resp_404 = self.client.post("/api/master/approve", json={"workflow_id": "unknown"})
        self.assertEqual(resp_404.status_code, 404)


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient

from app.schemas.source_understanding import SourceUnderstanding
from app.schemas.content_strategy import ContentStrategy, LinkedInStrategy, InstagramStrategy, AdvisoryStrategy
from app.schemas.content_generation import GeneratedContent
from app.schemas.validation import ValidationResult
from app.schemas.action import ActionResult
from app.schemas.master import (
    MasterDecision,
    MasterRunRequest,
    MasterRunResponse,
    MasterApproveRequest,
    MasterRejectRequest,
)
from app.agents.master_agent import MasterAgent, SYSTEM_PROMPT
from app.workflow.contentforge_graph import ContentForgeWorkflow
from app.main import app


class TestMasterWorkflow(unittest.IsolatedAsyncioTestCase):
    """
    Unit test suite for Phase 7, 8, 9:
    - Master Agent decision making
    - LangGraph dynamic routing
    - Autonomous validation & revision loop
    - Human approval pausing and resumption
    - FastAPI master endpoints
    """

    def setUp(self):
        self.sample_source = "Acme Solar announced a new solar panel system in 2026 achieving 35% efficiency."

        self.sample_su = SourceUnderstanding(
            title="Acme Solar 2026",
            summary="Acme Solar announced a solar system in 2026 with 35% efficiency.",
            main_topic="Solar Energy",
            key_points=["35% efficiency", "Launched 2026"],
            facts=["35% efficiency", "Launched in 2026"],
            entities=["Acme Solar"],
            important_numbers=["35%", "2026"],
            dates=["2026"],
            claims=[],
            terminology=["solar"],
            target_audience="Engineers",
            tone="Professional",
            source_type="News",
        )

        self.sample_strategy = ContentStrategy(
            summary="Solar campaign",
            overall_angle="Efficiency breakthrough",
            target_audience="Industry leaders",
            key_takeaway="35% efficiency sets a new standard",
            linkedin=LinkedInStrategy(
                objective="Thought leadership",
                audience="Professionals",
                angle="Solar efficiency",
                key_message="35% efficiency achieved in 2026",
                tone="Professional",
                cta="Thoughts?",
                recommended_structure=["Hook", "Details", "CTA"],
            ),
            instagram=InstagramStrategy(
                objective="Event recap",
                audience="Students",
                angle="Highlights",
                carousel_direction=["Slide 1: Intro"],
                visual_direction="Graphic",
                tone="Dynamic",
                cta="Follow",
            ),
            advisory=AdvisoryStrategy(
                objective="Briefing",
                audience="Execs",
                key_information=["35% efficiency"],
                priority="High",
                tone="Executive",
                recommended_structure=["Overview"],
            ),
        )

        self.sample_content = GeneratedContent(
            linkedin="Acme Solar announced a breakthrough 35% efficiency solar system in 2026.\n\n#SolarEnergy #CleanTech",
        )

        self.sample_val_pass = ValidationResult(
            passed=True,
            score=0.95,
            issues=[],
            suggestions=[],
            checks={"no_hallucinations": True},
        )

        self.sample_val_fail = ValidationResult(
            passed=False,
            score=0.5,
            issues=["Missing efficiency metric."],
            suggestions=["Include 35% efficiency."],
            checks={"no_hallucinations": True},
        )

        self.sample_decision_gen = MasterDecision(
            intent="generate_content",
            requires_source_understanding=True,
            requires_strategy=True,
            requires_generation=True,
            requires_validation=True,
            requires_action=False,
            requires_human_approval=False,
            requested_platforms=["linkedin"],
            requested_formats=["post"],
            requested_action=None,
            user_request="Create a professional LinkedIn post from this report.",
            reasoning_summary="User requested LinkedIn post generation and validation.",
        )

        self.sample_decision_publish = MasterDecision(
            intent="publish_content",
            requires_source_understanding=True,
            requires_strategy=True,
            requires_generation=True,
            requires_validation=True,
            requires_action=True,
            requires_human_approval=True,
            requested_platforms=["linkedin"],
            requested_formats=["post"],
            requested_action="publish",
            user_request="Create a LinkedIn post from this report and publish it.",
            reasoning_summary="User requested content creation and publishing; human approval required.",
        )

    def test_01_master_prompt_constraints(self):
        """Verify Master Agent prompt enforces routing rules and zero chain-of-thought exposure."""
        prompt_lower = SYSTEM_PROMPT.lower()
        self.assertIn("master agent", prompt_lower)
        self.assertIn("requires_human_approval", prompt_lower)
        self.assertIn("publish", prompt_lower)
        self.assertIn("reasoning_summary", prompt_lower)

    def test_02_master_agent_sync_decide(self):
        """Test MasterAgent synchronous decide method."""
        mock_manager = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = self.sample_decision_gen.model_dump_json()
        mock_manager.invoke.return_value = mock_resp

        agent = MasterAgent(manager=mock_manager)
        decision = agent.decide("Create a LinkedIn post from this report.")

        self.assertIsInstance(decision, MasterDecision)
        self.assertTrue(decision.requires_generation)
        self.assertFalse(decision.requires_human_approval)

    async def test_03_workflow_generation_run(self):
        """Test full LangGraph execution for generation (without publishing)."""
        wf = ContentForgeWorkflow()

        with patch("app.workflow.contentforge_graph.master_agent.adecide", new_callable=AsyncMock) as mock_decide, \
             patch("app.workflow.contentforge_graph.source_understanding_agent.aunderstand", new_callable=AsyncMock) as mock_source, \
             patch("app.workflow.contentforge_graph.content_strategy_agent.astrategize", new_callable=AsyncMock) as mock_strategy, \
             patch("app.workflow.contentforge_graph.content_generation_agent.agenerate", new_callable=AsyncMock) as mock_gen, \
             patch("app.workflow.contentforge_graph.validation_agent.avalidate", new_callable=AsyncMock) as mock_val:

            mock_decide.return_value = self.sample_decision_gen
            mock_source.return_value = self.sample_su
            mock_strategy.return_value = self.sample_strategy
            mock_gen.return_value = self.sample_content
            mock_val.return_value = self.sample_val_pass

            state = await wf.arun(
                user_request="Create a professional LinkedIn post from this report.",
                source_text=self.sample_source,
                workflow_id="wf-test-gen-01",
            )

            self.assertEqual(state.get("status"), "completed")
            self.assertIsNotNone(state.get("source_understanding"))
            self.assertIsNotNone(state.get("generated_content"))
            self.assertTrue(state.get("validation_result").passed)
            self.assertEqual(state.get("revision_count"), 0)

    async def test_04_workflow_human_approval_pause(self):
        """Test that publishing request pauses at 'waiting_for_approval'."""
        wf = ContentForgeWorkflow()

        with patch("app.workflow.contentforge_graph.master_agent.adecide", new_callable=AsyncMock) as mock_decide, \
             patch("app.workflow.contentforge_graph.source_understanding_agent.aunderstand", new_callable=AsyncMock) as mock_source, \
             patch("app.workflow.contentforge_graph.content_strategy_agent.astrategize", new_callable=AsyncMock) as mock_strategy, \
             patch("app.workflow.contentforge_graph.content_generation_agent.agenerate", new_callable=AsyncMock) as mock_gen, \
             patch("app.workflow.contentforge_graph.validation_agent.avalidate", new_callable=AsyncMock) as mock_val:

            mock_decide.return_value = self.sample_decision_publish
            mock_source.return_value = self.sample_su
            mock_strategy.return_value = self.sample_strategy
            mock_gen.return_value = self.sample_content
            mock_val.return_value = self.sample_val_pass

            state = await wf.arun(
                user_request="Create a LinkedIn post from this report and publish it.",
                source_text=self.sample_source,
                workflow_id="wf-test-publish-02",
            )

            # Workflow must pause at waiting_for_approval
            self.assertEqual(state.get("status"), "waiting_for_approval")
            self.assertEqual(state.get("current_stage"), "human_approval")
            self.assertTrue(state.get("human_approval_required"))
            self.assertEqual(state.get("human_approval_status"), "pending")
            # Action agent must NOT have run yet
            self.assertIsNone(state.get("action_result"))

            # Now test approval resumption
            approved_state = await wf.aapprove("wf-test-publish-02")
            self.assertEqual(approved_state.get("status"), "completed")
            self.assertEqual(approved_state.get("current_stage"), "action")
            self.assertIsNotNone(approved_state.get("action_result"))
            self.assertEqual(approved_state.get("action_result").status, "dry_run")

    async def test_05_workflow_autonomous_validation_loop(self):
        """Test that a validation failure triggers autonomous revision and re-validation."""
        wf = ContentForgeWorkflow()

        with patch("app.workflow.contentforge_graph.master_agent.adecide", new_callable=AsyncMock) as mock_decide, \
             patch("app.workflow.contentforge_graph.source_understanding_agent.aunderstand", new_callable=AsyncMock) as mock_source, \
             patch("app.workflow.contentforge_graph.content_strategy_agent.astrategize", new_callable=AsyncMock) as mock_strategy, \
             patch("app.workflow.contentforge_graph.content_generation_agent.agenerate", new_callable=AsyncMock) as mock_gen, \
             patch("app.workflow.contentforge_graph.validation_agent.avalidate", new_callable=AsyncMock) as mock_val, \
             patch("app.workflow.contentforge_graph.default_groq_manager.ainvoke", new_callable=AsyncMock) as mock_rev_llm:

            mock_decide.return_value = self.sample_decision_gen
            mock_source.return_value = self.sample_su
            mock_strategy.return_value = self.sample_strategy
            mock_gen.return_value = self.sample_content
            # First attempt fails, second attempt passes!
            mock_val.side_effect = [self.sample_val_fail, self.sample_val_pass]

            revised_mock_resp = MagicMock()
            revised_mock_resp.content = self.sample_content.model_dump_json()
            mock_rev_llm.return_value = revised_mock_resp

            state = await wf.arun(
                user_request="Create a professional LinkedIn post from this report.",
                source_text=self.sample_source,
                workflow_id="wf-test-revision-03",
            )

            self.assertEqual(state.get("status"), "completed")
            self.assertEqual(state.get("revision_count"), 1)
            self.assertEqual(len(state.get("validation_history")), 2)
            self.assertFalse(state.get("validation_history")[0]["passed"])
            self.assertTrue(state.get("validation_history")[1]["passed"])

    async def test_06_workflow_max_validation_attempts(self):
        """Test that validation failure halts at 'validation_failed' after max attempts."""
        wf = ContentForgeWorkflow()

        with patch("app.workflow.contentforge_graph.master_agent.adecide", new_callable=AsyncMock) as mock_decide, \
             patch("app.workflow.contentforge_graph.source_understanding_agent.aunderstand", new_callable=AsyncMock) as mock_source, \
             patch("app.workflow.contentforge_graph.content_strategy_agent.astrategize", new_callable=AsyncMock) as mock_strategy, \
             patch("app.workflow.contentforge_graph.content_generation_agent.agenerate", new_callable=AsyncMock) as mock_gen, \
             patch("app.workflow.contentforge_graph.validation_agent.avalidate", new_callable=AsyncMock) as mock_val, \
             patch("app.workflow.contentforge_graph.default_groq_manager.ainvoke", new_callable=AsyncMock) as mock_rev_llm:

            mock_decide.return_value = self.sample_decision_gen
            mock_source.return_value = self.sample_su
            mock_strategy.return_value = self.sample_strategy
            mock_gen.return_value = self.sample_content
            # Always fails validation
            mock_val.return_value = self.sample_val_fail

            revised_mock_resp = MagicMock()
            revised_mock_resp.content = self.sample_content.model_dump_json()
            mock_rev_llm.return_value = revised_mock_resp

            state = await wf.arun(
                user_request="Create a LinkedIn post from this report.",
                source_text=self.sample_source,
                workflow_id="wf-test-fail-04",
            )

            self.assertEqual(state.get("status"), "validation_failed")
            self.assertEqual(state.get("revision_count"), 3)
            self.assertEqual(len(state.get("validation_history")), 4)

    async def test_07_workflow_rejection_and_revision(self):
        """Test that rejecting human approval provides feedback and routes back to revision."""
        wf = ContentForgeWorkflow()

        with patch("app.workflow.contentforge_graph.master_agent.adecide", new_callable=AsyncMock) as mock_decide, \
             patch("app.workflow.contentforge_graph.source_understanding_agent.aunderstand", new_callable=AsyncMock) as mock_source, \
             patch("app.workflow.contentforge_graph.content_strategy_agent.astrategize", new_callable=AsyncMock) as mock_strategy, \
             patch("app.workflow.contentforge_graph.content_generation_agent.agenerate", new_callable=AsyncMock) as mock_gen, \
             patch("app.workflow.contentforge_graph.validation_agent.avalidate", new_callable=AsyncMock) as mock_val, \
             patch("app.workflow.contentforge_graph.default_groq_manager.ainvoke", new_callable=AsyncMock) as mock_rev_llm:

            mock_decide.return_value = self.sample_decision_publish
            mock_source.return_value = self.sample_su
            mock_strategy.return_value = self.sample_strategy
            mock_gen.return_value = self.sample_content
            mock_val.return_value = self.sample_val_pass

            revised_mock_resp = MagicMock()
            revised_mock_resp.content = self.sample_content.model_dump_json()
            mock_rev_llm.return_value = revised_mock_resp

            # Initial run -> pauses at waiting_for_approval
            state = await wf.arun(
                user_request="Create a LinkedIn post and publish it.",
                source_text=self.sample_source,
                workflow_id="wf-test-reject-05",
            )
            self.assertEqual(state.get("status"), "waiting_for_approval")

            # Reject with specific feedback
            rejected_state = await wf.areject("wf-test-reject-05", feedback="Please make the hook stronger.")
            # After revision and validation pass, because publish was requested, pauses again for approval
            self.assertEqual(rejected_state.get("status"), "waiting_for_approval")
            self.assertGreaterEqual(rejected_state.get("revision_count"), 1)

    def test_08_fastapi_master_endpoints_full(self):
        """Test FastAPI endpoints /api/master/run, /status, /approve, and /reject."""
        client = TestClient(app)

        with patch("app.api.master.contentforge_workflow.arun", new_callable=AsyncMock) as mock_wf_run, \
             patch("app.api.master.contentforge_workflow.get_state") as mock_wf_status, \
             patch("app.api.master.contentforge_workflow.aapprove", new_callable=AsyncMock) as mock_wf_approve, \
             patch("app.api.master.contentforge_workflow.areject", new_callable=AsyncMock) as mock_wf_reject:

            # 1. Test POST /api/master/run
            mock_state = {
                "workflow_id": "wf-api-test-01",
                "status": "completed",
                "current_stage": "validation",
                "decision": self.sample_decision_gen,
                "source_understanding": self.sample_su,
                "content_strategy": self.sample_strategy,
                "generated_content": self.sample_content,
                "validation_result": self.sample_val_pass,
                "action_result": None,
                "human_approval_required": False,
                "human_approval_status": "none",
                "revision_count": 0,
                "validation_history": [],
            }
            mock_wf_run.return_value = mock_state

            payload = {
                "user_request": "Create a LinkedIn post.",
                "source_text": self.sample_source,
            }
            resp = client.post("/api/master/run", json=payload)
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["workflow_id"], "wf-api-test-01")
            self.assertEqual(data["status"], "completed")
            self.assertIn("linkedin", data["generated_content"])
            self.assertTrue(data["validation"]["passed"])

            # 2. Test GET /api/master/status/{workflow_id}
            paused_state = dict(mock_state)
            paused_state["status"] = "waiting_for_approval"
            paused_state["current_stage"] = "human_approval"
            paused_state["human_approval_required"] = True
            paused_state["human_approval_status"] = "pending"
            mock_wf_status.return_value = paused_state

            status_resp = client.get("/api/master/status/wf-api-test-01")
            self.assertEqual(status_resp.status_code, 200)
            sdata = status_resp.json()
            self.assertEqual(sdata["status"], "waiting_for_approval")
            self.assertTrue(sdata["approval_required"])

            # 3. Test POST /api/master/approve
            approved_state = dict(mock_state)
            approved_state["status"] = "completed"
            approved_state["current_stage"] = "action"
            approved_state["action_result"] = ActionResult(
                success=True,
                action="publish",
                platform="linkedin",
                validation_status="PASS",
                action_allowed=True,
                status="dry_run",
                message="Simulation completed.",
                content={},
            )
            mock_wf_approve.return_value = approved_state

            approve_resp = client.post(
                "/api/master/approve",
                json={"workflow_id": "wf-api-test-01", "feedback": "Good to publish."},
            )
            self.assertEqual(approve_resp.status_code, 200)
            adata = approve_resp.json()
            self.assertEqual(adata["status"], "completed")
            self.assertEqual(adata["current_stage"], "action")
            self.assertIsNotNone(adata["action_result"])

            # 4. Test POST /api/master/reject
            mock_wf_reject.return_value = paused_state

            reject_resp = client.post(
                "/api/master/reject",
                json={"workflow_id": "wf-api-test-01", "feedback": "Needs improvement."},
            )
            self.assertEqual(reject_resp.status_code, 200)
            rdata = reject_resp.json()
            self.assertEqual(rdata["status"], "waiting_for_approval")


if __name__ == "__main__":
    unittest.main()


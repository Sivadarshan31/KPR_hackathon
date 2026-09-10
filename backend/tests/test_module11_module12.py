import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.services.export_service import ExportService, sanitize_filename
from app.agents.action import ActionAgent
from app.workflow.contentforge_graph import (
    ContentForgeWorkflow,
    build_contentforge_graph,
    MAX_VALIDATION_ATTEMPTS,
)
from app.schemas import (
    UserRequest,
    SourceUnderstanding,
    ContentStrategy,
    LinkedInContent,
    InstagramContent,
    AdvisoryContent,
    GeneratedContent,
    ValidationResult,
    ValidationStatus,
    ActionResult,
    MasterDecision,
)
from app.llm.exceptions import GroqConfigurationError, GroqRequestError


class TestModule11ExportFormatting(unittest.TestCase):
    """Tests for Module 11: Multi-Format Export Rendering & Safety (Markdown, HTML, JSON, Path Traversal Protection)."""

    def setUp(self):
        self.gc = GeneratedContent(
            linkedin=LinkedInContent(
                hook="LinkedIn Hook Text",
                body="LinkedIn Body Paragraph.",
                cta="Comment below!",
                hashtags=["#AI", "#Tech"],
            ),
            instagram=InstagramContent(
                caption="Instagram Caption Text",
                slides=["Slide 1: Intro", "Slide 2: Main Point"],
                hashtags=["#InstaAI"],
                call_to_action="Save this post!",
            ),
            advisory=AdvisoryContent(
                title="Advisory Brief Title",
                summary="Advisory Executive Summary.",
                important_information=["Metric A is 40%", "Metric B is 2026"],
                recommended_actions=["Action Item 1", "Action Item 2"],
                warning="Risk warning notice.",
            ),
        )

    def test_01_markdown_export_format(self):
        md = ExportService.to_markdown(self.gc, platform="all")
        self.assertIn("# ContentForge Generated Content Export", md)
        self.assertIn("## LinkedIn Post", md)
        self.assertIn("LinkedIn Hook Text", md)
        self.assertIn("## Instagram Post", md)
        self.assertIn("1. Slide 1: Intro", md)
        self.assertIn("## Advisory Briefing", md)
        self.assertIn("Advisory Executive Summary.", md)

    def test_02_html_export_format_and_xss_prevention(self):
        malicious_gc = GeneratedContent(
            linkedin=LinkedInContent(
                hook="<script>alert('xss')</script>",
                body="<img src='x' onerror='alert(1)'>",
                cta="Click <a href='javascript:alert(1)'>here</a>",
                hashtags=["#XSS"],
            )
        )
        html_out = ExportService.to_html(malicious_gc, platform="linkedin")
        self.assertIn("<!DOCTYPE html>", html_out)
        self.assertNotIn("<script>", html_out)
        self.assertIn("&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;", html_out)
        self.assertNotIn("onerror='alert(1)'", html_out)

    def test_03_json_export_format(self):
        json_str = ExportService.to_json(self.gc)
        self.assertIn('"hook": "LinkedIn Hook Text"', json_str)
        self.assertIn('"caption": "Instagram Caption Text"', json_str)

    def test_04_filename_sanitization_and_path_traversal_prevention(self):
        # Unsafe traversal inputs
        self.assertEqual(sanitize_filename("../../etc/passwd"), "etc_passwd")
        self.assertEqual(sanitize_filename("..\\..\\Windows\\System32\\cmd.exe"), "Windows_System32_cmd.exe")
        self.assertEqual(sanitize_filename("my_export\x00file.md"), "my_exportfile.md")

    def test_05_format_export_bundle(self):
        bundle = ExportService.format_export(self.gc, platform="linkedin", export_format="markdown")
        self.assertEqual(bundle["export_format"], "markdown")
        self.assertTrue(bundle["filename"].endswith(".md"))
        self.assertIn("rendered_content", bundle)

        with self.assertRaises(ValueError):
            ExportService.format_export(self.gc, export_format="unsupported_pdf")


class TestModule12IntegrationAndLiveTesting(unittest.IsolatedAsyncioTestCase):
    """Tests for Module 12: Comprehensive Integration, Workflow Paths, REST Endpoints, and Live Smoke Tests."""

    def setUp(self):
        self.client = TestClient(app)
        self.mock_su = SourceUnderstanding(
            title="Title", summary="Summary", main_topic="Topic",
            key_points=[], facts=[], entities=[], important_numbers=[],
            dates=[], claims=[], terminology=[],
        )
        self.mock_cs = ContentStrategy(summary="S", overall_angle="A", target_audience="Audience", key_takeaway="K")
        self.mock_gc = GeneratedContent(linkedin=LinkedInContent(hook="H", body="B", cta="C", hashtags=[]))
        self.mock_val_pass = ValidationResult(status=ValidationStatus.PASS, passed=True, score=95.0, issues=[], suggestions=[])
        self.mock_val_fail = ValidationResult(status=ValidationStatus.FAIL, passed=False, score=40.0, issues=["Mismatch"], suggestions=["Fix"])

    def test_06_health_and_docs_smoke_test(self):
        """Verify GET /health, GET /docs, and OpenAPI schema generation."""
        resp_health = self.client.get("/health")
        self.assertEqual(resp_health.status_code, 200)
        self.assertEqual(resp_health.json()["status"], "ok")

        resp_docs = self.client.get("/docs")
        self.assertEqual(resp_docs.status_code, 200)

        resp_openapi = self.client.get("/openapi.json")
        self.assertEqual(resp_openapi.status_code, 200)
        self.assertIn("ContentForge", resp_openapi.json()["info"]["title"])

    @patch("app.workflow.contentforge_graph.master_agent.adecide")
    @patch("app.workflow.contentforge_graph.source_understanding_agent.aunderstand")
    @patch("app.workflow.contentforge_graph.content_strategy_agent.astrategize")
    @patch("app.workflow.contentforge_graph.content_generation_agent.agenerate")
    @patch("app.workflow.contentforge_graph.validation_agent.avalidate")
    @patch("app.workflow.contentforge_graph.action_agent.aexecute")
    async def test_07_full_workflow_successful_path(
        self, mock_act, mock_val, mock_gen, mock_strat, mock_src, mock_master
    ):
        mock_master.return_value = MasterDecision(
            user_request="req", intent="i", requires_source_understanding=True,
            requires_strategy=True, requires_generation=True, requires_validation=True,
            requires_action=True, requested_action="preview", requires_human_approval=False, reasoning_summary="r"
        )
        mock_src.return_value = self.mock_su
        mock_strat.return_value = self.mock_cs
        mock_gen.return_value = self.mock_gc
        mock_val.return_value = self.mock_val_pass
        mock_act.return_value = ActionResult(
            success=True, action="preview", platform="all", action_allowed=True,
            status="completed", message="Success", validation_status="PASS"
        )

        wf = ContentForgeWorkflow()
        state = await wf.arun(user_request="req", source_text="Source text", workflow_id="wf-full-pass")
        self.assertEqual(state["status"], "completed")

    @patch("app.workflow.contentforge_graph.master_agent.adecide")
    @patch("app.workflow.contentforge_graph.source_understanding_agent.aunderstand")
    @patch("app.workflow.contentforge_graph.content_strategy_agent.astrategize")
    @patch("app.workflow.contentforge_graph.content_generation_agent.agenerate")
    @patch("app.workflow.contentforge_graph.validation_agent.avalidate")
    @patch("app.workflow.contentforge_graph.default_groq_manager.ainvoke")
    @patch("app.workflow.contentforge_graph._parse_generation_json")
    async def test_08_revision_and_recovery_flow(
        self, mock_parse, mock_ainvoke, mock_val, mock_gen, mock_strat, mock_src, mock_master
    ):
        mock_master.return_value = MasterDecision(
            user_request="req", intent="i", requires_source_understanding=True,
            requires_strategy=True, requires_generation=True, requires_validation=True,
            requires_action=False, requires_human_approval=False, reasoning_summary="r"
        )
        mock_src.return_value = self.mock_su
        mock_strat.return_value = self.mock_cs
        mock_gen.return_value = self.mock_gc
        mock_val.side_effect = [self.mock_val_fail, self.mock_val_pass]
        mock_parse.return_value = GeneratedContent(linkedin="Revised copy")
        mock_ainvoke.return_value = MagicMock(content="{}")

        wf = ContentForgeWorkflow()
        state = await wf.arun(user_request="req", source_text="Source text", workflow_id="wf-revision-recovery")
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["revision_count"], 1)

    @patch("app.workflow.contentforge_graph.master_agent.adecide")
    @patch("app.workflow.contentforge_graph.source_understanding_agent.aunderstand")
    @patch("app.workflow.contentforge_graph.content_strategy_agent.astrategize")
    @patch("app.workflow.contentforge_graph.content_generation_agent.agenerate")
    @patch("app.workflow.contentforge_graph.validation_agent.avalidate")
    @patch("app.workflow.contentforge_graph.default_groq_manager.ainvoke")
    @patch("app.workflow.contentforge_graph._parse_generation_json")
    async def test_09_max_retry_validation_failed_flow(
        self, mock_parse, mock_ainvoke, mock_val, mock_gen, mock_strat, mock_src, mock_master
    ):
        mock_master.return_value = MasterDecision(
            user_request="req", intent="i", requires_source_understanding=True,
            requires_strategy=True, requires_generation=True, requires_validation=True,
            requires_action=False, requires_human_approval=False, reasoning_summary="r"
        )
        mock_src.return_value = self.mock_su
        mock_strat.return_value = self.mock_cs
        mock_gen.return_value = self.mock_gc
        # Always fail validation
        mock_val.return_value = self.mock_val_fail
        mock_parse.return_value = self.mock_gc
        mock_ainvoke.return_value = MagicMock(content="{}")

        wf = ContentForgeWorkflow()
        state = await wf.arun(user_request="req", source_text="Source text", workflow_id="wf-max-retry")
        self.assertEqual(state["status"], "validation_failed")
        self.assertEqual(state["revision_count"], MAX_VALIDATION_ATTEMPTS)

    def test_10_action_api_export_endpoint(self):
        payload = {
            "action": "export",
            "platform": "linkedin",
            "export_format": "markdown",
            "content": {
                "linkedin": {
                    "hook": "Hook text",
                    "body": "Body text",
                    "cta": "CTA text",
                    "hashtags": ["#Tech"],
                }
            },
            "validation_status": "PASS",
        }
        resp = self.client.post("/api/agents/action", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["action"], "export")
        self.assertIn("rendered_content", data["exported_data"])


if __name__ == "__main__":
    unittest.main()

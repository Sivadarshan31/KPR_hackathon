import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.agents import (
    SourceUnderstandingAgent,
    source_understanding_agent,
    ContentStrategyAgent,
    content_strategy_agent,
    ContentGenerationAgent,
    content_generation_agent,
    ValidationAgent,
    validation_agent,
    ActionAgent,
    action_agent,
    MasterAgent,
    master_agent,
)
from app.schemas import (
    UserRequest,
    SourceUnderstanding,
    ContentStrategy,
    LinkedInStrategy,
    InstagramStrategy,
    AdvisoryStrategy,
    LinkedInContent,
    InstagramContent,
    AdvisoryContent,
    GeneratedContent,
    ValidationResult,
    ValidationStatus,
    ActionResult,
    MasterDecision,
)


class TestModule5AgentLayer(unittest.TestCase):
    """Tests for Module 5: Agent Layer interfaces, typed contracts, and decoupling."""

    def test_01_agent_imports_and_singletons(self):
        self.assertIsInstance(source_understanding_agent, SourceUnderstandingAgent)
        self.assertIsInstance(content_strategy_agent, ContentStrategyAgent)
        self.assertIsInstance(content_generation_agent, ContentGenerationAgent)
        self.assertIsInstance(validation_agent, ValidationAgent)
        self.assertIsInstance(action_agent, ActionAgent)
        self.assertIsInstance(master_agent, MasterAgent)

    @patch("app.agents.source_understanding._parse_source_json")
    def test_02_source_understanding_agent_interface(self, mock_parse):
        mock_result = SourceUnderstanding(
            title="Mock Title",
            summary="Mock summary",
            main_topic="AI",
            key_points=["Point 1"],
            facts=["Fact 1"],
            entities=["Org A"],
            important_numbers=["42%"],
            dates=["2026"],
            claims=["Claim A"],
            terminology=["Term A"],
        )
        mock_parse.return_value = mock_result

        mock_manager = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "{\"title\": \"Mock Title\"}"
        mock_manager.invoke.return_value = mock_response

        agent = SourceUnderstandingAgent(manager=mock_manager)
        res = agent.understand("Some raw text to analyze")
        self.assertEqual(res.title, "Mock Title")
        self.assertEqual(res.important_numbers, ["42%"])

    @patch("app.agents.content_strategy._parse_strategy_json")
    def test_03_content_strategy_agent_interface(self, mock_parse):
        mock_strategy = ContentStrategy(
            summary="Overall strategy",
            overall_angle="Tech Leader",
            target_audience="Devs",
            key_takeaway="AI efficiency",
            linkedin=LinkedInStrategy(
                objective="Awareness",
                audience="CEOs",
                angle="Cost reduction",
                key_message="Save time",
                tone="Professional",
                cta="Discuss below",
            ),
        )
        mock_parse.return_value = mock_strategy

        mock_manager = MagicMock()
        mock_manager.invoke.return_value = MagicMock(content="{}")

        su_input = SourceUnderstanding(
            title="Title",
            summary="Summary",
            main_topic="Topic",
            key_points=[],
            facts=[],
            entities=[],
            important_numbers=[],
            dates=[],
            claims=[],
            terminology=[],
        )

        agent = ContentStrategyAgent(manager=mock_manager)
        res = agent.strategize(su_input)
        self.assertEqual(res.overall_angle, "Tech Leader")
        self.assertIsNotNone(res.linkedin)

    @patch("app.agents.content_generation._parse_generation_json")
    def test_04_content_generation_agent_interface(self, mock_parse):
        mock_gen = GeneratedContent(
            linkedin=LinkedInContent(
                hook="Big news!",
                body="AI transforms workflows.",
                cta="Thoughts?",
                hashtags=["#AI"],
            ),
            instagram=InstagramContent(
                caption="Insta caption",
                slides=["Slide 1"],
                hashtags=["#Insta"],
                call_to_action="Save this",
            ),
        )
        mock_parse.return_value = mock_gen

        mock_manager = MagicMock()
        mock_manager.invoke.return_value = MagicMock(content="{}")

        su = SourceUnderstanding(title="T", summary="S", main_topic="M", key_points=[], facts=[], entities=[], important_numbers=[], dates=[], claims=[], terminology=[])
        cs = ContentStrategy(summary="S", overall_angle="A", target_audience="A", key_takeaway="K")

        agent = ContentGenerationAgent(manager=mock_manager)
        res = agent.generate(su, cs)
        self.assertEqual(res.linkedin.hook, "Big news!")
        self.assertEqual(res.instagram.caption, "Insta caption")

    @patch("app.agents.validation._parse_validation_json")
    def test_05_validation_agent_interface(self, mock_parse):
        mock_val = ValidationResult(
            status=ValidationStatus.PASS,
            passed=True,
            score=90.0,
            issues=[],
            suggestions=[],
        )
        mock_parse.return_value = mock_val

        mock_manager = MagicMock()
        mock_manager.invoke.return_value = MagicMock(content="{}")

        su = SourceUnderstanding(title="T", summary="S", main_topic="M", key_points=[], facts=[], entities=[], important_numbers=[], dates=[], claims=[], terminology=[])
        gc = GeneratedContent(linkedin="Post content")

        agent = ValidationAgent(manager=mock_manager)
        res = agent.validate(su, gc)
        self.assertTrue(res.passed)
        self.assertEqual(res.score, 90.0)

    def test_06_action_agent_preview_export_publish(self):
        agent = ActionAgent()
        gc = GeneratedContent(linkedin="Post text")

        # Preview
        res_preview = agent.execute(action="preview", platform="linkedin", content=gc, validation_status="PASS")
        self.assertEqual(res_preview.action, "preview")
        self.assertTrue(res_preview.success)

        # Export
        res_export = agent.execute(action="export", platform="linkedin", content=gc, validation_status="PASS")
        self.assertEqual(res_export.action, "export")
        self.assertIsNotNone(res_export.exported_data)

        # Publish (Validated PASS)
        res_pub_pass = agent.execute(action="publish", platform="linkedin", content=gc, validation_status="PASS")
        self.assertEqual(res_pub_pass.status, "dry_run")
        self.assertTrue(res_pub_pass.action_allowed)

        # Publish (Blocked FAIL)
        res_pub_fail = agent.execute(action="publish", platform="linkedin", content=gc, validation_status="FAIL", validation_issues=["Hallucination"])
        self.assertEqual(res_pub_fail.status, "blocked")
        self.assertFalse(res_pub_fail.action_allowed)

    @patch("app.agents.master_agent._parse_decision_json")
    def test_07_master_agent_routing(self, mock_parse):
        mock_decision = MasterDecision(
            user_request="Create and publish a LinkedIn post from this text.",
            intent="generate_and_publish",
            requires_source_understanding=True,
            requires_strategy=True,
            requires_generation=True,
            requires_validation=True,
            requires_action=True,
            requested_action="publish",
            requires_human_approval=True,
            target_platforms=["linkedin"],
            reasoning_summary="User asked to generate and publish LinkedIn post.",
        )
        mock_parse.return_value = mock_decision

        mock_manager = MagicMock()
        mock_manager.invoke.return_value = MagicMock(content="{}")

        agent = MasterAgent(manager=mock_manager)
        res = agent.decide("Create and publish a LinkedIn post from this text.")
        self.assertTrue(res.requires_source_understanding)
        self.assertEqual(res.requested_action, "publish")


class TestModule6ModularAPIs(unittest.TestCase):
    """Tests for Module 6: Modular Single-Agent REST API Endpoints via FastAPI TestClient."""

    def setUp(self):
        self.client = TestClient(app)

    @patch("app.agents.source_understanding.source_understanding_agent.aunderstand")
    def test_01_api_source_understanding_success(self, mock_aunderstand):
        mock_aunderstand.return_value = SourceUnderstanding(
            title="Sample Title",
            summary="Sample Summary",
            main_topic="Topic",
            key_points=["K1"],
            facts=["F1"],
            entities=["E1"],
            important_numbers=["10"],
            dates=["2026"],
            claims=["C1"],
            terminology=["T1"],
        )
        resp = self.client.post(
            "/api/agents/source-understanding",
            json={"source_text": "Valid source text payload for testing."},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["title"], "Sample Title")

    def test_02_api_source_understanding_validation_error(self):
        resp = self.client.post(
            "/api/agents/source-understanding",
            json={"source_text": "   "},
        )
        self.assertEqual(resp.status_code, 422)

    @patch("app.agents.content_strategy.content_strategy_agent.astrategize")
    def test_03_api_content_strategy_success(self, mock_astrategize):
        mock_astrategize.return_value = ContentStrategy(
            summary="Strategy summary",
            overall_angle="Angle",
            target_audience="Audience",
            key_takeaway="Takeaway",
        )
        payload = {
            "source_understanding": {
                "title": "T", "summary": "S", "main_topic": "M",
                "key_points": [], "facts": [], "entities": [],
                "important_numbers": [], "dates": [], "claims": [], "terminology": []
            }
        }
        resp = self.client.post("/api/agents/content-strategy", json=payload)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["overall_angle"], "Angle")

    @patch("app.agents.content_generation.content_generation_agent.agenerate")
    def test_04_api_content_generation_success(self, mock_agenerate):
        mock_agenerate.return_value = GeneratedContent(linkedin="LinkedIn copy")
        payload = {
            "source_understanding": {
                "title": "T", "summary": "S", "main_topic": "M",
                "key_points": [], "facts": [], "entities": [],
                "important_numbers": [], "dates": [], "claims": [], "terminology": []
            },
            "content_strategy": {
                "summary": "S", "overall_angle": "A", "target_audience": "Audience", "key_takeaway": "K"
            }
        }
        resp = self.client.post("/api/agents/content-generation", json=payload)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["linkedin"], "LinkedIn copy")

    @patch("app.agents.validation.validation_agent.avalidate")
    def test_05_api_validation_success(self, mock_avalidate):
        mock_avalidate.return_value = ValidationResult(
            status=ValidationStatus.PASS,
            passed=True,
            score=95.0,
            issues=[],
            suggestions=[],
        )
        payload = {
            "source_understanding": {
                "title": "T", "summary": "S", "main_topic": "M",
                "key_points": [], "facts": [], "entities": [],
                "important_numbers": [], "dates": [], "claims": [], "terminology": []
            },
            "generated_content": {"linkedin": "Post text"}
        }
        resp = self.client.post("/api/agents/validation", json=payload)
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["passed"])

    def test_06_api_action_success_and_blocked(self):
        payload_pass = {
            "action": "publish",
            "platform": "linkedin",
            "content": {"linkedin": "Valid copy"},
            "validation_status": "PASS",
        }
        resp = self.client.post("/api/agents/action", json=payload_pass)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "dry_run")

        payload_fail = {
            "action": "publish",
            "platform": "linkedin",
            "content": {"linkedin": "Invalid copy"},
            "validation_status": "FAIL",
            "validation_issues": ["Hallucination"],
        }
        resp_blocked = self.client.post("/api/agents/action", json=payload_fail)
        self.assertEqual(resp_blocked.status_code, 200)
        self.assertEqual(resp_blocked.json()["status"], "blocked")


if __name__ == "__main__":
    unittest.main()

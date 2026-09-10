import unittest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from app.schemas.content_generation import GeneratedContent
from app.schemas.action import ActionRequest, ActionResult
from app.agents.action import ActionAgent
from app.main import app


class TestActionAgent(unittest.IsolatedAsyncioTestCase):
    """
    Unit test suite for Agent 5 (Action / Publishing Agent).
    """

    def setUp(self):
        self.sample_content = GeneratedContent(
            linkedin="Validated LinkedIn post ready for action.",
        )
        self.agent = ActionAgent()

    def test_01_preview_action(self):
        """Test preview tool returns content unmodified."""
        result = self.agent.execute(
            action="preview",
            platform="linkedin",
            content=self.sample_content,
            validation_status="PASS",
        )
        self.assertTrue(result.success)
        self.assertTrue(result.action_allowed)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.action, "preview")
        self.assertEqual(result.content.linkedin, self.sample_content.linkedin)

    def test_02_export_action(self):
        """Test export tool generates structured export payload."""
        result = self.agent.execute(
            action="export",
            platform="linkedin",
            content=self.sample_content,
            validation_status="PASS",
        )
        self.assertTrue(result.success)
        self.assertEqual(result.action, "export")
        self.assertIsNotNone(result.exported_data)
        self.assertEqual(result.exported_data["target_platform"], "linkedin")

    def test_03_publish_gate_blocked_on_fail(self):
        """Strict validation gate: ActionAgent MUST block publishing if validation failed."""
        result = self.agent.execute(
            action="publish",
            platform="linkedin",
            content=self.sample_content,
            validation_status="FAIL",
            validation_issues=["Unsupported claim detected."],
        )
        self.assertFalse(result.success)
        self.assertFalse(result.action_allowed)
        self.assertEqual(result.status, "blocked")
        self.assertIn("failed validation", result.message.lower())
        self.assertEqual(len(result.validation_issues), 1)

    def test_04_publish_dry_run_on_pass(self):
        """Test publish executes safe dry-run simulation when content passes validation."""
        result = self.agent.execute(
            action="publish",
            platform="linkedin",
            content=self.sample_content,
            validation_status="PASS",
        )
        self.assertTrue(result.success)
        self.assertTrue(result.action_allowed)
        self.assertEqual(result.status, "dry_run")
        self.assertIn("dry-run", result.message.lower())

    def test_05_unsupported_action_rejected(self):
        """Test unsupported action is safely rejected."""
        result = self.agent.execute(
            action="unsupported_delete",
            platform="linkedin",
            content=self.sample_content,
            validation_status="PASS",
        )
        self.assertFalse(result.success)
        self.assertFalse(result.action_allowed)
        self.assertEqual(result.status, "failed")

    def test_06_api_endpoint_success(self):
        """Test POST /api/agents/action endpoint."""
        client = TestClient(app)
        payload = {
            "action": "preview",
            "platform": "linkedin",
            "content": self.sample_content.model_dump(),
            "validation_status": "PASS",
        }
        resp = client.post("/api/agents/action", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["status"], "completed")


if __name__ == "__main__":
    unittest.main()

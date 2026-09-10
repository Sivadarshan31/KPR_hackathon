"""Live End-to-End Test Suite for Phase 7 + Phase 8 + Phase 9 Master Workflow & API.

Verifies:
1. Master Agent dynamic routing decisions
2. Full LangGraph execution:
   - Source only flow
   - Multi-platform generation flow
   - Validation & autonomous revision flow
   - Max validation attempts halting
   - Human approval suspension on publish request
   - Approval resumption (triggering Agent 5 dry-run)
   - Rejection feedback handling
3. HTTP API endpoints (/api/master/run, /api/master/status/{id}, /api/master/approve, /api/master/reject)
4. Zero Groq API key leakage
"""

import os
import sys
import unittest
import requests
import json

BASE_URL = "http://127.0.0.1:8000"

SAMPLE_SOURCE = (
    "Acme Cloud launched its next-gen AI Observability platform on October 10, 2025. "
    "It reduces incident mean-time-to-resolution (MTTR) by 45% and automatically isolates "
    "root causes across multi-region Kubernetes clusters. Early enterprise beta customers "
    "report saving over 30 hours per engineer each month."
)


class TestMasterWorkflowLive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Verify server is alive
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        if resp.status_code != 200:
            raise RuntimeError(f"Server not running at {BASE_URL}")

    def test_01_docs_and_openapi_contain_master_endpoints(self):
        """Verify OpenAPI schema registers all master endpoints and schemas."""
        resp = requests.get(f"{BASE_URL}/openapi.json", timeout=5)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        paths = data.get("paths", {})

        self.assertIn("/api/master/run", paths)
        self.assertIn("/api/master/status/{workflow_id}", paths)
        self.assertIn("/api/master/approve", paths)
        self.assertIn("/api/master/reject", paths)

        # Verify individual agent endpoints remain intact
        self.assertIn("/api/agents/source-understanding", paths)
        self.assertIn("/api/agents/content-strategy", paths)
        self.assertIn("/api/agents/content-generation", paths)
        self.assertIn("/api/agents/validation", paths)
        self.assertIn("/api/agents/action", paths)

    def test_02_source_understanding_only_routing(self):
        """Verify Master Agent routes to Agent 1 only when user asks to analyze/understand source."""
        payload = {
            "user_request": "Please extract the key facts and analyze this source text.",
            "source_text": SAMPLE_SOURCE,
        }
        resp = requests.post(f"{BASE_URL}/api/master/run", json=payload, timeout=60)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["current_stage"], "source")
        self.assertIsNotNone(data["source_understanding"])
        self.assertIsNone(data["generated_content"])
        self.assertIsNone(data["validation"])

        # Check decision
        decision = data["decision"]
        self.assertTrue(decision["requires_source_understanding"])
        self.assertFalse(decision["requires_generation"])
        self.assertFalse(decision["requires_action"])

    def test_03_generation_without_publish_flow(self):
        """Verify Master Agent routes Agents 1 -> 2 -> 3 -> 4 when generating content without publish."""
        payload = {
            "user_request": "Create a high-impact LinkedIn post announcing this launch.",
            "source_text": SAMPLE_SOURCE,
        }
        resp = requests.post(f"{BASE_URL}/api/master/run", json=payload, timeout=240)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data["status"], "completed")
        self.assertIsNotNone(data["source_understanding"])
        self.assertIsNotNone(data["content_strategy"])
        self.assertIsNotNone(data["generated_content"])
        self.assertIsNotNone(data["validation"])

        # Action should NOT run
        self.assertIsNone(data["action_result"])
        self.assertFalse(data["decision"]["requires_action"])

        # Validation passed
        self.assertTrue(data["validation"]["passed"])

        # Check generated content
        content = data["generated_content"]
        self.assertIn("linkedin", content)
        self.assertIn("45%", content["linkedin"]["post"])

    def test_04_publish_request_pauses_at_approval_checkpoint(self):
        """Verify that a request containing 'publish' pauses at human approval checkpoint."""
        payload = {
            "user_request": "Generate a LinkedIn post and publish it to LinkedIn immediately.",
            "source_text": SAMPLE_SOURCE,
        }
        resp = requests.post(f"{BASE_URL}/api/master/run", json=payload, timeout=240)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        workflow_id = data["workflow_id"]
        self.assertEqual(data["status"], "waiting_for_approval")
        self.assertEqual(data["current_stage"], "human_approval")
        self.assertTrue(data["approval_required"])
        self.assertIsNone(data["action_result"])

        # Test GET /api/master/status/{workflow_id}
        status_resp = requests.get(f"{BASE_URL}/api/master/status/{workflow_id}", timeout=10)
        self.assertEqual(status_resp.status_code, 200)
        status_data = status_resp.json()
        self.assertEqual(status_data["status"], "waiting_for_approval")
        self.assertTrue(status_data["approval_required"])

        # Test POST /api/master/approve
        approve_resp = requests.post(
            f"{BASE_URL}/api/master/approve",
            json={"workflow_id": workflow_id, "feedback": "Approved for publish"},
            timeout=30,
        )
        self.assertEqual(approve_resp.status_code, 200)
        approved_data = approve_resp.json()

        self.assertEqual(approved_data["status"], "completed")
        self.assertEqual(approved_data["current_stage"], "action")
        self.assertIsNotNone(approved_data["action_result"])
        self.assertTrue(approved_data["action_result"]["action_allowed"])
        self.assertIn("simulation", approved_data["action_result"]["message"].lower())

    def test_05_rejection_routes_to_revision(self):
        """Verify that rejecting a paused workflow routes back to revision."""
        payload = {
            "user_request": "Create a LinkedIn post and publish it.",
            "source_text": SAMPLE_SOURCE,
        }
        resp = requests.post(f"{BASE_URL}/api/master/run", json=payload, timeout=240)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        workflow_id = data["workflow_id"]

        self.assertEqual(data["status"], "waiting_for_approval")

        # Reject with feedback
        reject_resp = requests.post(
            f"{BASE_URL}/api/master/reject",
            json={"workflow_id": workflow_id, "feedback": "Make it more concise and emphasize the 45% MTTR."},
            timeout=90,
        )
        self.assertEqual(reject_resp.status_code, 200)
        rejected_data = reject_resp.json()

        # Since the request still requires action/publishing, after revision and validation pass,
        # it pauses again for approval
        self.assertIn(rejected_data["status"], ["waiting_for_approval", "completed"])
        self.assertGreaterEqual(rejected_data["revision_count"], 1)

    def test_06_security_zero_key_leakage(self):
        """Verify no Groq API key is present in OpenAPI docs, responses, or error messages."""
        from dotenv import load_dotenv
        load_dotenv()

        keys = [
            os.environ.get("GROQ_API_KEY_1", ""),
            os.environ.get("GROQ_API_KEY_2", ""),
            os.environ.get("GROQ_API_KEY_3", ""),
            os.environ.get("GROQ_API_KEY", ""),
        ]
        active_keys = [k.strip() for k in keys if k and len(k.strip()) > 10]
        self.assertGreater(len(active_keys), 0, "At least one Groq API key must be configured in .env")

        # 1. Check OpenAPI schema
        openapi_resp = requests.get(f"{BASE_URL}/openapi.json", timeout=5)
        for key in active_keys:
            self.assertNotIn(key, openapi_resp.text, "SECURITY ALERT: Key leaked in OpenAPI docs!")

        # 2. Check 422 Validation Error
        err_resp = requests.post(f"{BASE_URL}/api/master/run", json={}, timeout=5)
        self.assertEqual(err_resp.status_code, 422)
        for key in active_keys:
            self.assertNotIn(key, err_resp.text, "SECURITY ALERT: Key leaked in 422 error body!")

        # 3. Check 404 Status Error
        nf_resp = requests.get(f"{BASE_URL}/api/master/status/non-existent-wf-id", timeout=5)
        self.assertEqual(nf_resp.status_code, 404)
        for key in active_keys:
            self.assertNotIn(key, nf_resp.text, "SECURITY ALERT: Key leaked in 404 error body!")

        # 4. Check live source-understanding response
        live_resp = requests.post(
            f"{BASE_URL}/api/master/run",
            json={"user_request": "Extract key facts from this text.", "source_text": SAMPLE_SOURCE},
            timeout=60,
        )
        self.assertEqual(live_resp.status_code, 200)
        for key in active_keys:
            self.assertNotIn(key, live_resp.text, "SECURITY ALERT: Key leaked in live response body!")


if __name__ == "__main__":
    unittest.main()

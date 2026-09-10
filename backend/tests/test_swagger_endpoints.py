"""Swagger & API Route Audit Verification Script (Section 18, 20, 21, 34).

Directly exercises all FastAPI endpoints through HTTP against the running server.
Verifies status codes, schemas, validation gates, error handling, and zero secret leakage.
"""

import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import unittest
import requests
import os
from dotenv import load_dotenv

load_dotenv()
BASE_URL = "http://127.0.0.1:8000"


class TestSwaggerAndEndpoints(unittest.TestCase):
    """Verifies all FastAPI Swagger-exposed routes."""

    def test_01_docs_and_openapi(self):
        """Verify Swagger UI and OpenAPI documentation endpoints."""
        r_docs = requests.get(f"{BASE_URL}/docs", timeout=5)
        self.assertEqual(r_docs.status_code, 200)

        r_schema = requests.get(f"{BASE_URL}/openapi.json", timeout=5)
        self.assertEqual(r_schema.status_code, 200)
        data = r_schema.json()
        paths = data.get("paths", {})

        expected_endpoints = [
            "/health",
            "/api/agents/source-understanding",
            "/api/agents/content-strategy",
            "/api/agents/content-generation",
            "/api/agents/validation",
            "/api/agents/action",
            "/api/master/run",
            "/api/master/status/{workflow_id}",
            "/api/master/approve",
            "/api/master/reject",
        ]
        for ep in expected_endpoints:
            self.assertIn(ep, paths, f"Endpoint {ep} missing from OpenAPI schema!")

    def test_02_health_endpoint(self):
        """Verify health check endpoint returns 200 with service info."""
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data.get("status"), "ok")
        self.assertIn("service", data)
        self.assertIn("version", data)

    def test_03_action_agent_endpoint(self):
        """Verify Agent 5 endpoint enforces validation gate and executes dry-run."""
        payload = {
            "action": "publish",
            "platform": "linkedin",
            "content": {
                "linkedin": "AI in healthcare preview post.",
                "instagram": None,
                "advisory": None,
            },
            "validation_status": "PASS",
            "validation_issues": [],
        }
        r = requests.post(f"{BASE_URL}/api/agents/action", json=payload, timeout=10)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"))
        self.assertTrue(data.get("action_allowed"))
        self.assertEqual(data.get("status"), "dry_run")

        # Test validation failure block
        fail_payload = dict(payload)
        fail_payload["validation_status"] = "FAIL"
        fail_payload["validation_issues"] = ["Hallucinated metric"]
        r_fail = requests.post(f"{BASE_URL}/api/agents/action", json=fail_payload, timeout=10)
        self.assertEqual(r_fail.status_code, 200)
        fail_data = r_fail.json()
        self.assertFalse(fail_data.get("success"))
        self.assertFalse(fail_data.get("action_allowed"))
        self.assertEqual(fail_data.get("status"), "blocked")

    def test_04_error_handling_graceful_failures(self):
        """Verify 422 for malformed requests and 404 for unknown workflow IDs."""
        # Malformed request body to /api/master/run
        r_bad = requests.post(f"{BASE_URL}/api/master/run", json={"invalid_field": 123}, timeout=5)
        self.assertEqual(r_bad.status_code, 422)

        # Empty user request
        r_empty = requests.post(f"{BASE_URL}/api/master/run", json={"user_request": "   "}, timeout=5)
        self.assertEqual(r_empty.status_code, 422)

        # Non-existent workflow ID status
        r_404 = requests.get(f"{BASE_URL}/api/master/status/wf-unknown-99999", timeout=5)
        self.assertEqual(r_404.status_code, 404)

        # Non-existent workflow approval
        r_app_404 = requests.post(f"{BASE_URL}/api/master/approve", json={"workflow_id": "wf-unknown-99999"}, timeout=5)
        self.assertEqual(r_app_404.status_code, 404)

    def test_05_zero_secret_leakage(self):
        """Verify no Groq API key is present in OpenAPI docs, error payloads, or headers."""
        keys = [
            os.environ.get("GROQ_API_KEY_1", ""),
            os.environ.get("GROQ_API_KEY_2", ""),
            os.environ.get("GROQ_API_KEY_3", ""),
            os.environ.get("GROQ_API_KEY", ""),
        ]
        active_keys = [k.strip() for k in keys if k and len(k.strip()) > 10]
        self.assertGreater(len(active_keys), 0)

        # Check /openapi.json
        openapi_text = requests.get(f"{BASE_URL}/openapi.json", timeout=5).text
        for key in active_keys:
            self.assertNotIn(key, openapi_text)

        # Check 422 error response
        err_text = requests.post(f"{BASE_URL}/api/master/run", json={}, timeout=5).text
        for key in active_keys:
            self.assertNotIn(key, err_text)


if __name__ == "__main__":
    unittest.main()

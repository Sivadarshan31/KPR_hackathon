"""End-to-End Audit Verification Test for Section 19 & Section 33.

Tests the full ContentForge agentic pipeline on realistic healthcare pilot source:
Source -> Source Understanding -> Content Strategy -> Content Generation ->
Validation -> Validation Loop -> Human Approval Gate -> Action Agent -> Final Result
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import unittest
from app.workflow.contentforge_graph import ContentForgeWorkflow
from app.schemas.master import MasterDecision
from app.schemas.source_understanding import SourceUnderstanding
from app.schemas.content_strategy import ContentStrategy
from app.schemas.content_generation import GeneratedContent
from app.schemas.validation import ValidationResult
from app.schemas.action import ActionResult

REALISTIC_SOURCE = (
    "Artificial intelligence is transforming healthcare. "
    "A six-month pilot program analyzed 50,000 medical images. "
    "The system was designed to assist doctors with medical image analysis. "
    "The organization reported improved workflow efficiency during the pilot."
)

NATURAL_LANGUAGE_REQUEST = (
    "Take this source and create a professional LinkedIn post and Instagram content. "
    "Analyze the source first, create an appropriate strategy, generate the content, "
    "validate it against the source, and ask for approval before any publishing action."
)


class TestAuditEndToEnd(unittest.IsolatedAsyncioTestCase):
    """End-to-end verification of ContentForge agent pipeline."""

    async def test_complete_pipeline_flow(self):
        wf = ContentForgeWorkflow()
        workflow_id = "wf-e2e-audit-001"

        print("\n[Audit E2E] Running full workflow with natural-language prompt...")
        state = await wf.arun(
            user_request=NATURAL_LANGUAGE_REQUEST,
            source_text=REALISTIC_SOURCE,
            workflow_id=workflow_id,
        )

        # 1. Verify Master Decision
        decision: MasterDecision = state.get("decision")
        self.assertIsNotNone(decision, "Master decision must be populated.")
        print(f"[Audit E2E] Master intent: {decision.intent}")
        print(f"[Audit E2E] Requested platforms: {decision.requested_platforms}")
        self.assertTrue(decision.requires_source_understanding, "Must require source understanding.")
        self.assertTrue(decision.requires_strategy, "Must require strategy.")
        self.assertTrue(decision.requires_generation, "Must require generation.")
        self.assertTrue(decision.requires_validation, "Must require validation.")
        self.assertTrue(decision.requires_action, "Must require action for publishing.")
        self.assertTrue(decision.requires_human_approval, "Must require human approval for publish.")

        # 2. Verify Agent 1 (Source Understanding)
        su: SourceUnderstanding = state.get("source_understanding")
        self.assertIsNotNone(su, "Source understanding must be generated.")
        print(f"[Audit E2E] Agent 1 extracted title: {su.title}")
        print(f"[Audit E2E] Agent 1 numbers: {su.important_numbers}")
        # Verify 50,000 is preserved
        numbers_text = " ".join(su.important_numbers)
        self.assertTrue("50,000" in numbers_text or "50000" in numbers_text or "50,000" in str(su.facts),
                        "Expected 50,000 medical images to be captured in facts/numbers.")

        # 3. Verify Agent 2 (Content Strategy)
        cs: ContentStrategy = state.get("content_strategy")
        self.assertIsNotNone(cs, "Content strategy must be generated.")
        print(f"[Audit E2E] Agent 2 angle: {cs.overall_angle}")
        self.assertIsNotNone(cs.linkedin, "LinkedIn strategy must exist.")
        self.assertIsNotNone(cs.instagram, "Instagram strategy must exist.")

        # 4. Verify Agent 3 (Content Generation)
        gc: GeneratedContent = state.get("generated_content")
        self.assertIsNotNone(gc, "Generated content must be created.")
        self.assertIsNotNone(gc.linkedin, "LinkedIn copy must exist.")
        self.assertIsNotNone(gc.instagram, "Instagram copy must exist.")
        print(f"[Audit E2E] Agent 3 LinkedIn sample: {gc.linkedin[:80]}...")
        print(f"[Audit E2E] Agent 3 Instagram sample: {gc.instagram.caption[:80]}...")

        # 5. Verify Agent 4 (Validation)
        val: ValidationResult = state.get("validation_result")
        self.assertIsNotNone(val, "Validation result must be produced.")
        print(f"[Audit E2E] Agent 4 validation score: {val.score}, passed: {val.passed}")
        self.assertTrue(val.passed, "Generated content must pass factual validation.")
        self.assertGreaterEqual(val.score, 0.8, "Passing validation score must be >= 0.8.")

        # 6. Verify Human Approval Gate
        self.assertEqual(state.get("status"), "waiting_for_approval",
                         "Workflow must pause at waiting_for_approval before publishing.")
        self.assertEqual(state.get("current_stage"), "human_approval")
        self.assertTrue(state.get("human_approval_required"))
        self.assertEqual(state.get("human_approval_status"), "pending")
        self.assertIsNone(state.get("action_result"), "Action agent must NOT run before approval.")

        # 7. Resume with Human Approval
        print("[Audit E2E] Providing human approval to resume workflow...")
        approved_state = await wf.aapprove(workflow_id)

        # 8. Verify Agent 5 (Action Agent)
        self.assertEqual(approved_state.get("status"), "completed")
        self.assertEqual(approved_state.get("current_stage"), "action")
        action_res: ActionResult = approved_state.get("action_result")
        self.assertIsNotNone(action_res, "Action result must be present after approval.")
        self.assertTrue(action_res.action_allowed, "Action must be allowed for validated content.")
        self.assertEqual(action_res.status, "dry_run", "Publishing must execute dry-run simulation.")
        print(f"[Audit E2E] Agent 5 Action result: {action_res.message}")
        print("[Audit E2E] FULL PIPELINE TEST COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    unittest.main()

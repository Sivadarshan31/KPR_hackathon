import logging
from typing import Any, Dict, List, Optional

from app.schemas.content_generation import GeneratedContent
from app.schemas.action import ActionResult

logger = logging.getLogger(__name__)


class ActionAgent:
    """
    Agent 5: Action / Publishing Agent.
    Executes preview, export, and publication workflows on validated content.
    Enforces a strict server-side validation gate to block unvalidated or failed content.
    """

    def preview_content(
        self,
        content: GeneratedContent,
        platform: str,
        validation_status: str,
        validation_issues: List[str],
    ) -> ActionResult:
        """
        Preview Tool: Returns the generated content unmodified for user inspection.
        """
        logger.info("Executing preview tool for platform '%s'.", platform)
        return ActionResult(
            success=True,
            action="preview",
            platform=platform,
            action_allowed=True,
            status="completed",
            message=f"Preview generated successfully for platform '{platform}'.",
            content=content,
            validation_status=validation_status,
            validation_issues=validation_issues,
        )

    def export_content(
        self,
        content: GeneratedContent,
        platform: str,
        validation_status: str,
        validation_issues: List[str],
    ) -> ActionResult:
        """
        Export Tool: Packages validated content into an export-ready structured dictionary.
        """
        logger.info("Executing export tool for platform '%s'.", platform)
        export_payload: Dict[str, Any] = {
            "format_version": "1.0",
            "target_platform": platform,
            "export_bundle": content.model_dump(exclude_none=True),
        }
        return ActionResult(
            success=True,
            action="export",
            platform=platform,
            action_allowed=True,
            status="completed",
            message=f"Content successfully packaged for export on platform '{platform}'.",
            content=content,
            validation_status=validation_status,
            validation_issues=validation_issues,
            exported_data=export_payload,
        )

    def publish_content(
        self,
        content: GeneratedContent,
        platform: str,
        validation_status: str,
        validation_issues: List[str],
    ) -> ActionResult:
        """
        Publish Tool: Validates gate status and stages publishing.
        Strictly blocks any content that did not pass validation.
        Provides a transparent dry-run simulation when live publishing credentials are unconfigured.
        """
        logger.info("Evaluating publish tool for platform '%s'. Validation status: %s", platform, validation_status)
        
        # Strict Validation Gate: Block if validation failed
        if validation_status.upper() != "PASS":
            logger.warning("Publish action blocked: validation status is '%s'.", validation_status)
            return ActionResult(
                success=False,
                action="publish",
                platform=platform,
                action_allowed=False,
                status="blocked",
                message="Publishing blocked: Content failed validation checks. Please review issues and revise.",
                content=content,
                validation_status=validation_status,
                validation_issues=validation_issues,
            )

        # Content passed validation: Execute safe dry-run simulation
        logger.info("Validation passed. Executing publish dry-run simulation for platform '%s'.", platform)
        return ActionResult(
            success=True,
            action="publish",
            platform=platform,
            action_allowed=True,
            status="dry_run",
            message="Dry-run simulation: Content validated and staged for publishing. (External live social-media API integration not connected in MVP).",
            content=content,
            validation_status=validation_status,
            validation_issues=[],
        )

    def execute(
        self,
        action: str,
        platform: str,
        content: GeneratedContent,
        validation_status: str,
        validation_issues: Optional[List[str]] = None,
    ) -> ActionResult:
        """
        Synchronously routes to the requested deterministic tool based on action type.
        """
        issues = validation_issues or []
        act = action.strip().lower()

        if act == "preview":
            return self.preview_content(content, platform, validation_status, issues)
        elif act == "export":
            return self.export_content(content, platform, validation_status, issues)
        elif act == "publish":
            return self.publish_content(content, platform, validation_status, issues)
        else:
            return ActionResult(
                success=False,
                action=action,
                platform=platform,
                action_allowed=False,
                status="failed",
                message=f"Unsupported action: '{action}'. Allowed: preview, export, publish.",
                content=content,
                validation_status=validation_status,
                validation_issues=issues,
            )

    async def aexecute(
        self,
        action: str,
        platform: str,
        content: GeneratedContent,
        validation_status: str,
        validation_issues: Optional[List[str]] = None,
    ) -> ActionResult:
        """
        Asynchronous wrapper for action execution.
        """
        return self.execute(
            action=action,
            platform=platform,
            content=content,
            validation_status=validation_status,
            validation_issues=validation_issues,
        )


# Reusable singleton instance
action_agent = ActionAgent()

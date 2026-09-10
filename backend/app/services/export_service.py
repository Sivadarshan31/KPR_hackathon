import html
import json
import os
import re
from typing import Any, Dict, Optional
from app.schemas.content_generation import (
    GeneratedContent,
    LinkedInContent,
    InstagramContent,
    AdvisoryContent,
)


def sanitize_filename(filename: str, default: str = "contentforge_export") -> str:
    """
    Sanitizes user/platform-supplied filenames to prevent path traversal
    and directory breakout vulnerabilities.
    Strips directory separators, null bytes, and non-alphanumeric characters.
    """
    if not filename or not filename.strip():
        return default

    # Remove path traversal sequences (../, ..\, absolute paths, null bytes)
    cleaned = filename.replace("\x00", "").replace("/", "_").replace("\\", "_")
    cleaned = re.sub(r"\.\.+", "_", cleaned)
    # Strip any remaining unsafe characters
    cleaned = re.sub(r"[^\w\.\-]", "_", cleaned)
    # Ensure baseline safety
    cleaned = cleaned.strip("._")
    return cleaned if cleaned else default


class ExportService:
    """
    Service layer for converting validated GeneratedContent into multi-format outputs
    (Markdown, HTML, JSON) and generating safe export bundles.
    """

    @staticmethod
    def to_markdown(content: GeneratedContent, platform: str = "all") -> str:
        """
        Converts GeneratedContent into structured Markdown.
        """
        md_lines = ["# ContentForge Generated Content Export\n"]

        # LinkedIn Format
        if platform in ("linkedin", "all") and content.linkedin:
            md_lines.append("## LinkedIn Post\n")
            if isinstance(content.linkedin, LinkedInContent):
                md_lines.append(f"### {content.linkedin.hook}\n")
                md_lines.append(f"{content.linkedin.body}\n")
                if content.linkedin.cta:
                    md_lines.append(f"**Call to Action:** {content.linkedin.cta}\n")
                if content.linkedin.hashtags:
                    tags = " ".join(content.linkedin.hashtags)
                    md_lines.append(f"*{tags}*\n")
            else:
                md_lines.append(f"{content.linkedin}\n")

        # Instagram Format
        if platform in ("instagram", "all") and content.instagram:
            md_lines.append("## Instagram Post\n")
            ig = content.instagram
            if ig.caption:
                md_lines.append(f"{ig.caption}\n")
            if ig.slides:
                md_lines.append("### Carousel Slides\n")
                for idx, slide in enumerate(ig.slides, start=1):
                    md_lines.append(f"{idx}. {slide}")
                md_lines.append("")
            if ig.call_to_action:
                md_lines.append(f"**Call to Action:** {ig.call_to_action}\n")
            if ig.hashtags:
                tags = " ".join(ig.hashtags)
                md_lines.append(f"*{tags}*\n")

        # Advisory Format
        if platform in ("advisory", "all") and content.advisory:
            md_lines.append("## Advisory Briefing\n")
            if isinstance(content.advisory, AdvisoryContent):
                adv = content.advisory
                md_lines.append(f"### {adv.title}\n")
                md_lines.append(f"{adv.summary}\n")
                if adv.important_information:
                    md_lines.append("#### Key Information\n")
                    for info in adv.important_information:
                        md_lines.append(f"- {info}")
                    md_lines.append("")
                if adv.recommended_actions:
                    md_lines.append("#### Recommended Actions\n")
                    for act in adv.recommended_actions:
                        md_lines.append(f"- {act}")
                    md_lines.append("")
                if adv.warning:
                    md_lines.append(f"> **WARNING:** {adv.warning}\n")
            else:
                md_lines.append(f"{content.advisory}\n")

        return "\n".join(md_lines).strip()

    @staticmethod
    def to_html(content: GeneratedContent, platform: str = "all") -> str:
        """
        Converts GeneratedContent into valid HTML with HTML escaping to prevent XSS.
        """
        html_parts = [
            "<!DOCTYPE html>",
            "<html lang=\"en\">",
            "<head>",
            '<meta charset="UTF-8">',
            "<title>ContentForge Content Export</title>",
            "<style>body{font-family:sans-serif;line-height:1.6;margin:2rem;} h1,h2{color:#1e293b;} blockquote{background:#f1f5f9;padding:10px;border-left:4px solid #ef4444;}</style>",
            "</head>",
            "<body>",
            "<h1>ContentForge Content Export</h1>",
        ]

        # LinkedIn
        if platform in ("linkedin", "all") and content.linkedin:
            html_parts.append("<section class=\"linkedin\">")
            html_parts.append("<h2>LinkedIn Post</h2>")
            if isinstance(content.linkedin, LinkedInContent):
                hook = html.escape(content.linkedin.hook)
                body = html.escape(content.linkedin.body).replace("\n", "<br>")
                html_parts.append(f"<h3>{hook}</h3>")
                html_parts.append(f"<p>{body}</p>")
                if content.linkedin.cta:
                    cta = html.escape(content.linkedin.cta)
                    html_parts.append(f"<p><strong>Call to Action:</strong> {cta}</p>")
                if content.linkedin.hashtags:
                    tags = html.escape(" ".join(content.linkedin.hashtags))
                    html_parts.append(f"<p><em>{tags}</em></p>")
            else:
                body = html.escape(str(content.linkedin)).replace("\n", "<br>")
                html_parts.append(f"<p>{body}</p>")
            html_parts.append("</section>")

        # Instagram
        if platform in ("instagram", "all") and content.instagram:
            ig = content.instagram
            html_parts.append("<section class=\"instagram\">")
            html_parts.append("<h2>Instagram Post</h2>")
            if ig.caption:
                cap = html.escape(ig.caption).replace("\n", "<br>")
                html_parts.append(f"<p>{cap}</p>")
            if ig.slides:
                html_parts.append("<h3>Carousel Slides</h3><ol>")
                for slide in ig.slides:
                    s_esc = html.escape(slide)
                    html_parts.append(f"<li>{s_esc}</li>")
                html_parts.append("</ol>")
            if ig.call_to_action:
                cta = html.escape(ig.call_to_action)
                html_parts.append(f"<p><strong>Call to Action:</strong> {cta}</p>")
            if ig.hashtags:
                tags = html.escape(" ".join(ig.hashtags))
                html_parts.append(f"<p><em>{tags}</em></p>")
            html_parts.append("</section>")

        # Advisory
        if platform in ("advisory", "all") and content.advisory:
            html_parts.append("<section class=\"advisory\">")
            html_parts.append("<h2>Advisory Briefing</h2>")
            if isinstance(content.advisory, AdvisoryContent):
                adv = content.advisory
                title = html.escape(adv.title)
                summary = html.escape(adv.summary)
                html_parts.append(f"<h3>{title}</h3>")
                html_parts.append(f"<p>{summary}</p>")
                if adv.important_information:
                    html_parts.append("<h4>Key Information</h4><ul>")
                    for info in adv.important_information:
                        html_parts.append(f"<li>{html.escape(info)}</li>")
                    html_parts.append("</ul>")
                if adv.recommended_actions:
                    html_parts.append("<h4>Recommended Actions</h4><ul>")
                    for act in adv.recommended_actions:
                        html_parts.append(f"<li>{html.escape(act)}</li>")
                    html_parts.append("</ul>")
                if adv.warning:
                    warn = html.escape(adv.warning)
                    html_parts.append(f"<blockquote><strong>WARNING:</strong> {warn}</blockquote>")
            else:
                body = html.escape(str(content.advisory)).replace("\n", "<br>")
                html_parts.append(f"<p>{body}</p>")
            html_parts.append("</section>")

        html_parts.append("</body></html>")
        return "\n".join(html_parts)

    @staticmethod
    def to_json(content: GeneratedContent) -> str:
        """
        Converts GeneratedContent into formatted JSON.
        """
        return json.dumps(content.model_dump(exclude_none=True), indent=2)

    @classmethod
    def format_export(
        cls,
        content: GeneratedContent,
        platform: str = "all",
        export_format: str = "markdown",
        custom_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Builds a safe, formatted export bundle dictionary.
        """
        fmt = export_format.strip().lower()
        if fmt not in {"markdown", "md", "html", "json"}:
            raise ValueError(f"Unsupported export format: '{export_format}'. Supported: markdown, html, json.")

        base_name = custom_filename or f"contentforge_export_{platform}"
        safe_name = sanitize_filename(base_name)

        if fmt in ("markdown", "md"):
            rendered = cls.to_markdown(content, platform)
            file_extension = ".md"
            mime_type = "text/markdown"
        elif fmt == "html":
            rendered = cls.to_html(content, platform)
            file_extension = ".html"
            mime_type = "text/html"
        else:  # json
            rendered = cls.to_json(content)
            file_extension = ".json"
            mime_type = "application/json"

        filename = f"{safe_name}{file_extension}"

        return {
            "format_version": "1.0",
            "target_platform": platform,
            "export_format": fmt,
            "filename": filename,
            "mime_type": mime_type,
            "rendered_content": rendered,
            "structured_bundle": content.model_dump(exclude_none=True),
        }


export_service = ExportService()

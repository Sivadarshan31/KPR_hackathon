import json
import urllib.request
import urllib.error
import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://127.0.0.1:8000"
AGENT1_ENDPOINT = f"{BASE_URL}/api/agents/source-understanding"
AGENT2_ENDPOINT = f"{BASE_URL}/api/agents/content-strategy"
AGENT3_ENDPOINT = f"{BASE_URL}/api/agents/content-generation"
PIPELINE_AGENT1_AGENT2_ENDPOINT = f"{BASE_URL}/api/pipeline/source-to-strategy"
PIPELINE_FULL_ENDPOINT = f"{BASE_URL}/api/pipeline/content-generation"


import time


def post_json(url, data, max_retries=3):
    req_data = json.dumps(data).encode("utf-8")
    for attempt in range(max_retries):
        req = urllib.request.Request(
            url,
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                status = resp.status
                body = json.loads(resp.read().decode("utf-8"))
                return status, body
        except urllib.error.HTTPError as err:
            status = err.code
            try:
                body = json.loads(err.read().decode("utf-8"))
            except Exception:
                body = {"detail": str(err.reason)}
            if status == 429 and attempt < max_retries - 1:
                wait_s = 20 * (attempt + 1)
                print(f"[!] Hit 429 rate limit. Waiting {wait_s}s for token quota window replenishment (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait_s)
                continue
            return status, body
        except Exception as exc:
            return 500, {"detail": str(exc)}
    return status, body


def run_live_tests():
    print("=" * 80)
    print("CONTENTFORGE: LIVE SWAGGER & AGENT 3 / FULL PIPELINE VERIFICATION")
    print("=" * 80)

    results = {}
    collected_responses = []

    # 0. Health & Docs Verification
    print("\n--- 0. Health & Swagger OpenAPI Verification ---")
    try:
        with urllib.request.urlopen(f"{BASE_URL}/health") as resp:
            assert resp.status == 200
            print("[+] Health check: PASS (HTTP 200)")
        with urllib.request.urlopen(f"{BASE_URL}/docs") as resp:
            assert resp.status == 200
            print("[+] Swagger UI (/docs): PASS (HTTP 200)")
        with urllib.request.urlopen(f"{BASE_URL}/openapi.json") as resp:
            schema = json.loads(resp.read().decode("utf-8"))
            paths = schema.get("paths", {})
            assert "/api/agents/source-understanding" in paths
            assert "/api/agents/content-strategy" in paths
            assert "/api/agents/content-generation" in paths
            assert "/api/pipeline/source-to-strategy" in paths
            assert "/api/pipeline/content-generation" in paths
            print("[+] OpenAPI registered endpoints:")
            print("    - POST /api/agents/source-understanding")
            print("    - POST /api/agents/content-strategy")
            print("    - POST /api/agents/content-generation")
            print("    - POST /api/pipeline/source-to-strategy")
            print("    - POST /api/pipeline/content-generation")
            results["Swagger UI & Endpoints"] = "PASS"
    except Exception as exc:
        print(f"[-] Swagger / Health verification failed: {exc}")
        results["Swagger UI & Endpoints"] = "FAIL"
        sys.exit(1)

    # -------------------------------------------------------------------------
    # TEST A: Agent 3 Independently (POST /api/agents/content-generation)
    # -------------------------------------------------------------------------
    print("\n--- TEST A: Agent 3 Independently ---")
    test_su = {
        "title": "AI-Powered Healthcare Diagnostic Pilot",
        "summary": "An AI-powered healthcare diagnostic platform was launched in 2026, achieving a 40% reduction in processing time across 50,000 patient records.",
        "main_topic": "AI healthcare diagnostic pilot",
        "key_points": [
            "AI-powered healthcare platform launched in 2026",
            "Achieved a 40% reduction in processing time",
            "Tested on 50,000 patient records",
        ],
        "facts": [
            "Platform launched in 2026",
            "Reduced processing time by 40%",
            "Evaluated with 50,000 patient records",
        ],
        "entities": ["AI Healthcare Platform"],
        "important_numbers": ["40%", "50,000", "2026"],
        "dates": ["2026"],
        "claims": [],
        "terminology": ["diagnostic platform", "patient records"],
        "target_audience": "Healthcare technology professionals",
        "tone": "Professional",
        "source_type": "Report",
    }

    test_strat = {
        "summary": "Multi-channel launch campaign highlighting clinical efficiency gains.",
        "overall_angle": "How AI diagnostics delivered a 40% reduction in processing time in 2026.",
        "target_audience": "Hospital CIOs, clinical IT leaders, and healthcare technologists.",
        "key_takeaway": "Clinically validated AI delivers measurable speed improvements without compromising care.",
        "linkedin": {
            "objective": "Establish thought leadership in health tech efficiency",
            "audience": "Healthcare technology executives and hospital leaders",
            "angle": "Operational efficiency meets patient care in 2026",
            "key_message": "A 40% reduction in processing time across 50,000 records proves the viability of clinical AI.",
            "tone": "Authoritative and professional",
            "cta": "How is your hospital system approaching AI workflow automation? Share your thoughts below.",
            "recommended_structure": ["Hook", "The 40% efficiency metric", "Clinical impact", "Discussion CTA"],
        },
        "instagram": {
            "objective": "Visually showcase healthcare AI transformation",
            "audience": "Tech and health enthusiasts",
            "angle": "Can AI speed up diagnosis by 40%?",
            "carousel_direction": [
                "Slide 1: Can AI cut healthcare wait times?",
                "Slide 2: The 2026 pilot breakdown",
                "Slide 3: 40% faster processing across 50,000 records",
                "Slide 4: What this means for patients",
                "Slide 5: Follow for the future of medicine",
            ],
            "visual_direction": "Clean medical blue palette with bold typography and data visualizations",
            "tone": "Engaging and inspiring",
            "cta": "Save this post to share with someone interested in healthcare innovation.",
        },
        "advisory": {
            "objective": "Brief hospital executive committee on pilot results",
            "audience": "Chief Medical Officers and Chief Information Officers",
            "key_information": [
                "40% reduction in record processing time",
                "Pilot validated across 50,000 records in 2026",
            ],
            "priority": "High",
            "tone": "Objective and executive",
            "recommended_structure": ["Executive Summary", "Key Metrics", "Strategic Implications"],
        },
    }

    st_a, body_a = post_json(AGENT3_ENDPOINT, {
        "source_understanding": test_su,
        "content_strategy": test_strat,
    })
    collected_responses.append(body_a)
    print(f"Status: {st_a}")
    linkedin_a = body_a.get("linkedin")
    instagram_a = body_a.get("instagram", {})
    advisory_a = body_a.get("advisory")

    print(f"LinkedIn preview: {linkedin_a[:140] if linkedin_a else None}...")
    print(f"Instagram caption preview: {instagram_a.get('caption', '')[:100]}...")
    print(f"Instagram slides count: {len(instagram_a.get('slides', [])) if instagram_a.get('slides') else 0}")
    print(f"Instagram hashtags: {instagram_a.get('hashtags', [])}")
    print(f"Advisory preview: {advisory_a[:140] if advisory_a else None}...")

    # Verification criteria
    test_a_pass = (
        st_a == 200
        and bool(linkedin_a)
        and ("40%" in linkedin_a or "40 percent" in linkedin_a.lower())
        and bool(instagram_a.get("caption"))
        and len(instagram_a.get("slides", [])) >= 3
        and bool(advisory_a)
        and ("40%" in advisory_a or "50,000" in advisory_a)
    )
    results["Agent 3 Independent"] = "PASS" if test_a_pass else "FAIL"
    print(f"Result: {results['Agent 3 Independent']}")

    # -------------------------------------------------------------------------
    # TEST B: Agent 2 -> Agent 3 Connection (Dynamic Flow)
    # -------------------------------------------------------------------------
    print("\n--- TEST B: Agent 2 -> Agent 3 Connection ---")
    # Take real Agent 2 output generated from test_su and feed it into Agent 3
    st_b2, body_b2 = post_json(AGENT2_ENDPOINT, {"source_understanding": test_su})
    collected_responses.append(body_b2)
    print(f"Agent 2 call status: {st_b2}")

    st_b3, body_b3 = post_json(AGENT3_ENDPOINT, {
        "source_understanding": test_su,
        "content_strategy": body_b2,
    })
    collected_responses.append(body_b3)
    print(f"Agent 3 call status (consuming Agent 2 output): {st_b3}")
    linkedin_b = body_b3.get("linkedin")
    instagram_b = body_b3.get("instagram", {})
    advisory_b = body_b3.get("advisory")

    test_b_pass = (
        st_b2 == 200
        and st_b3 == 200
        and bool(linkedin_b)
        and bool(instagram_b.get("caption"))
        and bool(advisory_b)
    )
    results["Agent 2 -> Agent 3 Connection"] = "PASS" if test_b_pass else "FAIL"
    print(f"Result: {results['Agent 2 -> Agent 3 Connection']}")

    # -------------------------------------------------------------------------
    # TEST C: Full Pipeline (Agent 1 -> Agent 2 -> Agent 3)
    # Using the prompt's exact University Workshop test input
    # -------------------------------------------------------------------------
    print("\n--- TEST C: Full Pipeline (POST /api/pipeline/content-generation) ---")
    university_source = (
        "A university technology club organized a two-day artificial intelligence workshop in 2026 "
        "for undergraduate students. The workshop covered machine learning fundamentals, prompt engineering, "
        "and practical AI applications. More than 150 students participated in the event. The organizers also "
        "conducted hands-on activities to help students build small AI-powered applications."
    )

    st_c, body_c = post_json(PIPELINE_FULL_ENDPOINT, {"source_text": university_source})
    collected_responses.append(body_c)
    print(f"Status: {st_c}")

    su_c = body_c.get("source_understanding", {})
    cs_c = body_c.get("content_strategy", {})
    gc_c = body_c.get("generated_content", {})

    print(f"\n[Agent 1 Output in Pipeline]")
    print(f"Title: {su_c.get('title')}")
    print(f"Important numbers: {su_c.get('important_numbers')}")
    print(f"Entities: {su_c.get('entities')}")
    print(f"Key Points: {su_c.get('key_points')}")

    print(f"\n[Agent 2 Output in Pipeline]")
    print(f"Overall Angle: {cs_c.get('overall_angle')}")
    print(f"LinkedIn Angle: {cs_c.get('linkedin', {}).get('angle')}")
    print(f"Instagram Visual Direction: {cs_c.get('instagram', {}).get('visual_direction')}")
    print(f"Advisory Priority: {cs_c.get('advisory', {}).get('priority')}")

    print(f"\n[Agent 3 Output in Pipeline]")
    linkedin_c = gc_c.get("linkedin")
    instagram_c = gc_c.get("instagram", {})
    advisory_c = gc_c.get("advisory")

    print(f"LinkedIn Post:\n{linkedin_c}\n")
    print(f"Instagram Caption:\n{instagram_c.get('caption')}\n")
    print(f"Instagram Slides: {instagram_c.get('slides')}\n")
    print(f"Instagram Hashtags: {instagram_c.get('hashtags')}\n")
    print(f"Instagram CTA: {instagram_c.get('call_to_action')}\n")
    print(f"Advisory Post:\n{advisory_c}\n")

    # Verify Agent 1 identified key concepts
    su_str = json.dumps(su_c).lower()
    has_workshop = "workshop" in su_str or "ai" in su_str
    has_150 = "150" in su_str
    has_2026 = "2026" in su_str

    # Verify Agent 3 generated all 3 formats preserving source numbers
    gc_str = json.dumps(gc_c).lower()
    has_150_in_gc = "150" in gc_str
    has_2026_in_gc = "2026" in gc_str
    has_slides = isinstance(instagram_c.get("slides"), list) and len(instagram_c.get("slides")) >= 2

    test_c_pass = (
        st_c == 200
        and has_workshop
        and has_150
        and has_2026
        and bool(linkedin_c)
        and bool(instagram_c.get("caption"))
        and has_slides
        and bool(advisory_c)
        and has_150_in_gc
        and has_2026_in_gc
    )
    results["Agent 1 -> Agent 2 -> Agent 3 Pipeline"] = "PASS" if test_c_pass else "FAIL"
    print(f"Result: {results['Agent 1 -> Agent 2 -> Agent 3 Pipeline']}")

    # -------------------------------------------------------------------------
    # TEST D: Anti-Hallucination Negative Test (Section 26)
    # Source mentions workshop in 2026, strategy mentions "revenue"
    # Agent 3 must NOT invent revenue!
    # -------------------------------------------------------------------------
    print("\n--- TEST D: Anti-Hallucination Negative Test ---")
    sparse_su = {
        "title": "University AI Workshop",
        "summary": "Our university hosted an AI workshop in 2026.",
        "main_topic": "AI workshop",
        "key_points": ["University hosted an AI workshop in 2026"],
        "facts": ["Workshop hosted in 2026"],
        "entities": ["University"],
        "important_numbers": ["2026"],
        "dates": ["2026"],
        "claims": [],
        "terminology": ["artificial intelligence"],
        "target_audience": "University community",
        "tone": "Neutral",
        "source_type": "Announcement",
    }

    hallucination_bait_strat = {
        "summary": "Promote workshop financial success and revenue.",
        "overall_angle": "Highlight the workshop's financial revenue and ticket sales in 2026.",
        "target_audience": "Investors and university leadership.",
        "key_takeaway": "The workshop was a massive financial success.",
        "linkedin": {
            "objective": "Showcase financial results",
            "audience": "Investors",
            "angle": "Financial return from the 2026 AI workshop",
            "key_message": "Include the workshop's revenue, profit numbers, and ticket sales.",
            "tone": "Professional",
            "cta": "Reach out to discuss funding.",
            "recommended_structure": ["Revenue metrics", "Profit overview"],
        },
        "instagram": {
            "objective": "Event promotion",
            "audience": "Students",
            "angle": "Workshop highlights",
            "carousel_direction": ["Slide 1: Highlights"],
            "visual_direction": "Photos",
            "tone": "Dynamic",
            "cta": "Check our bio.",
        },
        "advisory": {
            "objective": "Budget report",
            "audience": "Leadership",
            "key_information": ["Report total revenue and profit generated"],
            "priority": "Medium",
            "tone": "Executive",
            "recommended_structure": ["Financial summary"],
        },
    }

    st_d, body_d = post_json(AGENT3_ENDPOINT, {
        "source_understanding": sparse_su,
        "content_strategy": hallucination_bait_strat,
    })
    collected_responses.append(body_d)
    print(f"Status: {st_d}")
    body_d_text = json.dumps(body_d).lower()

    # Verify Agent 3 did NOT invent dollar amounts or fake currency numbers like $50,000 or $100,000
    invented_dollar = "$" in body_d_text or "usd" in body_d_text or "profit of" in body_d_text
    print(f"Did Agent 3 invent dollar / financial numbers? {invented_dollar}")

    test_d_pass = st_d == 200 and not invented_dollar
    results["Anti-Hallucination (No Invented Facts)"] = "PASS" if test_d_pass else "FAIL"
    print(f"Result: {results['Anti-Hallucination (No Invented Facts)']}")

    # -------------------------------------------------------------------------
    # TEST E: Platform Selectivity (Section 27)
    # -------------------------------------------------------------------------
    print("\n--- TEST E: Platform Selectivity ---")
    # Strategy that explicitly only asks for LinkedIn
    single_platform_strat = {
        "summary": "LinkedIn thought leadership post.",
        "overall_angle": "Professional reflections on AI in 2026.",
        "target_audience": "Tech professionals.",
        "key_takeaway": "Continuous education is vital.",
        "linkedin": {
            "objective": "Professional awareness",
            "audience": "Tech professionals",
            "angle": "Why practical learning drives tech adoption",
            "key_message": "150 students learned AI in 2026.",
            "tone": "Thoughtful",
            "cta": "Join the discussion.",
            "recommended_structure": ["Hook", "Insight", "CTA"],
        },
        "instagram": {
            "objective": "",
            "audience": "",
            "angle": "",
            "carousel_direction": [],
            "visual_direction": "",
            "tone": "",
            "cta": "",
        },
        "advisory": {
            "objective": "",
            "audience": "",
            "key_information": [],
            "priority": "",
            "tone": "",
            "recommended_structure": [],
        },
    }

    st_e, body_e = post_json(AGENT3_ENDPOINT, {
        "source_understanding": sparse_su,
        "content_strategy": single_platform_strat,
    })
    collected_responses.append(body_e)
    print(f"Status: {st_e}")
    print(f"LinkedIn generated: {bool(body_e.get('linkedin'))}")
    test_e_pass = st_e == 200 and bool(body_e.get("linkedin"))
    results["Platform Selectivity"] = "PASS" if test_e_pass else "FAIL"
    print(f"Result: {results['Platform Selectivity']}")

    # -------------------------------------------------------------------------
    # TEST F: Validation Errors (HTTP 422)
    # -------------------------------------------------------------------------
    print("\n--- TEST F: Validation Errors on Empty Inputs ---")
    st_f1, _ = post_json(PIPELINE_FULL_ENDPOINT, {"source_text": ""})
    st_f2, _ = post_json(PIPELINE_FULL_ENDPOINT, {"source_text": "   "})
    st_f3, _ = post_json(AGENT3_ENDPOINT, {})
    print(f"Empty source text status: {st_f1}")
    print(f"Whitespace source text status: {st_f2}")
    print(f"Empty Agent 3 payload status: {st_f3}")

    test_f_pass = (st_f1 == 422 and st_f2 == 422 and st_f3 == 422)
    results["Validation & Error Handling"] = "PASS" if test_f_pass else "FAIL"
    print(f"Result: {results['Validation & Error Handling']}")

    # -------------------------------------------------------------------------
    # TEST G: Security - Zero Secret Leakage
    # -------------------------------------------------------------------------
    print("\n--- TEST G: Security - Zero Secret Leakage ---")
    all_responses_dump = " ".join(json.dumps(r) for r in collected_responses)
    has_secrets = (
        "gsk_" in all_responses_dump
        or "authorization" in all_responses_dump.lower()
        or "api_key" in all_responses_dump.lower()
    )
    print(f"Secret marker detected in any response: {has_secrets}")
    test_g_pass = not has_secrets
    results["Security (Zero Secret Leakage)"] = "PASS" if test_g_pass else "FAIL"
    print(f"Result: {results['Security (Zero Secret Leakage)']}")

    # Summary
    print("\n" + "=" * 80)
    print("FINAL PHASE 4 / PHASE 4->3 TEST SUMMARY")
    print("=" * 80)
    all_passed = True
    for test_name, res in results.items():
        print(f"{test_name:40}: {res}")
        if res != "PASS":
            all_passed = False

    return all_passed


if __name__ == "__main__":
    success = run_live_tests()
    sys.exit(0 if success else 1)

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
PIPELINE_ENDPOINT = f"{BASE_URL}/api/pipeline/source-to-strategy"


def post_json(url, data):
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
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
        return status, body


def run_live_tests():
    print("=" * 80)
    print("CONTENTFORGE: LIVE SWAGGER & PIPELINE VERIFICATION")
    print("=" * 80)

    results = {}

    # 0. Health & Docs
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
            assert "/api/pipeline/source-to-strategy" in paths
            print("[+] OpenAPI registered endpoints: PASS")
            results["Swagger UI & Endpoints"] = "PASS"
    except Exception as exc:
        print(f"[-] Swagger / Health verification failed: {exc}")
        results["Swagger UI & Endpoints"] = "FAIL"
        sys.exit(1)

    # -------------------------------------------------------------------------
    # TEST 1: Agent 1 Independently
    # -------------------------------------------------------------------------
    print("\n--- TEST 1: Agent 1 Independently ---")
    source_1 = (
        "Artificial intelligence is transforming healthcare. A recent pilot program "
        "analyzed 50,000 medical images over six months to assist doctors with diagnosis."
    )
    st1, body1 = post_json(AGENT1_ENDPOINT, {"source_text": source_1})
    print(f"Status: {st1}")
    print(f"Title: {body1.get('title')}")
    print(f"Important numbers: {body1.get('important_numbers')}")
    print(f"Summary: {body1.get('summary')}")

    test1_pass = (
        st1 == 200
        and bool(body1.get("title"))
        and any("50,000" in s or "50000" in s for s in body1.get("important_numbers", []))
    )
    results["Agent 1 Independent"] = "PASS" if test1_pass else "FAIL"
    print(f"Result: {results['Agent 1 Independent']}")

    # -------------------------------------------------------------------------
    # TEST 2: Agent 2 Independently (Using Agent 1's real output)
    # -------------------------------------------------------------------------
    print("\n--- TEST 2: Agent 2 Independently ---")
    st2, body2 = post_json(AGENT2_ENDPOINT, {"source_understanding": body1})
    print(f"Status: {st2}")
    print(f"Overall Angle: {body2.get('overall_angle')}")
    print(f"LinkedIn Objective: {body2.get('linkedin', {}).get('objective')}")
    print(f"Instagram CTA: {body2.get('instagram', {}).get('cta')}")
    print(f"Advisory Priority: {body2.get('advisory', {}).get('priority')}")

    test2_pass = (
        st2 == 200
        and "linkedin" in body2
        and "instagram" in body2
        and "advisory" in body2
        and bool(body2["linkedin"].get("angle"))
        and bool(body2["instagram"].get("visual_direction"))
        and bool(body2["advisory"].get("key_information"))
    )
    results["Agent 2 Independent"] = "PASS" if test2_pass else "FAIL"
    print(f"Result: {results['Agent 2 Independent']}")

    # -------------------------------------------------------------------------
    # TEST 3: Connected Pipeline (Agent 1 -> Agent 2)
    # -------------------------------------------------------------------------
    print("\n--- TEST 3: Connected Pipeline (POST /api/pipeline/source-to-strategy) ---")
    st3, body3 = post_json(PIPELINE_ENDPOINT, {"source_text": source_1})
    print(f"Status: {st3}")
    su3 = body3.get("source_understanding", {})
    cs3 = body3.get("content_strategy", {})

    print(f"Agent 1 output in Pipeline: Title = {su3.get('title')}")
    print(f"Agent 2 output in Pipeline: Key Takeaway = {cs3.get('key_takeaway')}")
    print(f"LinkedIn Angle: {cs3.get('linkedin', {}).get('angle')}")

    test3_pass = (
        st3 == 200
        and bool(su3.get("title"))
        and bool(su3.get("important_numbers"))
        and bool(cs3.get("linkedin", {}).get("objective"))
        and bool(cs3.get("instagram", {}).get("visual_direction"))
        and bool(cs3.get("advisory", {}).get("recommended_structure"))
    )
    results["Connected Pipeline"] = "PASS" if test3_pass else "FAIL"
    print(f"Result: {results['Connected Pipeline']}")

    # -------------------------------------------------------------------------
    # TEST 4: Dynamic Test with Different Source
    # -------------------------------------------------------------------------
    print("\n--- TEST 4: Dynamic Strategy with Different Source ---")
    source_marine = (
        "A university robotics laboratory has developed an autonomous underwater vehicle "
        "for monitoring marine pollution. The vehicle uses water-quality sensors and can "
        "operate continuously for up to 18 hours."
    )
    st4, body4 = post_json(PIPELINE_ENDPOINT, {"source_text": source_marine})
    print(f"Status: {st4}")
    su4 = body4.get("source_understanding", {})
    cs4 = body4.get("content_strategy", {})

    print(f"Marine Source Title: {su4.get('title')}")
    print(f"Marine Numbers: {su4.get('important_numbers')}")
    print(f"Marine LinkedIn Objective: {cs4.get('linkedin', {}).get('objective')}")
    print(f"Marine Instagram Angle: {cs4.get('instagram', {}).get('angle')}")

    # Verify that the strategy is dynamically adapted (not the same as healthcare)
    is_distinct = (
        cs4.get("overall_angle") != cs3.get("overall_angle")
        and ("marine" in str(su4).lower() or "ocean" in str(su4).lower() or "underwater" in str(su4).lower())
        and ("marine" in str(cs4).lower() or "ocean" in str(cs4).lower() or "underwater" in str(cs4).lower() or "pollution" in str(cs4).lower())
    )
    test4_pass = st4 == 200 and is_distinct
    results["Dynamic Source Strategy"] = "PASS" if test4_pass else "FAIL"
    print(f"Result: {results['Dynamic Source Strategy']}")

    # -------------------------------------------------------------------------
    # TEST 5: Distinctive Value Flow & Anti-Hallucination
    # -------------------------------------------------------------------------
    print("\n--- TEST 5: Distinctive Value Flow (Anti-hallucination) ---")
    source_solar = (
        "The startup launched its solar monitoring platform in 2026 and reported "
        "that the pilot involved 1,250 households."
    )
    st5, body5 = post_json(PIPELINE_ENDPOINT, {"source_text": source_solar})
    print(f"Status: {st5}")
    su5 = body5.get("source_understanding", {})
    cs5 = body5.get("content_strategy", {})

    su5_str = json.dumps(su5).lower()
    cs5_str = json.dumps(cs5).lower()

    has_2026 = "2026" in su5_str
    has_households = "1,250" in su5_str or "1250" in su5_str
    has_solar = "solar" in su5_str

    # Verify Agent 2 strategy reflects these specific facts
    strat_reflects_facts = (
        ("1,250" in cs5_str or "1250" in cs5_str or "households" in cs5_str)
        and ("solar" in cs5_str)
    )

    test5_pass = st5 == 200 and has_2026 and has_households and has_solar and strat_reflects_facts
    results["Data Flow & Anti-Hallucination"] = "PASS" if test5_pass else "FAIL"
    print(f"Agent 1 captured 2026: {has_2026}, 1,250 households: {has_households}, solar: {has_solar}")
    print(f"Agent 2 reflected facts in strategy: {strat_reflects_facts}")
    print(f"Result: {results['Data Flow & Anti-Hallucination']}")

    # -------------------------------------------------------------------------
    # TEST 6: Validation Error on Empty Input
    # -------------------------------------------------------------------------
    print("\n--- TEST 6: Validation Error on Empty Input ---")
    st6, body6 = post_json(PIPELINE_ENDPOINT, {"source_text": ""})
    print(f"Status (empty string): {st6}")
    st6_ws, body6_ws = post_json(PIPELINE_ENDPOINT, {"source_text": "   "})
    print(f"Status (whitespace): {st6_ws}")

    test6_pass = st6 == 422 and st6_ws == 422
    results["Empty Input Validation"] = "PASS" if test6_pass else "FAIL"
    print(f"Result: {results['Empty Input Validation']}")

    # -------------------------------------------------------------------------
    # TEST 7: Security - Zero Secret Leakage
    # -------------------------------------------------------------------------
    print("\n--- TEST 7: Security - Zero Secret Leakage ---")
    # Check that responses across all tests never contain groq keys or authorization headers
    raw_responses = [json.dumps(b) for b in [body1, body2, body3, body4, body5, body6]]
    all_responses_text = " ".join(raw_responses)
    has_secret_marker = "gsk_" in all_responses_text or "authorization" in all_responses_text.lower()

    test7_pass = not has_secret_marker
    results["Security (No Secret Leaks)"] = "PASS" if test7_pass else "FAIL"
    print(f"Result: {results['Security (No Secret Leaks)']}")

    # Summary
    print("\n" + "=" * 80)
    print("FINAL TEST SUMMARY")
    print("=" * 80)
    all_passed = True
    for test_name, res in results.items():
        print(f"{test_name:35}: {res}")
        if res != "PASS":
            all_passed = False

    return all_passed


if __name__ == "__main__":
    success = run_live_tests()
    sys.exit(0 if success else 1)

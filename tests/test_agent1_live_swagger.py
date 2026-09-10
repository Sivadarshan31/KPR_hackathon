import json
import urllib.request
import urllib.error
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://127.0.0.1:8000"
ENDPOINT = f"{BASE_URL}/api/agents/source-understanding"

def post_json(url, data):
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req) as resp:
            status = resp.status
            body = json.loads(resp.read().decode("utf-8"))
            return status, body
    except urllib.error.HTTPError as err:
        status = err.code
        try:
            body = json.loads(err.read().decode("utf-8"))
        except Exception:
            body = err.reason
        return status, body

def run_tests():
    results = {}
    print("=" * 70)
    print("STARTING INDEPENDENT SWAGGER TESTING FOR AGENT 1 (LIVE API)")
    print("=" * 70)

    # Health check
    try:
        with urllib.request.urlopen(f"{BASE_URL}/health") as resp:
            assert resp.status == 200
            print("[+] Server and health endpoint operational (HTTP 200)")
    except Exception as e:
        print(f"[-] Server health check failed: {e}")
        sys.exit(1)

    # Check Swagger docs endpoint
    try:
        with urllib.request.urlopen(f"{BASE_URL}/docs") as resp:
            assert resp.status == 200
            print("[+] Swagger UI documentation accessible at /docs (HTTP 200)")
    except Exception as e:
        print(f"[-] Swagger UI failed to load: {e}")
        sys.exit(1)

    # -------------------------------------------------------------
    # TEST 1: Basic Source
    # -------------------------------------------------------------
    print("\n--- Running Test 1: Basic Source ---")
    t1_payload = {
        "source_text": "Artificial intelligence is being used in healthcare to assist doctors with medical image analysis. A six-month pilot program analyzed 50,000 medical images to evaluate the system."
    }
    status, body = post_json(ENDPOINT, t1_payload)
    print(f"Status: {status}")
    print(f"Response: {json.dumps(body, indent=2)}")

    t1_pass = (
        status == 200
        and "ai" in body.get("main_topic", "").lower() or "healthcare" in body.get("main_topic", "").lower() or "medical" in body.get("main_topic", "").lower()
        and any("50,000" in s or "50000" in s for s in body.get("important_numbers", []))
        and any("six-month" in s.lower() or "6-month" in s.lower() or "six month" in s.lower() or "6 month" in s.lower() for s in (body.get("important_numbers", []) + body.get("dates", []) + body.get("facts", [])))
    )
    results["Basic source"] = "PASS" if t1_pass else "FAIL"
    print(f"Test 1 Result: {results['Basic source']}")

    # -------------------------------------------------------------
    # TEST 2: Business Source
    # -------------------------------------------------------------
    print("\n--- Running Test 2: Business Source ---")
    t2_payload = {
        "source_text": "Acme Technologies announced a new solar energy platform in Chennai in 2026. The company stated that the platform can reduce electricity consumption by up to 35 percent for participating commercial buildings."
    }
    status, body = post_json(ENDPOINT, t2_payload)
    print(f"Status: {status}")
    print(f"Response: {json.dumps(body, indent=2)}")

    entities_str = " ".join(body.get("entities", [])).lower()
    numbers_str = " ".join(body.get("important_numbers", [])).lower()
    dates_str = " ".join(body.get("dates", [])).lower()
    claims_str = " ".join(body.get("claims", [])).lower()

    t2_pass = (
        status == 200
        and "acme" in entities_str
        and ("solar" in entities_str or "solar" in body.get("main_topic", "").lower() or "solar" in " ".join(body.get("terminology", [])).lower())
        and ("chennai" in entities_str or "chennai" in " ".join(body.get("facts", [])).lower())
        and "2026" in dates_str
        and "35" in numbers_str
        and ("reduce" in claims_str or "35" in claims_str or len(body.get("claims", [])) > 0)
    )
    results["Business source"] = "PASS" if t2_pass else "FAIL"
    print(f"Test 2 Result: {results['Business source']}")

    # -------------------------------------------------------------
    # TEST 3: Technical Source
    # -------------------------------------------------------------
    print("\n--- Running Test 3: Technical Source ---")
    t3_payload = {
        "source_text": "The university developed an autonomous underwater vehicle for monitoring marine pollution. The vehicle uses water-quality sensors and can operate continuously for up to 18 hours. Researchers plan to conduct field trials during 2026."
    }
    status, body = post_json(ENDPOINT, t3_payload)
    print(f"Status: {status}")
    print(f"Response: {json.dumps(body, indent=2)}")

    t3_text = json.dumps(body).lower()
    t3_pass = (
        status == 200
        and "autonomous underwater vehicle" in t3_text
        and "marine pollution" in t3_text
        and "18 hours" in " ".join(body.get("important_numbers", [])).lower()
        and "2026" in " ".join(body.get("dates", [])).lower()
    )
    results["Technical source"] = "PASS" if t3_pass else "FAIL"
    print(f"Test 3 Result: {results['Technical source']}")

    # -------------------------------------------------------------
    # TEST 4: Very Short Source
    # -------------------------------------------------------------
    print("\n--- Running Test 4: Very Short Source ---")
    t4_payload = {
        "source_text": "The university launched a robotics laboratory in 2026."
    }
    status, body = post_json(ENDPOINT, t4_payload)
    print(f"Status: {status}")
    print(f"Response: {json.dumps(body, indent=2)}")

    # Ensure no fabricated statistics/numbers or funding
    invented = ["million", "billion", "dollar", "students", "professors", "mit", "stanford", "harvard", "budget"]
    t4_has_invention = any(inv in json.dumps(body).lower() for inv in invented)
    t4_pass = (
        status == 200
        and not t4_has_invention
        and "2026" in " ".join(body.get("dates", []))
    )
    results["Short source"] = "PASS" if t4_pass else "FAIL"
    print(f"Test 4 Result: {results['Short source']}")

    # -------------------------------------------------------------
    # TEST 5: Hallucination Test
    # -------------------------------------------------------------
    print("\n--- Running Test 5: Hallucination Test ---")
    t5_payload = {
        "source_text": "Our company launched a new AI platform yesterday."
    }
    status, body = post_json(ENDPOINT, t5_payload)
    print(f"Status: {status}")
    print(f"Response: {json.dumps(body, indent=2)}")

    forbidden = ["google", "microsoft", "openai", "apple", "amazon", "$", "dollar", "revenue", "percent", "99%"]
    t5_has_forbidden = any(f in json.dumps(body).lower() for f in forbidden)
    t5_pass = status == 200 and not t5_has_forbidden
    results["Hallucination test"] = "PASS" if t5_pass else "FAIL"
    print(f"Test 5 Result: {results['Hallucination test']}")

    # -------------------------------------------------------------
    # TEST 6: Numerical Accuracy
    # -------------------------------------------------------------
    print("\n--- Running Test 6: Numerical Accuracy ---")
    t6_payload = {
        "source_text": "The pilot system processed 12,500 transactions over 45 days and achieved a reported processing speed improvement of 27.5 percent."
    }
    status, body = post_json(ENDPOINT, t6_payload)
    print(f"Status: {status}")
    print(f"Response: {json.dumps(body, indent=2)}")

    nums = " ".join(body.get("important_numbers", []))
    t6_pass = (
        status == 200
        and "12,500" in nums or "12500" in nums
        and "45 days" in nums or "45" in nums
        and "27.5" in nums
    )
    results["Numerical accuracy"] = "PASS" if t6_pass else "FAIL"
    print(f"Test 6 Result: {results['Numerical accuracy']}")

    # -------------------------------------------------------------
    # TEST 7: Dates
    # -------------------------------------------------------------
    print("\n--- Running Test 7: Dates ---")
    t7_payload = {
        "source_text": "The organization announced the project on March 15, 2026. The first phase is scheduled to begin in July 2026 and continue until December 2026."
    }
    status, body = post_json(ENDPOINT, t7_payload)
    print(f"Status: {status}")
    print(f"Response: {json.dumps(body, indent=2)}")

    dates_str = " ".join(body.get("dates", []))
    t7_pass = (
        status == 200
        and "March 15, 2026" in dates_str or "March 15" in dates_str
        and "July 2026" in dates_str
        and "December 2026" in dates_str
    )
    results["Date extraction"] = "PASS" if t7_pass else "FAIL"
    print(f"Test 7 Result: {results['Date extraction']}")

    # -------------------------------------------------------------
    # TEST 8: Empty / Invalid Input
    # -------------------------------------------------------------
    print("\n--- Running Test 8: Empty & Whitespace Input ---")
    status_empty, body_empty = post_json(ENDPOINT, {"source_text": ""})
    print(f"Empty status: {status_empty} (Response: {body_empty})")
    status_ws, body_ws = post_json(ENDPOINT, {"source_text": "     \n\t  "})
    print(f"Whitespace status: {status_ws} (Response: {body_ws})")

    t8_pass = status_empty == 422 and status_ws == 422
    results["Empty input"] = "PASS" if t8_pass else "FAIL"
    print(f"Test 8 Result: {results['Empty input']}")

    # -------------------------------------------------------------
    # TEST 9: Missing Field
    # -------------------------------------------------------------
    print("\n--- Running Test 9: Missing Field ---")
    status_missing, body_missing = post_json(ENDPOINT, {})
    print(f"Missing field status: {status_missing} (Response: {body_missing})")

    t9_pass = status_missing == 422
    results["Missing field"] = "PASS" if t9_pass else "FAIL"
    print(f"Test 9 Result: {results['Missing field']}")

    # -------------------------------------------------------------
    # TEST 10: Longer Source
    # -------------------------------------------------------------
    print("\n--- Running Test 10: Longer Multi-Paragraph Source ---")
    t10_source = """
    Nexus Robotics Inc. revealed its next-generation logistics automation framework, Project Horizon, at the International Automation Summit on January 14, 2026.

    Designed specifically for high-throughput fulfillment centers, Project Horizon integrates computer vision and collaborative mobile robots. During an intensive 90-day pilot across three distribution hubs, the framework handled over 1,200,000 package dispatches while reducing item misplacement rates by 42.8 percent.

    The primary objective is addressing seasonal labor shortages and decreasing dispatch latency from 15 minutes to under 4 minutes per parcel. However, integrating the system into older facilities with legacy conveyor software proved challenging, requiring custom middleware connectors.

    The management claimed that full enterprise deployment will yield an estimated 300 percent return on investment within two fiscal years. Full commercial rollout is slated for September 2026.
    """
    status, body = post_json(ENDPOINT, {"source_text": t10_source})
    print(f"Status: {status}")
    print(f"Response: {json.dumps(body, indent=2)}")

    t10_pass = (
        status == 200
        and len(body.get("key_points", [])) >= 3
        and len(body.get("facts", [])) >= 2
        and len(body.get("important_numbers", [])) >= 2
        and len(body.get("dates", [])) >= 2
        and len(body.get("claims", [])) >= 1
    )
    results["Long source"] = "PASS" if t10_pass else "FAIL"
    print(f"Test 10 Result: {results['Long source']}")

    # -------------------------------------------------------------
    # TEST 11: Multilingual / Non-English Source
    # -------------------------------------------------------------
    print("\n--- Running Test 11: Non-English Source ---")
    t11_source = "L'entreprise Avenir Solaire a inauguré un nouveau parc photovoltaïque à Lyon en mars 2026. L'installation produit 45 mégawatts d'énergie propre."
    status, body = post_json(ENDPOINT, {"source_text": t11_source})
    print(f"Status: {status}")
    print(f"Response: {json.dumps(body, indent=2)}")
    t11_pass = (
        status == 200
        and ("lyon" in json.dumps(body).lower())
        and ("45" in " ".join(body.get("important_numbers", [])) or "45" in json.dumps(body))
    )
    results["Multilingual source"] = "PASS" if t11_pass else "FAIL"
    print(f"Test 11 Result: {results['Multilingual source']}")

    # -------------------------------------------------------------
    # TEST 12: Groq Error Handling & Structured Output
    # -------------------------------------------------------------
    print("\n--- Running Test 12: Error Handling & Security ---")
    # Verify structured output fields
    required_fields = ["title", "summary", "main_topic", "key_points", "facts", "entities", "important_numbers", "dates", "claims", "terminology", "target_audience", "tone", "source_type"]
    has_all_fields = all(f in body for f in required_fields)
    results["Structured output"] = "PASS" if has_all_fields else "FAIL"
    print(f"Structured Output Validation: {results['Structured output']}")

    # Check that keys are not leaked in any response
    full_dump = json.dumps(body)
    has_no_keys = "gsk_" not in full_dump and "bearer" not in full_dump.lower()
    results["Error handling"] = "PASS" if has_no_keys else "FAIL"
    print(f"Secret Redaction & Error Handling: {results['Error handling']}")

    print("\n" + "=" * 70)
    print("ALL TEST RESULTS SUMMARY:")
    print("=" * 70)
    for test_name, res in results.items():
        icon = "✅" if res == "PASS" else "❌"
        print(f"{test_name.ljust(25)} {icon} ({res})")

if __name__ == "__main__":
    run_tests()

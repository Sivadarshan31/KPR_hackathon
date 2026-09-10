import sys
import unittest
from pathlib import Path

# Ensure root is in sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.config import get_configured_keys_count, GROQ_MODEL
from app.llm.groq_manager import groq_manager
from app.llm.exceptions import GroqConfigurationError, GroqRequestError


def run_diagnostics():
    print("=" * 60)
    print("CONTENTFORGE — GROQ INFRASTRUCTURE VERIFICATION")
    print("=" * 60)
    keys_count = get_configured_keys_count()
    print(f"Groq keys loaded: {keys_count}")
    print(f"Groq model configured: {GROQ_MODEL}")

    if keys_count == 0:
        print("\nNotice: 0 Groq API keys currently configured in .env.")
        print("To run the real connection test, configure GROQ_API_KEY_1 in .env.")
    else:
        print("\nExecuting live connection test (1 single request)...")
        prompt = "Reply with exactly: GROQ_CONNECTION_SUCCESS"
        try:
            response = groq_manager.invoke(prompt)
            content = getattr(response, "content", str(response)).strip()
            print(f"Model response: {content}")
            if "GROQ_CONNECTION_SUCCESS" in content:
                print("Real Groq connection: PASS")
            else:
                print(f"Real Groq connection: FAIL (Unexpected output: {content})")
        except Exception as exc:
            print(f"Real Groq connection: FAIL ({exc})")

    print("\nRunning unit tests suite (mocked / zero quota usage)...")
    suite = unittest.defaultTestLoader.discover("tests", pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("=" * 60)
    if result.wasSuccessful():
        print("Unit test suite: ALL PASSED")
    else:
        print(f"Unit test suite: FAILED ({len(result.failures)} failures, {len(result.errors)} errors)")
    print("=" * 60)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_diagnostics())

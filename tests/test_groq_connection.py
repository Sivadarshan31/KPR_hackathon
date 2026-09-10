import sys
from pathlib import Path

# Ensure project root is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.config import get_configured_keys_count, GROQ_MODEL
from app.llm.groq_manager import groq_manager
from app.llm.exceptions import GroqConfigurationError, GroqRequestError


def main():
    keys_count = get_configured_keys_count()
    print(f"Groq keys loaded: {keys_count}")
    print(f"Groq model configured: {GROQ_MODEL}")

    if keys_count == 0:
        print(
            "\nCONFIGURATION NOTICE: 0 Groq API keys are configured in .env.\n"
            "Please configure at least GROQ_API_KEY_1 in your .env file to perform a live API request."
        )
        return 2

    print("\nInitiating live Groq connection test...")
    prompt = "Reply with exactly: GROQ_CONNECTION_SUCCESS"

    try:
        response = groq_manager.invoke(prompt)
        content = getattr(response, "content", str(response)).strip()
        print(f"Model response: {content}")

        if "GROQ_CONNECTION_SUCCESS" in content:
            print("\nReal Groq connection: PASS")
            return 0
        else:
            print(
                f"\nReal Groq connection: FAIL (Response did not contain GROQ_CONNECTION_SUCCESS: {content})"
            )
            return 1
    except (GroqConfigurationError, GroqRequestError) as err:
        print(f"\nReal Groq connection: FAIL ({err})")
        return 1
    except Exception as exc:
        print(f"\nReal Groq connection: FAIL (Unexpected error: {type(exc).__name__})")
        return 1


if __name__ == "__main__":
    sys.exit(main())

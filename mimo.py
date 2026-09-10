
import requests

URL = "https://opencode.ai/inference/openai/v1/chat/completions"
MODEL = "mimo-v2.5-free"

messages = [
    {
        "role": "system",
        "content": "You are a helpful AI coding assistant."
    }
]
print("MiMo-V2.5 Free CLI")
print("Type 'exit' to quit.\n")

while True:
    try:
        user = input("You: ").strip()

        if user.lower() in ("exit", "quit"):
            print("Bye!")
            break

        if not user:
            continue

        messages.append({
            "role": "user",
            "content": user
        })

        response = requests.post(
            URL,
            json={
                "model": MODEL,
                "messages": messages,
                "stream": False
            },
            timeout=120
        )

        if response.status_code != 200:
            print(f"\nAPI error {response.status_code}:")
            print(response.text)
            messages.pop()
            continue

        data = response.json()

        answer = data["choices"][0]["message"]["content"]

        print(f"\nMiMo: {answer}\n")

        messages.append({
            "role": "assistant",
            "content": answer
        })

    except KeyboardInterrupt:
        print("\nBye!")
        break

    except requests.RequestException as e:
        print(f"\nNetwork error: {e}\n")

    except Exception as e:
        print(f"\nError: {e}\n")


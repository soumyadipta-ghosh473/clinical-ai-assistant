from groq import Groq
import os

# Test prompts (robustness testing)
tests = [
    "Patient fever",
    "HR high maybe infection ???",
    "tachycardia but stable??"
]

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

system_prompt = """
You are a clinical AI assistant.

Rules:
- Interpret incomplete or unclear medical inputs.
- Provide medically relevant reasoning.
- Do NOT hallucinate.
"""

print("\n--- Prompt Robustness Testing ---\n")

for t in tests:

    print("TEST INPUT:", t)

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": t}
            ]
        )

        print("RESPONSE:", response.choices[0].message.content)
        print("-" * 50)

    except Exception as e:
        print("ERROR:", e)
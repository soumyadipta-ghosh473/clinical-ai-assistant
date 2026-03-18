from groq import Groq
import os


def multimodal_fusion(risk, shap_features, clinical_note):

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # ---------- HARD CONTROL ----------
    if not clinical_note or clinical_note.strip() == "":
        clinical_note = "No clinical notes provided."

    # ---------- STRICT PROMPT ----------
    prompt = f"""
You are a clinical decision support system.

INPUT DATA (DO NOT MODIFY OR QUESTION):

Risk Score: {round(risk,3)}

Top Features:
{shap_features}

Clinical Notes:
{clinical_note}

STRICT RULES:
- DO NOT ask for more data
- DO NOT say "please provide notes"
- DO NOT assume missing information
- DO NOT use placeholders like [Insert ...]
- ONLY use the data given above
- If notes are "No clinical notes provided", clearly state that

OUTPUT FORMAT:

Clinical Decision Support Output

1. Risk Interpretation:
Explain the risk score in 1-2 lines

2. Key Insights:
Explain ONLY based on given features

3. Clinical Notes Summary:
- If notes exist → summarize briefly
- If no notes → say "No clinical notes provided"

4. Final Decision:
Short realistic clinical decision

5. Next Steps:
Bullet points (3–4)

KEEP IT CONCISE AND REALISTIC.
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict medical AI that only uses provided data."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Multimodal system error: {str(e)}"
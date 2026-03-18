from groq import Groq
import os


def multimodal_fusion(risk, shap_features, clinical_note):

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # ---------- HANDLE EMPTY NOTES ----------
    if not clinical_note or clinical_note.strip() == "":
        clinical_note = "No clinical notes provided."

    # ---------- STRONG PROMPT ----------
    prompt = f"""
You are a clinical decision support system.

Patient risk score: {round(risk,3)}

Top contributing features:
{shap_features}

Clinical notes:
{clinical_note}

STRICT INSTRUCTIONS:
- Do NOT use placeholders like [Insert ...]
- If clinical notes are missing, explicitly say "No clinical notes provided"
- Only use the given data
- Do NOT assume abnormalities unless stated
- Keep response professional and realistic

Provide:
1. Risk interpretation
2. Key clinical insights
3. Final decision
4. Recommended next steps
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role":"system","content":"You are a professional ICU clinical AI assistant."},
            {"role":"user","content":prompt}
        ]
    )

    # ---------- RETURN OUTPUT ----------
    return response.choices[0].message.content
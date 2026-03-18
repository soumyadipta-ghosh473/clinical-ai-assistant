from groq import Groq
import os


def multimodal_fusion(risk, shap_features, clinical_note):

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    prompt = f"""
Structured model risk: {risk}

Top features:
{shap_features}

Clinical notes:
{clinical_note}

Combine structured data and clinical notes to give final clinical decision.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role":"system","content":"You are a clinical decision support system."},
            {"role":"user","content":prompt}
        ]
    )

    return response.choices[0].message.content
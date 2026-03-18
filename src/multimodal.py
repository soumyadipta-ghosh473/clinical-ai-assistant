from groq import Groq
import os
from src.text_processing import extract_keywords  # ✅ NEW


def multimodal_fusion(risk, shap_features, clinical_note):

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    if not clinical_note or clinical_note.strip() == "":
        clinical_note = "No clinical notes provided."

    # ✅ NEW NLP STEP
    keywords = extract_keywords(clinical_note)

    prompt = f"""
You are a clinical decision support system.

INPUT DATA (DO NOT MODIFY OR QUESTION):

Risk Score: {round(risk,3)}

Top Features:
{shap_features}

Clinical Notes:
{clinical_note}

Extracted Keywords:
{keywords}

STRICT RULES:
- DO NOT ask for more data
- DO NOT assume missing info
- DO NOT use placeholders
- ONLY use given data

OUTPUT FORMAT:

Clinical Decision Support Output

1. Risk Interpretation
2. Key Insights
3. Clinical Notes Summary
4. Final Decision
5. Next Steps
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a strict clinical AI."},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Multimodal system error: {str(e)}"
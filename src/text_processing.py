def extract_keywords(clinical_note):

    if not clinical_note or clinical_note.strip() == "":
        return "No keywords extracted"

    note = clinical_note.lower()

    keywords = []

    if "fever" in note:
        keywords.append("Fever")

    if "pain" in note:
        keywords.append("Pain")

    if "infection" in note:
        keywords.append("Infection")

    if "cough" in note:
        keywords.append("Respiratory Issue")

    if "diabetes" in note:
        keywords.append("Diabetes")

    if "hypertension" in note:
        keywords.append("Hypertension")

    return ", ".join(keywords) if keywords else "General observation"
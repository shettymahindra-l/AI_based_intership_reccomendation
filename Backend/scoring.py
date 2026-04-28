# ===============================
# WEIGHTS (DEFAULT - will tune later)
# ===============================

EDU_WEIGHT = 0.3
LOC_WEIGHT = 0.1
DOMAIN_WEIGHT = 0.2
TITLE_WEIGHT = 0.2


# ===============================
# HELPERS
# ===============================

def normalize(text):
    return text.lower().strip() if isinstance(text, str) else ""


def jaccard(a, b):
    intersection = 0

    for skill1 in a:
        for skill2 in b:
            if skill1 in skill2 or skill2 in skill1:
                intersection += 1

    union = len(a) + len(b) - intersection

    return intersection / union if union != 0 else 0


# ===============================
# MAIN SCORING FUNCTION
# ===============================
SYNONYMS = {
    "ai": "machine learning",
    "ml": "machine learning",
    "dl": "deep learning",
    "js": "javascript",
    "py": "python"
}

def normalize_skill(skill):
    skill = skill.lower().strip()
    return SYNONYMS.get(skill, skill)

def compute_scores(candidate, knn_results, alpha, beta):

    candidate_skills = set(normalize_skill(s) for s in candidate.get("skills", []))

    domain = normalize(candidate.get("domain", ""))
    education = normalize(candidate.get("education", ""))
    user_location = normalize(candidate.get("location", ""))

    results = []

    for item in knn_results:

        op = item["data"]
        knn_score = item["knn_score"]   # FROM KNN MODEL
        op_skills = set(item["skills"])

        # ===============================
        # Jaccard Similarity
        # ===============================
        j_score = jaccard(candidate_skills, op_skills)

        # ===============================
        # Hybrid Score (MAIN LOGIC)
        # ===============================
        hybrid_score = (alpha * knn_score) + (beta * j_score)

        # ===============================
        # Boosting Factors
        # ===============================

        # Title boost
        title_score = 0
        if any(skill in normalize(op.get("title", "")) for skill in candidate_skills):
            title_score = TITLE_WEIGHT

        # Domain boost
        domain_score = DOMAIN_WEIGHT if domain == normalize(op.get("domain", "")) else 0

        # Education boost (10th, 12th, UG supported)
        edu_score = 0
        qualifications = normalize(op.get("qualification", "")).split("/")

        if any(education in q for q in qualifications):
            edu_score = EDU_WEIGHT

        # Location boost
        loc_score = LOC_WEIGHT if user_location == normalize(op.get("location", "")) else 0

        # ===============================
        # Final Score
        # ===============================
        final_score = (
            hybrid_score +
            title_score +
            domain_score +
            edu_score +
            loc_score
        )

        results.append({
            "data": op,
            "score": final_score
        })

    # ===============================
    # Sort Results
    # ===============================
    results.sort(key=lambda x: x["score"], reverse=True)

    return results
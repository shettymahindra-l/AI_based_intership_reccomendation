from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ALPHA = 0.5
BETA = 0.5

EDU_WEIGHT = 15
LOC_WEIGHT = 3
DOMAIN_WEIGHT = 5
TITLE_WEIGHT = 5


def normalize(text):
    return text.lower().strip() if isinstance(text, str) else ""


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0


def compute_scores(candidate, opportunities):

    candidate_skills = set(normalize(s) for s in candidate.get("skills", []))
    candidate_text = " ".join(candidate_skills)

    domain = normalize(candidate.get("domain", ""))
    education = normalize(candidate.get("education", ""))
    user_location = normalize(candidate.get("location", ""))

    docs = [candidate_text]

    for op in opportunities:
        docs.append(" ".join(normalize(s) for s in op.get("skills", [])))

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(docs)

    tfidf_scores = cosine_similarity(matrix[0:1], matrix[1:])[0]

    results = []

    for i, op in enumerate(opportunities):

        op_skills = set(normalize(s) for s in op.get("skills", []))

        j_score = jaccard(candidate_skills, op_skills)
        hybrid_score = (ALPHA * tfidf_scores[i]) + (BETA * j_score)

        # Title boost
        title_score = 0
        if any(skill in normalize(op.get("title", "")) for skill in candidate_skills):
            title_score = TITLE_WEIGHT

        # Domain boost
        domain_score = DOMAIN_WEIGHT if domain == normalize(op.get("domain", "")) else 0

        # Education boost
        edu_score = 0
        qualifications = normalize(op.get("qualification", "")).split("/")
        if any(education in q for q in qualifications):
            edu_score = EDU_WEIGHT

        # Location boost
        loc_score = LOC_WEIGHT if user_location == normalize(op.get("location", "")) else 0

        final_score = hybrid_score + title_score + domain_score + edu_score + loc_score

        results.append({
            "data": op,
            "score": final_score
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    return results
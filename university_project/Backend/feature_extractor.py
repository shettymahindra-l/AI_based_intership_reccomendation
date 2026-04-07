import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ===============================
# Load Internship Data
# ===============================

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "opportunity.json")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# ===============================
# Build Corpus (for TF-IDF)
# ===============================

def build_corpus():
    corpus = []

    for op in data:
        skills = op.get("skills", [])

        # handle both list and string
        if isinstance(skills, list):
            skills_text = " ".join(skills)
        else:
            skills_text = skills

        text = (
            skills_text + " " +
            str(op.get("domain", "")) + " " +
            str(op.get("description", ""))
        )

        corpus.append(text.lower())

    return corpus


# ===============================
# TF-IDF Vectorization
# ===============================

vectorizer = TfidfVectorizer(stop_words="english")
corpus = build_corpus()
X = vectorizer.fit_transform(corpus)

# ===============================
# KNN Model (Cosine Distance)
# ===============================

knn = NearestNeighbors(metric='cosine')
knn.fit(X)


# ===============================
# Candidate Vector
# ===============================

def get_candidate_vector(candidate):
    skills = candidate.get("skills", [])
    domain = candidate.get("domain", "")

    text = " ".join(skills) + " " + domain

    return vectorizer.transform([text.lower()])


# ===============================
# KNN Results
# ===============================

def get_knn_results(candidate_vector, topk=5):

    distances, indices = knn.kneighbors(candidate_vector, n_neighbors=topk)

    results = []

    for i in indices[0]:
        op = data[i]

        skills_data = op.get("skills", [])

        # normalize skills properly
        if isinstance(skills_data, list):
            skills_list = [s.strip().lower() for s in skills_data]
        else:
            skills_list = [s.strip().lower() for s in skills_data.split(",")]

        results.append({
            "title": op.get("title"),
            "company": op.get("company"),
            "location": op.get("location"),
            "qualification": op.get("qualification"),
            "skills": skills_list,
            "description": op.get("description"),
            "domain": op.get("domain")
        })

    return results
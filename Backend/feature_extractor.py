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
# Build Corpus
# ===============================

def build_corpus():
    corpus = []

    for op in data:
        skills = op.get("skills", [])

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
# TF-IDF Vectorization (FIXED)
# ===============================

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=2000
)

corpus = build_corpus()
X = vectorizer.fit_transform(corpus)

# ===============================
# KNN Model
# ===============================

knn = NearestNeighbors(metric='cosine')
knn.fit(X)

# ===============================
# Candidate Vector
# ===============================

def get_candidate_vector(candidate):
    skills = candidate.get("skills", [])
    domain = candidate.get("domain", "")
    location = candidate.get("location", "")

    text = " ".join(skills) + " " + domain + " " + location

    return vectorizer.transform([text.lower()])


# ===============================
# KNN Results
# ===============================

def get_knn_results(candidate_vector, topk=5):

    distances, indices = knn.kneighbors(candidate_vector, n_neighbors=topk)

    results = []

    for idx, dist in zip(indices[0], distances[0]):

        op = data[idx]

        similarity = 1 - dist

        skills_data = op.get("skills", [])

        if isinstance(skills_data, list):
            skills_list = [s.strip().lower() for s in skills_data]
        else:
            skills_list = [s.strip().lower() for s in skills_data.split(",")]

        results.append({
            "data": op,
            "skills": skills_list,
            "knn_score": similarity
        })

    return results
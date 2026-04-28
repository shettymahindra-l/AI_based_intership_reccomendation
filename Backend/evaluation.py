import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, recall_score, f1_score


# ---------------------------
# LOAD DATA
# ---------------------------
def load_data():
    with open("opportunity.json", "r") as f:
        return json.load(f)


# ---------------------------
# FEATURE EXTRACTION
# ---------------------------
def extract_features(data):
    documents = []

    for item in data:
        text = item["title"] + " " + item["description"] + " " + " ".join(item["skills"])
        documents.append(text)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    return tfidf_matrix, vectorizer


# ---------------------------
# BUILD KNN
# ---------------------------
def build_knn(tfidf_matrix, k):
    knn = NearestNeighbors(n_neighbors=k, metric='cosine')
    knn.fit(tfidf_matrix)
    return knn


# ---------------------------
# COMPUTE SCORES
# ---------------------------
def compute_scores(query_vec, tfidf_matrix, knn, alpha, beta):
    distances, indices = knn.kneighbors(query_vec)

    results = []

    for i, idx in enumerate(indices[0]):
        similarity = 1 - distances[0][i]

        tfidf_score = (query_vec @ tfidf_matrix[idx].T).toarray()[0][0]

        final_score = alpha * similarity + beta * tfidf_score

        results.append((int(idx), float(final_score)))  # convert from np types

    results.sort(key=lambda x: x[1], reverse=True)

    return results


# ---------------------------
# AUTO GROUND TRUTH (IMPORTANT)
# ---------------------------
def generate_ground_truth(data, query):
    ground_truth = []

    query_words = query.lower().split()

    for i, item in enumerate(data):
        text = (item["title"] + " " + item["description"]).lower()

        # check if any keyword matches
        if any(word in text for word in query_words):
            ground_truth.append(i)

    return ground_truth


# ---------------------------
# EVALUATION
# ---------------------------
def evaluate(predicted_indices, ground_truth, total_items):
    y_true = []
    y_pred = []

    for i in range(total_items):
        y_true.append(1 if i in ground_truth else 0)
        y_pred.append(1 if i in predicted_indices else 0)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return precision, recall, f1


# ---------------------------
# MAIN FUNCTION
# ---------------------------
def tune_parameters():
    data = load_data()

    tfidf_matrix, vectorizer = extract_features(data)
    total_items = len(data)

    # 🔥 CHANGE QUERY IF NEEDED
    query = "machine learning python data science"
    query_vec = vectorizer.transform([query])

    # 🔥 AUTO GROUND TRUTH
    ground_truth = generate_ground_truth(data, query)

    print("\n✅ Ground Truth Indices:", ground_truth[:10], "...")  # show first few

    best_f1 = 0
    best_config = None

    for k in [3, 5, 7, 10]:
        knn = build_knn(tfidf_matrix, k)

        for alpha in np.arange(0.1, 0.9, 0.1):
            beta = 1 - alpha

            results = compute_scores(query_vec, tfidf_matrix, knn, alpha, beta)

            predicted_indices = [idx for idx, _ in results]

            precision, recall, f1 = evaluate(predicted_indices, ground_truth, total_items)

            print("\n----------------------------------")
            print(f"K={k}, Alpha={round(alpha,2)}, Beta={round(beta,2)}")
            print(f"Precision={precision:.3f}")
            print(f"Recall={recall:.3f}")
            print(f"F1 Score={f1:.3f}")

            if f1 > best_f1:
                best_f1 = f1
                best_config = (k, alpha, beta)

    print("\n====================")
    print("BEST CONFIGURATION")
    print("====================")

    if best_config is not None:
        print(f"K = {best_config[0]}")
        print(f"Alpha = {best_config[1]:.2f}")
        print(f"Beta = {best_config[2]:.2f}")
        print(f"Best F1 Score = {best_f1:.3f}")
    else:
        print("❌ No valid predictions found.")


# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    tune_parameters()
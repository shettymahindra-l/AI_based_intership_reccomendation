from pymongo import MongoClient
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
from io import BytesIO
import pdfplumber
import docx

from feature_extractor import get_candidate_vector, get_knn_results
from scoring import compute_scores

# ==============================
# App Setup
# ==============================

BASE_DIR = os.path.dirname(__file__)
PAGES_DIR = os.path.join(BASE_DIR, "../pages")

app = Flask(__name__, static_folder=PAGES_DIR)
CORS(app)

# ==============================
-=# MongoDB Setup
# ==============================

client = MongoClient("mongodb://localhost:27017/")
db = client["internship_db"]

user_collection = db["users"]
data_collection = db["user_activity"]
applied_collection = db["applied_internships"


]
# ==============================
# DEFAULT WEIGHTS
# ==============================

ALPHA = 0.5
BETA = 0.5

# ==============================
# Recommendation Logic (FIXED)
# ==============================

def recommend(candidate):

    print("Candidate Input:", candidate)

    candidate_vector = get_candidate_vector(candidate)

    # Step 1: KNN
    knn_results = get_knn_results(candidate_vector, 50)  # or 30

    print("KNN Results:", len(knn_results))

    # Step 2: Hybrid scoring
    scored = compute_scores(candidate, knn_results, ALPHA, BETA)

    if not scored:
        return []

    max_score = scored[0]["score"] if scored else 1

    results = []

    for r in scored:
        op = r["data"]

        normalized_score = (r["score"] / max_score) * 100 if max_score != 0 else 0

        results.append({
            "company": op.get("company"),
            "title": op.get("title"),
            "location": op.get("location"),
            "domain": op.get("domain"),
            "qualification": op.get("qualification"),
            "skills": op.get("skills"),
            "description": op.get("description"),
            "score": round(normalized_score, 2)
        })

    print("Final Results:", len(results))

    return results[:candidate.get("topk", 5)]

# ==============================
# Resume Parsing
# ==============================

COMMON_SKILLS = [
    "python", "java", "sql", "machine learning",
    "django", "flask", "react", "html", "css",
    "excel", "power bi", "data analysis",
    "spring boot", "docker", "microservices", "javascript"
]

def extract_text_from_file(file):

    filename = file.filename.lower()

    try:
        file_stream = BytesIO(file.read())

        if filename.endswith(".pdf"):
            text = ""
            with pdfplumber.open(file_stream) as pdf:
                for page in pdf.pages:
                    content = page.extract_text()
                    if content:
                        text += content + " "
            return text.strip()

        elif filename.endswith(".docx"):
            doc = docx.Document(file_stream)
            return " ".join([para.text for para in doc.paragraphs])

    except Exception as e:
        print("Error reading file:", e)

    return ""

def extract_skills(text):
    text = text.lower()
    return list(set([skill for skill in COMMON_SKILLS if skill in text]))

def extract_education(text):

    text = text.lower()

    if "10th" in text or "sslc" in text:
        return "10th"
    elif "12th" in text or "puc" in text:
        return "12th"
    elif "b.tech" in text:
        return "B.Tech"
    elif "bca" in text:
        return "BCA"
    elif "b.sc" in text:
        return "B.Sc"
    elif "bcom" in text:
        return "B.Com"

    return ""

# ==============================
# Routes
# ==============================

@app.route("/")
def home():
    return send_from_directory(PAGES_DIR, "index.html")

@app.route("/<path:path>")
def serve_files(path):
    return send_from_directory(PAGES_DIR, path)

# ==============================
# REGISTER
# ==============================

@app.route("/register", methods=["POST"])
def register():

    data = request.json

    email = data.get("email").lower()
    password = data.get("password")

    existing = user_collection.find_one({"email": email})

    if existing:
        return jsonify({"success": False, "message": "User already exists"})

    user_collection.insert_one({
        "name": data.get("name"),
        "email": email,
        "password": password
    })

    return jsonify({"success": True, "message": "User registered successfully"})

# ==============================
# LOGIN
# ==============================

@app.route("/login", methods=["POST"])
def login():

    data = request.json

    email = data.get("email", "").lower()
    password = data.get("password", "")

    user = user_collection.find_one({"email": email})

    if not user:
        return jsonify({"success": False, "message": "User not found"})

    if user["password"] != password:
        return jsonify({"success": False, "message": "Incorrect password"})

    return jsonify({
    "success": True,
    "message": "Login successful",
    "name": user["name"],
    "email": user["email"]
})

# ==============================
# MANUAL INPUT (FIXED)
# ==============================

@app.route("/recommend", methods=["POST"])
def get_recommendations():

    data = request.json

    print("Incoming Request:", data)

    if not data:
        return jsonify({"error": "No input provided"}), 400

    candidate = {
        "skills": data.get("skills", []),
        "education": data.get("education", ""),
        "domain": data.get("domain", ""),
        "location": data.get("location", ""),
        "topk": data.get("topk", 5)
    }

    data_collection.insert_one({
        "type": "manual",
        **candidate
    })

    results = recommend(candidate)

    print("Response Sent:", results)

    return jsonify(results)

# ==============================
# RESUME UPLOAD
# ==============================

@app.route("/upload-resume", methods=["POST"])
def upload_resume():

    if "resume" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["resume"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    text = extract_text_from_file(file)

    if not text.strip():
        return jsonify({"error": "Could not extract text"}), 400

    skills = extract_skills(text)
    education = extract_education(text)

    candidate = {
        "skills": skills,
        "education": education,
        "domain": "IT & Software",
        "location": "",
        "topk": 5
    }

    results = recommend(candidate)

    data_collection.insert_one({
        "type": "resume",
        "skills": skills,
        "education": education
    })

    return jsonify({
        "results": results,
        "skills": skills,
        "education": education
    })

# ==============================
# GET SKILLS
# ==============================

@app.route("/skills", methods=["GET"])
def get_all_skills():

    with open("opportunity.json") as f:
        data = json.load(f)

    skills_set = set()

    for item in data:
        for skill in item.get("skills", []):
            skills_set.add(skill)

    return jsonify(sorted(list(skills_set)))

# ==============================
# APPLY INTERNSHIP
# ==============================

@app.route("/apply", methods=["POST"])
def apply_internship():

    data = request.json

    applied_collection.insert_one(data)

    return jsonify({"message": "Applied successfully"})


# ==============================
# GET APPLIED INTERNSHIPS
# ==============================

@app.route("/applied", methods=["GET"])
def get_applied():

    data = list(applied_collection.find({}, {"_id": 0}))

    return jsonify(data)
# ==============================
# RUN
# ==============================

if __name__ == "__main__":
    app.run(debug=True)
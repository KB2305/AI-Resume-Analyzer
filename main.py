import streamlit as st
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import spacy
from transformers import pipeline
import os
from rapidfuzz import fuzz
import matplotlib.pyplot as plt

# ------------------ Load Models ------------------
@st.cache_resource
def load_models():
    similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
    nlp = spacy.load("en_core_web_sm")
    summarizer = pipeline("summarization", model="t5-small")
    return similarity_model, nlp, summarizer

similarity_model, nlp, summarizer = load_models()

# ------------------ Load Skills List ------------------
def load_skills():
    skills_file = "skills_list.txt"
    if not os.path.exists(skills_file):
        return []
    with open(skills_file, "r", encoding="utf-8") as f:
        skills = [line.strip().lower() for line in f if line.strip()]
    return skills

SKILLS_DB = load_skills()

# ------------------ Utility Functions ------------------

# Extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

# AI semantic similarity
def ai_match_score(resume_text, jd_text):
    resume_embedding = similarity_model.encode(resume_text, convert_to_tensor=True)
    jd_embedding = similarity_model.encode(jd_text, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(resume_embedding, jd_embedding).item()
    return similarity * 100

# Improved skills extraction
def extract_skills(text):
    text_lower = text.lower()
    found = set()

    # 1. Dictionary lookup
    for skill in SKILLS_DB:
        if skill in text_lower:
            found.add(skill)

    # 2. Fuzzy matching
    for skill in SKILLS_DB:
        if fuzz.partial_ratio(skill, text_lower) > 90:
            found.add(skill)

    # 3. NER entities
    doc = nlp(text)
    ner_skills = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT"]]
    found.update([s.lower() for s in ner_skills])

    return sorted(found)

# Summarize resume
def summarize_resume(text):
    if len(text.split()) < 50:
        return "Resume too short for summarization."
    summary = summarizer(text[:1000], max_length=60, min_length=25, do_sample=False)
    return summary[0]['summary_text']

# ATS check
def ats_check(text):
    issues = []
    if "table" in text.lower():
        issues.append("❌ Contains tables (ATS may not read them).")
    if len(text.split()) < 150:
        issues.append("⚠️ Resume seems too short (<150 words).")
    if not any(word in text.lower() for word in ["experience", "education", "skills"]):
        issues.append("❌ Missing common sections (Experience/Education/Skills).")
    if not any(char.isdigit() for char in text):
        issues.append("⚠️ No dates found — ATS may think experience is missing.")
    return issues if issues else ["✅ ATS-friendly formatting detected."]

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="AI Resume Analyzer", page_icon="🤖", layout="wide")
st.title("🤖 AI Resume Analyzer")
st.write("Upload your resume and paste a job description to check alignment, skills, ATS-friendliness, and get a summary.")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
jd_text = st.text_area("Paste Job Description Here")

if uploaded_file and jd_text.strip():
    st.subheader("🔍 Processing Resume...")
    resume_text = extract_text_from_pdf(uploaded_file)

    # Match score
    score = ai_match_score(resume_text, jd_text)
    st.success(f"✅ AI Match Score: **{score:.2f}%**")

    if score < 40:
        st.error("⚠️ Weak Match: Resume does not align well with the job description. Add missing skills & keywords.")
    elif score < 70:
        st.warning("⚠️ Medium Match: Resume covers some skills, but could be improved.")
    else:
        st.success("🎉 Strong Match: Resume aligns well with the job description!")

    # Skills extraction
    st.subheader("🛠️ Extracted Skills")
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)

    matched_skills = set(resume_skills).intersection(set(jd_skills))
    missing_skills = set(jd_skills) - set(resume_skills)

    st.write("**Skills in Resume:**", resume_skills if resume_skills else "No specific skills detected.")
    st.write("**Matched Skills (Resume & JD):**", matched_skills if matched_skills else "No matches found.")
    st.write("**Missing Skills (from JD):**", missing_skills if missing_skills else "None! ✅")

    # Visualization
    st.subheader("📊 Skills Match Visualization")
    labels = ["Matched", "Missing"]
    values = [len(matched_skills), len(missing_skills)]
    fig, ax = plt.subplots()
    ax.bar(labels, values, color=["green", "red"])
    ax.set_ylabel("Number of Skills")
    ax.set_title("Resume vs Job Description Skills")
    st.pyplot(fig)

    # Summarization
    st.subheader("📝 Resume Summary")
    st.write(summarize_resume(resume_text))

    # ATS Check
    st.subheader("📋 ATS Compatibility Check")
    ats_issues = ats_check(resume_text)
    for issue in ats_issues:
        st.write(issue)

    # Raw text (optional)
    with st.expander("📄 Full Extracted Resume Text"):
        st.write(resume_text)

# AI-Resume-Analyzer
I-powered Resume Analyzer that evaluates resumes against job descriptions, extracts and compares skills, checks ATS compatibility, and generates a concise summary. Built with Streamlit, NLP, and Machine Learning.

# 🤖 AI Resume Analyzer  

An **AI-powered Resume Analyzer** built with **Streamlit** that evaluates resumes against job descriptions using **NLP & Machine Learning**.  
It extracts key skills, generates an AI match score, checks ATS compatibility, and provides a summary of the resume.  

---

## 🚀 Features
- 📄 **Resume Text Extraction** from PDF  
- 🧠 **AI Match Score** using Sentence-BERT semantic similarity  
- 🛠 **Skill Extraction & Comparison** (Resume vs Job Description)  
- 📊 **Skill Match Visualization** (Matched vs Missing Skills)  
- ✅ **ATS Compatibility Check** (formatting, sections, dates)  
- 📝 **Resume Summarization** with HuggingFace transformers  

---

## 🛠 Tech Stack
- [Python 3.8+](https://www.python.org/)  
- [Streamlit](https://streamlit.io/) – UI framework  
- [Sentence-Transformers](https://www.sbert.net/) – Semantic similarity  
- [spaCy](https://spacy.io/) – Named Entity Recognition  
- [HuggingFace Transformers](https://huggingface.co/) – Summarization  
- [RapidFuzz](https://github.com/maxbachmann/RapidFuzz) – Fuzzy skill matching  
- [Matplotlib](https://matplotlib.org/) – Visualization  

---

## 📂 Project Structure
```
AI-Resume-Analyzer/
│── main.py # Streamlit app
│── skills_list.txt # Skills dictionary
│── requirements.txt # Dependencies
│── README.md # Project documentation
```


---

## ⚙️ Installation

1. Clone the repository:
   ```
   bash
   git clone https://github.com/your-username/ai-resume-analyzer.git
   cd ai-resume-analyzer
   ```
2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Download spaCy model:
   ```
   python -m spacy download en_core_web_sm
   ```

## ▶️ Usage
Run the App with:
```
streamlit run main.py
```
Then open http://localhost:8501 in your browser.

## 📌 Example Workflow

1. Upload your resume (PDF)
2. Paste the job description
3. Get:
   - AI match score
  - Extracted & compared skills
  - ATS compatibility check
  - Auto-generated resume summary
  - Skills match visualization

## 📜 License

This project is open-source under the MIT License.

## ✨ Future Improvements
- Multi-resume upload and ranking
- More advanced ATS compliance checks
- Integration with job portals

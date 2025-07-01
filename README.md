# 📄 AI PDF Interview Q&A Generator

This project uses a Hugging Face model (like `flan-t5-base`) with LangChain and Streamlit to automatically generate interview-style questions and detailed answers from a PDF document.

## 🚀 Features

- Upload any PDF (e.g., research paper, article, etc.)
- Automatically extract text
- Generate 8+ insightful interview questions
- Get detailed technical answers for each question
- Runs locally via Streamlit or can be deployed on Vercel

---

## 🧰 Requirements

Make sure you have **Python 3.9+** installed.

### 🔧 Installation Steps

1. **Clone the repository**
bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
Create a virtual environment (optional but recommended)

bash
Copy
Edit
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
📦 Required Python Packages
Here’s what’s in requirements.txt:

txt
Copy
Edit
transformers
torch
langchain
langchain-community
langchain-huggingface
streamlit
pypdf
You can install them manually as well:

bash
Copy
Edit
pip install transformers torch langchain langchain-community langchain-huggingface streamlit pypdf
🧪 Run Locally
After installing everything:

bash
Copy
Edit
streamlit run app.py
Then go to the URL shown in terminal (usually http://localhost:8501)

🖥️ Deployment (Optional – Vercel)
Make sure your app.py or main.py is at the root.

Add a vercel.json (optional):

json
Copy
Edit
{
  "builds": [{ "src": "app.py", "use": "@vercel/python" }]
}
Push to GitHub and connect the repo on vercel.com.

In the "Framework Preset", choose: Other

Set:

Build Command: pip install -r requirements.txt

Output Directory: .

📂 Folder Structure
css
Copy
Edit
.
├── app.py (or main.py)
├── data/
│   └── your PDF files
├── requirements.txt
└── README.md
📩 Example PDF to Test
You can use any technical research paper, CV, whitepaper, etc.

📃 License
MIT License

🤝 Contributions
Pull requests and stars are welcome!

✨ Author
Developed by Sumit

yaml
Copy
Edit

requiment text
transformers>=4.39.0
torch>=2.1.0
streamlit>=1.30.0
langchain>=0.1.13
langchain-community>=0.0.25
langchain-huggingface>=0.0.1
pypdf>=3.17.0

---

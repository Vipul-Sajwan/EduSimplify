# 📚 EduSimplify

**AI-Powered Course Content Simplification Agent**  
*Problem Statement #24 — Edunet Foundation × IBM*

EduSimplify helps students — especially first-generation learners and those from non-English-medium backgrounds — understand complex academic content by breaking it down into simple language, extracting key takeaways, and generating self-test quizzes.

---


## Features

| Feature | Description |
|---|---|
| **Content Simplification** | Streams simplified explanations via IBM Granite in real time |
| **Difficulty Levels** | Beginner / Intermediate / Advanced |
| **Language Support** | English and Hindi output |
| **PDF Upload** | Extracts text from text-based PDFs (up to 200 pages) |
| **Text Input** | Paste any academic content directly |
| **Key Takeaways** | Extracts 5 must-remember bullet points |
| **Quiz Generation** | Generates 5 MCQs with instant answer feedback and score |

---

## Tech Stack

- **LLM**: IBM Granite via [IBM watsonx.ai](https://www.ibm.com/products/watsonx-ai)
- **Frontend**: [Streamlit](https://streamlit.io/)
- **PDF Parsing**: [PyMuPDF](https://pymupdf.readthedocs.io/)
- **Language**: Python 3.10+

---

## Project Structure

```
EduSimplify/
├── app.py                  # Streamlit UI
├── modules/
│   ├── simplifier.py       # IBM Granite simplification (streaming, chunking, cache)
│   ├── takeaways.py        # Key takeaways extraction
│   ├── quiz.py             # MCQ quiz generation
│   └── pdf_reader.py       # PDF text extraction (PyMuPDF)
├── requirements.txt
├── .env                    # API credentials (do NOT commit)
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd EduSimplify
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure IBM watsonx credentials

Create a `.env` file in the project root (copy the template below):

```env
WATSONX_API_KEY=your_ibm_cloud_api_key
WATSONX_PROJECT_ID=your_watsonx_project_id
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_MODEL_ID=ibm/granite-13b-instruct-v2
```

| Variable | Where to find it |
|---|---|
| `WATSONX_API_KEY` | [IBM Cloud → Manage → Access (IAM) → API keys](https://cloud.ibm.com/iam/apikeys) |
| `WATSONX_PROJECT_ID` | [watsonx.ai → Your project → Manage → General → Project ID](https://dataplatform.cloud.ibm.com/) |
| `WATSONX_URL` | Regional endpoint — `us-south.ml.cloud.ibm.com` for Dallas (default) |
| `WATSONX_MODEL_ID` | Any Granite model available on your IBM Cloud Lite plan, e.g. `ibm/granite-13b-instruct-v2` |

> ⚠️ Never commit your `.env` file. It is already listed in `.gitignore`.

### 5. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` in your browser.

---

## Usage

1. Upload a PDF **or** paste academic text into the text box.
2. Choose a **Difficulty Level** and **Language**.
3. Click **Simplify Content**.
4. Read the streamed simplified explanation.
5. Review the **Key Takeaways** bullets.
6. Attempt the **5-question MCQ quiz** and click **Submit Answers** to see your score.

---

## IBM Cloud Lite — Free Tier Notes

- IBM Cloud Lite accounts get free access to watsonx.ai Foundation Models.
- Granite models (e.g. `granite-13b-instruct-v2`) are available on the Lite plan.
- Rate limits apply; the app retries automatically with exponential backoff.
- Input is capped at **60,000 characters** to stay within token budgets.

---

## License

MIT

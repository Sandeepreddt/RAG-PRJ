# Real-Time Research Paper Q&A RAG Pipeline

A lightweight, end-to-end Retrieval-Augmented Generation (RAG) and machine learning project implemented in a Google Colab notebook (`GEN_AI_068.ipynb`). This pipeline reads research documents (such as academic PDFs on AI and machine learning), processes and chunks the text, creates local dense vector embeddings, performs vector similarity searches, and leverages Google's Gemini models to generate precise, grounded answers with strict source citations.

## 🚀 Key Features

* **PDF Parsing & Cleaning:** Uses `pypdf` to extract text page-by-page from research papers while cleaning whitespace and formatting artifacts.
* **Configurable Chunking:** Splits extracted content into manageable text chunks with customized word sizes and overlap parameters to preserve context[cite: 1].
* **Local Semantic Embeddings:** Utilizes Hugging Face's lightweight `all-MiniLM-L6-v2` SentenceTransformer model to compute dense vector embeddings completely locally[cite: 1].
* **Cosine Similarity Search:** Normalizes embeddings to compute vector dot products for rapid, efficient similarity indexing and top-$K$ passage retrieval[cite: 1].
* **Grounded LLM Generation (Google GenAI):** Integrates the `google-genai` SDK (`gemini-2.5-flash` / `gemini-3.6-flash`) with custom system instructions to ensure answers are strictly derived from the context without hallucinations[cite: 1].
* **Citation Tracking:** Automatically maps responses back to specific source files, paper titles, and page numbers[cite: 1].

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3[cite: 1]
* **Core Libraries:**
  * `google-genai` (Google Gemini API integration)[cite: 1]
  * `sentence-transformers` (Local semantic text embedding)[cite: 1]
  * `pypdf` (PDF extraction)[cite: 1]
  * `numpy` (Vector calculations & similarity matching)[cite: 1]
  * `scikit-learn`[cite: 1]

## 📋 Getting Started (Google Colab)

1. Open the [`GEN_AI_068.ipynb`](https://github.com/Sandeepreddt/RAG-PRJ/blob/main/GEN_AI_068.ipynb) notebook in **Google Colab**[cite: 1].
2. Run the environment setup cell to install required dependencies[cite: 1]:
   ```bash
   !pip -q install pypdf sentence-transformers openai scikit-learn google-genai

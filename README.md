
````markdown
# 🌺 VedaAI: Ancient Wisdom Meets Modern Intelligence

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)



## 🧭 Project Overview

VedaAI is a specialized conversational AI tool designed for research and interpretation of ancient Indian manuscripts — including the Vedas, Puranas, Ramayana, Mahabharata, and other sacred texts. We trained this model specifically on 28+ manuscipts and indian literatures.  

Unlike general-purpose chatbots like GPT models (e.g., ChatGPT), **VedaAI** emphasizes factual accuracy, authentic citations, and domain-specific expertise. It is built for academic, linguistic, and spiritual research, bridging ancient wisdom with modern intelligence.



## ⚖️ Key Differences from General-Purpose GPT Models

### 🕉️ 1. Specialized Domain Focus
- VedaAI: Trained and optimized for ancient Indian scriptures using Retrieval-Augmented Generation (RAG) to fetch authentic verses from indexed texts such as Rigveda, Bhagavad Gita, and Puranas.  
- General GPTs: Broad internet-trained models that often lack depth and factual precision in niche areas like scriptural studies.

### 📚 2. Fact-Based & Research-Oriented
- VedaAI: Delivers evidence-backed answers with citations, combining offline scripture data and real-time web search (via Bing) for scholarly accuracy.  
- General GPTs: Often prioritize creative or conversational engagement over factual reliability.

### 🪶 3. Manuscript-Centric Functionality
- VedaAI: Built for manuscript analysis, featuring semantic search using Sentence Transformers and FAISS for context-aware retrieval.  
- General GPTs: Treat all inputs generically without referencing curated historical corpora.

### 🔰 4. Hybrid AI Architecture
- VedaAI: Uses a hybrid approach, merging Gemini 2.0 Flash (primary), DialoGPT (fallback), and web augmentation for fact-checked responses.  
- General GPTs: Depend solely on pre-trained data, with no integrated domain retrieval.



## 🌟 Core Features

- Scripture-Focused Retrieval: Contextual RAG pipeline fetching relevant verses.  
- Fact-Based Responses: Cited, verifiable, and Sanskrit-inclusive answers.  
- Hybrid AI Integration: Combines local corpus retrieval with Gemini AI and DialoGPT.  
- Interactive Web Interface: Clean, responsive UI inspired by ancient Indian aesthetics.  
- Semantic Search Engine: Uses Sentence Transformers + FAISS for efficient retrieval.  
- Research-Ready Tool: Ideal for scholars, linguists, and spiritual researchers.



## 🧩 System Requirements

- Python: 3.8+  
- Google Gemini API Key (for AI model access)  
- Internet Connection (for web augmentation and live search)

---

## ⚙️ **Installation Guide**

### 1. Clone the Repository
```bash
cd vedaai
````

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
```

Then add your Gemini API key:

```
GEMINI_API_KEY=your_api_key_here
```

### 5. Index the Corpus

* Place your scripture files (e.g., `Rigveda.txt`, `Mahabharata.txt`) in the `corpus/` directory.
* Run:

```bash
python scripture_retriever.py
```

---

## 🚀 **Usage**

### Start the Server

```bash
python main.py
```

* Access the app in your browser: [http://localhost:8000](http://localhost:8000)
* API endpoint available at `/api/chat` for programmatic access.

### Example Queries

* “What does the Bhagavad Gita say about karma?”
* “Explain the concept of dharma in the Vedas.”
* “Summarize the story of Rama and Sita in the Ramayana.”

---

## 🏗️ **Project Structure**

```
vedaai/
├── main.py                 # FastAPI app with embedded chat logic
├── app.py                  # Entry point for running the server
├── scripture_retriever.py  # FAISS-based retriever for sacred texts
├── build_corpus.py         # (Deprecated) ChromaDB corpus builder
├── retriever_old.py        # (Deprecated) Legacy retriever
├── corpus/                 # Directory for scripture text files
│   ├── RigVeda_text.txt
│   └── ...
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── faiss_index.idx         # Generated FAISS index
├── data.pkl                # Indexed data file
└── TODO.md                 # Developer notes
```

---

## 🤝 **Contributing**

Contributions are welcome!
To contribute:

1. Fork the repository
2. Create a feature branch
3. Commit and push your changes
4. Open a Pull Request

---

## 📄 **License**

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

* Inspired by the **timeless knowledge of ancient Indian scriptures**.
* Built with open-source tools: **FastAPI**, **Sentence Transformers**, **FAISS**, and **Google Gemini**.
* Sanskrit Quote:

  > “ॐ तत् सत्” *(Om Tat Sat)* — *That which is eternal truth.*

---

### 🕉️ **Connect • Learn • Explore**

> “Bridging ancient wisdom with modern AI — VedaAI empowers knowledge seekers to explore the depth of India’s sacred heritage.”

```

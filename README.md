# Multimodal Q&A AI Agent

> **“How can we make research papers talk back?”**

This multimodal AI agent ingests academic publications, interprets content—including text, figures, tables, and equations—and answers complex queries using LLM-based pipelines. It also integrates with ArXiv via Hugging Face Space and Hugging Face-hosted Spaces code to demonstrate retrieval-augmented generation and real-time academic search.

---

## 🎯 Motivation

Reading and understanding research papers is a time-consuming process, especially when searching for specific information (e.g., evaluation metrics, methods, or conclusions). Researchers, students, and data scientists often face the same challenges:

- _"Where in the paper is the accuracy reported?"_
- _"What technique was used in Section 3?"_
- _"Can I get a quick summary of this 10-page document?"_

The goal of this project is to **minimize the cognitive overhead of reading research papers** by enabling **natural language interaction** with them—leveraging large language models and structured parsing techniques.

---

## 💡 Thought Process & Design Philosophy

1. **Document Parsing ≠ Just Text Extraction**

   Academic PDFs are complex: they contain nested sections, citations, figures, tables, and math. Traditional OCR or text extractors often lose this structure. So, we began by:
   - Preserving layout using PDF layout parsers.
   - Using LLMs to understand section-level semantics.

2. **Modularity as a Design Principle**

   The pipeline is broken into modular notebooks:
   - Parsing: Layout-aware and LLM-enhanced extraction
   - Visuals: Tables, figures, and equations
   - QA Interface: Natural language-driven queries across docs
   - ArXiv RAG Agent: Document retrieval and LLM reasoning

   This makes it easier to test, extend, or integrate each component individually.

3. **LLMs as Multimodal Reasoners**

   Large Language Models, especially those with function calling or tool use, can handle complex reasoning:
   - _“Summarize Section 3”_
   - _“What is the dropout rate used?”_
   - _“Retrieve a recent paper on Vision Transformers”_

4. **Bonus Challenge: Retrieval-Augmented Generation with ArXiv**

   We explored augmenting the agent’s knowledge by making it call the **ArXiv API** when users describe a paper vaguely:
   - _"Find the paper on SAM by Meta AI"_

5. **Live Demo Innovation:**
   Deployed a Hugging Face Space to demonstrate real-time information retrieval and agentic interaction

---

## 🚀 Features

- 📄 **Structured PDF Ingestion**
  - Hierarchical extraction preserving sections, captions, and equations
- 🔄 **LLM-Powered Semantic Parsing**
-  Using models to understand academic language and context
- 🧠 **Multimodal Extraction**
  - Tables, figures, equations are preserved and referenced in the context
- 💬 **Natural Language Question Answering**
  - Summarization, direct lookups, and metric extraction
- 🔍 **ArXiv API Integration** 
  - Automatically queries papers based on user prompts
- 🤖 **Multi-Document Reasoning**
  - Supports Q&A across multiple PDFs in a single session

---

## 📁 Repository Contents

```plaintext

├── Layout_PDF_Reader.ipynb                  # Layout-aware text extraction
├── llama_Parse_final.ipynb                  # Semantic parsing using LLaMA-based models
├── multi_document_agents_final.ipynb        # Multi-document querying with LLM agent
├── llamaindex_arxiv_agentic_rag_final.ipynb # ArXiv-aware agent using RAG (bonus)
├── Images_and_tables_extraction_final.ipynb # Extract tables, charts, figures
├── requirements.txt                         # Python dependencies
└── README.md                                # This file
```
```plaintext
├── https://huggingface.co/spaces/darth15vader/Arxiv-CS-RAG         # Live demo
├── https://huggingface.co/spaces/darth15vader/Arxiv-CS-RAG/tree/main # Source code (app.py, logic, models, embeddings)
```
---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/dreamboat26/Stochastic-Assessment.git
cd Stochastic-Assessment
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```
### 3. Install Dependencies 
```bash
pip install -r requirements.txt
```
### 4. Configure API Keys
```bash
export OPENAI_API_KEY="your-openai-key"
export HUGGINGFACEHUB_API_TOKEN="your-huggingface-token"
export MISTRAL_API_KEY="your-mistral-key"
export LLAMA_CLOUD_API_KEY="your-llama-key"
```

---

## 🧪 Usage Instructions
Description about the Colab notebooks and Huggingface spaces implementation :

### A. PDF Ingestion
- **Layout_PDF_Reader.ipynb**: layout-aware parsing
- **llama_Parse_final.ipynb**: semantic parsing via LLM

### B. Tables & Figures Extraction (OCR)
- **Images_and_tables_extraction_final.ipynb**: extract visual content (e.g., matplotlib-based extraction, figure segmentation)

### C. Multi-Document Q&A
- **multi_document_agents_final.ipynb**: upload and query multiple papers with LangChain agents

### D. ArXiv RAG Agent & Hugging Face Integration
- **llamaindex_arxiv_agentic_rag_final.ipynb**: Index ArXiv papers using llama-index + RAG
- **Huggingface Spaces (app.py)**: Implements a full-fledged LLM application using ColBERTv2 + Mistral 7B / Gemma via Hugging Face pipelines

---

## 🛠️ Hugging Face Space: Arxiv-CS-RAG Demo

### What it does

- Accepts user query about a topic or paper
- Uses ColBERTv2 embeddings to search ~200k CS ArXiv abstracts
- Retrieves relevant abstracts
- Feeds those abstracts into a lightweight LLM like Mistral 7B Instruct or Gemma 7B via HF inference API
- Returns a grounded answer with citations

🔗 [Live: Arxiv‑CS‑RAG on Hugging Face Spaces](https://huggingface.co/spaces/darth15vader/Arxiv-CS-RAG)  
📂 [Source: View Space Code](https://huggingface.co/spaces/darth15vader/Arxiv-CS-RAG/tree/main)

---

## 🎥 Demo Video

![Screen-Recording(1)](https://github.com/user-attachments/assets/7494145c-f374-4ab3-86cf-52f42dac4931)

---

## 🧾 Requirements
- Python 3.8+
- Google Colab / Jupyter
### Core libraries:
- llama-index, langchain, pymupdf, pdf2image
- openai, huggingface_hub, arxiv
- colbert, transformers, gradio, pandas, matplotlib, Pillow

---

## 📄 License

Open-source under the MIT License

---

## 🏁 Future Work
- Topic-wise summarization and citation graph visualization
- Integration to TensorFlow Lite
---

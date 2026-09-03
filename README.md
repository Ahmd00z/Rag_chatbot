# 📚 Multi-PDF RAG System with Retriever Evaluation

A practical **Retrieval-Augmented Generation (RAG)** project that builds a question-answering system over multiple PDF documents using **LangChain, Cohere Embeddings, Cohere LLMs, and ChromaDB**.

The project goes beyond building a basic RAG pipeline by evaluating the **retrieval component independently** using Information Retrieval metrics such as **Precision@K, Recall@K, Average Precision (AP), and Mean Average Precision (MAP)**.

---

## 🚀 Project Overview

Traditional LLMs can generate fluent answers but may not have access to private or domain-specific documents.

This project implements a RAG pipeline that allows an LLM to answer questions based **only on information retrieved from a collection of PDF documents**.

The pipeline follows:

```text
PDF Documents
      ↓
PDF Loading
      ↓
Document Chunking
      ↓
Text Embeddings
      ↓
Chroma Vector Database
      ↓
Similarity Retrieval
      ↓
Relevant Context
      ↓
Cohere LLM
      ↓
Grounded Answer
```

The system is also designed to evaluate whether the retriever is actually retrieving the correct information.

---

## 🎯 Objectives

The main objectives of this project are:

* Build a complete RAG pipeline for PDF documents.
* Experiment with different chunk sizes and overlaps.
* Generate vector embeddings using Cohere.
* Store and retrieve document chunks using ChromaDB.
* Ask questions across multiple PDF documents.
* Support document-specific retrieval using metadata filtering.
* Prevent the LLM from answering outside the retrieved context.
* Evaluate the retriever using standard Information Retrieval metrics.
* Analyze the effect of retrieval configuration such as `k`, `chunk_size`, and `chunk_overlap`.

---

## 🛠️ Tech Stack

| Technology                         | Purpose                    |
| ---------------------------------- | -------------------------- |
| **Python**                         | Core programming language  |
| **LangChain**                      | RAG pipeline orchestration |
| **Cohere Embed v4.0**              | Text embeddings            |
| **Cohere Command A**               | Large Language Model       |
| **ChromaDB**                       | Vector database            |
| **PyPDF**                          | PDF document loading       |
| **RecursiveCharacterTextSplitter** | Document chunking          |
| **Google Colab**                   | Development environment    |

The notebook uses `CohereEmbeddings` with `embed-v4.0` and `ChatCohere` with `command-a-03-2025`.

---

# 🧩 Project Components

## 1. Basic RAG Pipeline

The first stage implements a reusable RAG function that receives documents, chunk size, and chunk overlap.

```python
AI(documents, chunk_size, chunk_overlap)
```

Inside the pipeline:

1. Documents are split using `RecursiveCharacterTextSplitter`.
2. Chunks are embedded using Cohere.
3. Embeddings are stored in ChromaDB.
4. A retriever returns the top `k=5` relevant chunks.
5. Retrieved chunks are formatted into context.
6. Cohere generates the final answer.

---

## 2. Grounded Generation

The prompt explicitly instructs the LLM to answer using **only the retrieved context**.

If the required information cannot be found, the model is instructed to respond:

```text
I don't know.
```

This helps reduce unsupported answers and demonstrates an important RAG principle:

> The LLM should generate answers from retrieved evidence rather than relying only on its internal knowledge.

## The notebook tests this behavior with questions that are intentionally absent from the documents.

# 📄 Single-PDF RAG

The notebook first experiments with a single PDF covering **RAG chunking techniques**.

The PDF is loaded using:

```python
PyPDFLoader
```

and contains 8 pages.

Different chunking strategies are explored, including:

* Fixed / Sliding Window Chunking
* Semantic Chunking
* Hierarchical Chunking
* LLM-Based Chunking
* Agentic Chunking

The notebook also demonstrates how chunking can affect retrieval quality. For example, fixed-size chunking may split ideas or sentences and potentially damage semantic context.

---

# 📚 Multi-PDF RAG

The project then extends the pipeline to multiple PDF documents.

The documents are loaded and combined into a single collection before being chunked and embedded.

```text
PDF 1
PDF 2
PDF 3
  ↓
Document Collection
  ↓
Chunking
  ↓
Embeddings
  ↓
ChromaDB
```

The notebook loads a collection of PDF pages and applies:

```python
chunk_size = 400
chunk_overlap = 30
```

before storing the resulting chunks in ChromaDB.

---

# 🔎 Metadata-Based Retrieval

One important feature of the system is the ability to retrieve information from a **specific PDF** using metadata.

For example:

```python
filter = {
    "source": pdf1_source
}
```

This allows the retriever to restrict its search to a particular document.

The notebook tests:

* Questions about PDF 1
* Questions about PDF 2
* Questions about PDF 3
* Questions requiring information from multiple PDFs
* Questions that are not present in any PDF

---

# 🧪 RAG Retriever Evaluation

A major part of this project is evaluating the **retriever independently from the LLM**.

Instead of only asking:

> "Does the chatbot answer correctly?"

the project asks:

> "Did the retriever retrieve the right documents in the first place?"

This distinction is important because many RAG failures originate from the retrieval stage.

---

## 📊 Evaluation Metrics

### Precision@K

Measures how many of the retrieved top-K results are relevant.

```text
Precision@K =
Relevant Retrieved Documents / K
```

Higher precision means less irrelevant information is retrieved.

---

### Recall@K

Measures how much of the relevant information was successfully retrieved.

```text
Recall@K =
Relevant Retrieved Sources Found /
Total Relevant Sources
```

Higher recall means fewer relevant sources are missed.

---

### Average Precision (AP)

AP evaluates not only whether relevant information was retrieved, but also **where it appeared in the ranking**.

Relevant results appearing earlier receive more credit.

---

### Mean Average Precision (MAP)

MAP averages AP across the complete evaluation dataset and provides a single score that can be used to compare different retrieval configurations.

---

# 🏆 Golden Test Set

The evaluation uses a manually defined **ground-truth test set**.

Each question is associated with the PDF or PDFs known to contain the correct information.

Example:

```python
{
    "question": "...",
    "relevant_sources": [pdf1_source]
}
```

The test set contains questions requiring:

* One specific PDF
* Multiple PDFs
* Information across all PDFs
* Information that does not exist in the document collection

A retrieved chunk is considered relevant when its `source` metadata matches one of the expected source PDFs.

---

# 📈 Evaluation Results

The notebook evaluates the retriever with:

```text
k = 5
```

The resulting aggregate metrics were:

| Metric           |     Score |
| ---------------- | --------: |
| Mean Precision@5 | **0.367** |
| Mean Recall@5    | **0.555** |
| MAP              | **1.175** |

The individual questions also show significant variation in retrieval performance, demonstrating that retrieval quality depends strongly on the question and document structure.

> **Note:** The reported MAP value should be interpreted carefully because the notebook's AP implementation can produce values greater than 1 for multi-source cases. Therefore, the implementation is useful for experimentation, but the metric calculation should be refined before treating MAP as a standard normalized evaluation score.

---

# ⚙️ Experimentation

The notebook provides a mechanism for comparing different values of:

```python
k = [1, 3, 5, 10]
```

It can also be reused to compare different:

```text
chunk_size
chunk_overlap
k
```

and observe how these choices affect:

* Precision
* Recall
* MAP

This makes the project useful not only as a RAG implementation but also as an **experimental framework for retrieval optimization**.

---

# 🧠 Key RAG Insights

The project demonstrates several important concepts:

### 1. Retrieval Quality Matters

A powerful LLM cannot compensate for completely irrelevant retrieved context.

```text
Bad Retrieval
     ↓
Bad Context
     ↓
Bad Answer
```

### 2. Chunking Is Critical

Poor chunking can split related information and negatively affect retrieval.

The notebook specifically highlights that retrieval failures can originate from bad chunking strategies.

### 3. More Retrieved Documents ≠ Always Better

Increasing `k` can improve recall but may also introduce irrelevant information.

This creates a trade-off:

```text
Small k
→ Higher precision
→ Potentially lower recall

Large k
→ Higher recall
→ Potentially more noise
```

The notebook explicitly discusses this precision/recall trade-off.

---

# 🔬 Example Questions

The RAG system was tested with questions such as:

```text
What is the document about?

What are the techniques of chunking?

What is Agentic Chunking?

How to evaluate RAG?

What is the relation between chunking and evaluating RAG?

What are my main technical skills?

What machine learning experience do I have?
```

It also tests questions outside the available knowledge base, such as:

```text
What clinical trial results for COVID-19 vaccines are reported in these papers?
```

For unsupported questions, the system is designed to return:

```text
I don't know.
```

---

# 🏗️ Architecture

```text
                    ┌──────────────────┐
                    │   PDF Documents  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   PyPDFLoader    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    Chunking      │
                    │ Recursive Split  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Cohere Embeddings│
                    │   embed-v4.0     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    ChromaDB      │
                    │  Vector Store    │
                    └────────┬─────────┘
                             │
                       Similarity Search
                             │
                             ▼
                    ┌──────────────────┐
                    │    Retriever     │
                    │      Top-K       │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Retrieved       │
                    │    Context       │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   Cohere LLM     │
                    │ Command A        │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Grounded Answer │
                    └──────────────────┘


              ┌────────────────────────────┐
              │ Retriever Evaluation       │
              │                            │
              │ Precision@K                │
              │ Recall@K                   │
              │ Average Precision          │
              │ Mean Average Precision     │
              └────────────────────────────┘
```

---

# 📂 Notebook Structure

```text
Multi_PDF_RAG_Lab_with_Evaluation.ipynb
│
├── Environment & Dependencies
│
├── Part 1 — Basic RAG Pipeline
│   ├── Embeddings
│   ├── Chunking
│   ├── ChromaDB
│   ├── Retriever
│   └── Cohere LLM
│
├── Part 2 — Ask Questions About a PDF
│   ├── PDF Loading
│   ├── Chunking Experiments
│   └── RAG Questions
│
├── Part 3 — Multi-PDF RAG
│   ├── Multiple PDF Loading
│   ├── Metadata
│   ├── Vector Store
│   ├── Document Filtering
│   └── Cross-PDF Questions
│
└── Part 4 — Retriever Evaluation
    ├── Golden Test Set
    ├── Precision@K
    ├── Recall@K
    ├── Average Precision
    ├── MAP
    └── k / Chunking Experiments
```

---

# ⚡ Installation

Install the required packages:

```bash
pip install -U langchain
pip install -U langchain-cohere
pip install -U langchain-community
pip install -U langchain-chroma
pip install -U pypdf
pip install -U chromadb
```

---

# 🔑 API Key

The notebook requires a **Cohere API key** for:

* Cohere embeddings
* Cohere Chat model

Set your API key before running the pipeline:

```python
key = "YOUR_COHERE_API_KEY"
```

For production applications, use environment variables or a secrets manager instead of hard-coding API keys.

---

# ▶️ Running the Project

1. Open the notebook in **Google Colab** or Jupyter.
2. Install the required dependencies.
3. Configure your Cohere API key.
4. Upload/provide the required PDF files.
5. Run the notebook cells sequentially.
6. Experiment with:

   * `chunk_size`
   * `chunk_overlap`
   * `k`
7. Run the retriever evaluation.
8. Compare Precision@K, Recall@K, and AP/MAP across configurations.

---

# 🔮 Future Improvements

Possible improvements for turning this experimental notebook into a production-grade RAG system include:

* Add **RAGAS** or **DeepEval** for end-to-end RAG evaluation.
* Fix and normalize the AP/MAP implementation.
* Add **reranking** after initial vector retrieval.
* Experiment with semantic and hierarchical chunking.
* Add hybrid search combining lexical and vector retrieval.
* Add citation/source tracking to generated answers.
* Persist the ChromaDB collection instead of rebuilding it.
* Add a web interface using Streamlit or FastAPI.
* Add automated evaluation datasets.
* Compare multiple embedding models.
* Compare different LLMs.
* Track latency and retrieval cost.
* Add document ingestion for arbitrary PDF uploads.

---

# 📌 What This Project Demonstrates

This project demonstrates practical knowledge of:

* **Retrieval-Augmented Generation (RAG)**
* **Vector Embeddings**
* **Vector Databases**
* **Document Chunking**
* **Semantic Search**
* **Metadata Filtering**
* **Information Retrieval**
* **Retriever Evaluation**
* **Precision@K**
* **Recall@K**
* **Average Precision**
* **Mean Average Precision**
* **LLM Grounding**
* **Multi-document Question Answering**
* **LangChain**

---

# 👨‍💻 Author

**Ahmed Maged Motea**

AI & Data Science / Machine Learning Engineer

Computer Engineering — Zagazig University

---

## ⭐ Project Summary

This project implements a complete **Multi-PDF Retrieval-Augmented Generation system** and evaluates the retrieval component using classical Information Retrieval metrics.

Rather than treating RAG as simply:

```text
Documents → LLM → Answer
```

the project analyzes the complete retrieval pipeline:

```text
Documents
   ↓
Chunking
   ↓
Embeddings
   ↓
Vector Database
   ↓
Retrieval
   ↓
Context Quality
   ↓
LLM Generation
   ↓
Evaluation
```

This makes the notebook a practical exploration of how **document processing, chunking, retrieval configuration, and ranking quality influence the reliability of RAG systems**.

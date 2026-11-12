<div align="center">

# ⚡ RAG-Engineering: The Complete Mastery Journey

> **From Vector Math & Chunking to Multi-Agent Workflows, GraphRAG, and Enterprise Production**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-v0.2+-1C3C3C.svg?style=for-the-badge&logo=langchain&logoColor=white)](https://python.langchain.com/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-v0.10+-6A1B9A.svg?style=for-the-badge)](https://www.llamaindex.ai/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production-009688.svg?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen.svg?style=for-the-badge)](CONTRIBUTING.md)

<p align="center">
  <a href="#-curriculum--navigation-directory"><b>Explore Modules</b></a> •
  <a href="#-12-hands-on-production-projects"><b>Hands-On Projects</b></a> •
  <a href="#-quick-start-guide"><b>Quick Start</b></a> •
  <a href="#-frameworks--tools-deep-dives"><b>Frameworks</b></a> •
  <a href="#-curated-resources--knowledge-base"><b>Resources</b></a>
</p>

</div>

---

## 📖 Introduction

**Retrieval-Augmented Generation (RAG)** has emerged as the architectural cornerstone for enterprise Generative AI systems. While foundational LLMs are powerful, they suffer from knowledge obsolescence, hallucination, lack of private context, and uncontrollable reasoning loops. RAG bridges this gap by grounding LLM responses in real-time, domain-specific, and verifiable context.

**RAG-Engineering** is a production-grade, end-to-end curriculum and engineering reference. Built specifically for AI engineers, software developers, and ML researchers, this repository deconstructs RAG from fundamental linear algebra and document parsing to state-of-the-art hybrid search (Dense + BM25 + RRF), Knowledge Graph RAG (GraphRAG), Multi-Agent State Machines (LangGraph), automated evaluation (RAGAS, DeepEval), and containerized microservice deployments.

---

## 🧭 Curriculum & Navigation Directory

Use this hyperlink index for instantaneous navigation across all 13 modules, practical implementations, and curated resources:

| Stage | Module | Core Engineering Concepts | Direct Link |
| :--- | :--- | :--- | :---: |
| **Stage 1: Core Fundamentals** | [**00 · Introduction to RAG**](00-Introduction-to-RAG/README.md) | Why RAG, RAG vs Fine-Tuning, High-Level Architecture, End-to-End Lifecycle & Enterprise Use Cases | [Explore ↗](00-Introduction-to-RAG/README.md) |
| | [**01 · RAG Fundamentals**](01-RAG-Fundamentals/README.md) | LLM Tokens, Context Windows, Prompt Engineering, Dense Embeddings, Vector Similarity & Cosine Math | [Explore ↗](01-RAG-Fundamentals/README.md) |
| | [**02 · Document Processing**](02-Document-Processing/README.md) | Loaders (PDF, Web, HTML), Text Extraction & Cleaning, Fixed/Recursive/Semantic Chunking & Metadata | [Explore ↗](02-Document-Processing/README.md) |
| | [**03 · Embeddings & Vector DBs**](03-Embeddings-and-Vector-Databases/README.md) | Dense vs Sparse, HNSW Indexing, Metadata Filtering, FAISS, ChromaDB, Pinecone, Qdrant, Weaviate | [Explore ↗](03-Embeddings-and-Vector-Databases/README.md) |
| **Stage 2: Retrieval Engineering** | [**04 · Retrieval Strategies**](04-Retrieval-Strategies/README.md) | BM25, Hybrid Search (Reciprocal Rank Fusion), MMR, Query Expansion, Rewriting, Parent-Doc & Contextual | [Explore ↗](04-Retrieval-Strategies/README.md) |
| | [**05 · RAG Pipeline Architecture**](05-RAG-Pipeline/README.md) | Ingestion & Indexing Pipelines, Retrieval & Generation Orchestration, Citation Attribution, Hallucination Mitigation | [Explore ↗](05-RAG-Pipeline/README.md) |
| **Stage 3: Advanced & Agentic Systems** | [**06 · Advanced RAG**](06-Advanced-RAG/README.md) | Cross-Encoder Reranking, ColBERT Late-Interaction, Self-RAG, Corrective RAG (CRAG), GraphRAG, RAPTOR, HyDE | [Explore ↗](06-Advanced-RAG/README.md) |
| | [**07 · Agentic RAG**](07-Agentic-RAG/README.md) | ReAct Reasoning Loops, Autonomous Tool-Calling, LangGraph State Machines, Multi-Agent Swarms & Human-in-the-Loop | [Explore ↗](07-Agentic-RAG/README.md) |
| **Stage 4: Evaluation & Production** | [**08 · Evaluation & Optimization**](08-RAG-Evaluation-and-Optimization/README.md) | RAGAS Metrics (Faithfulness, Precision, Recall), DeepEval, LLM-as-a-Judge, Latency & Token Cost Profiling | [Explore ↗](08-RAG-Evaluation-and-Optimization/README.md) |
| | [**09 · Deployment & Production**](09-RAG-Deployment-and-Production/README.md) | FastAPI Async Microservices, Streamlit UIs, Dockerization, Redis Semantic Caching, RBAC Security & OpenTelemetry | [Explore ↗](09-RAG-Deployment-and-Production/README.md) |
| **Stage 5: Applications & Tools** | [**10 · 12 Hands-On Projects**](10-Hands-On-Projects/README.md) | 12 Complete Production-Grade Applications: Simple QA, Codebase Search, GraphRAG, Vision RAG, Enterprise APIs | [Explore ↗](10-Hands-On-Projects/README.md) |
| | [**11 · Frameworks & Ecosystem**](11-Frameworks-and-Tools/README.md) | In-Depth Architecture Deep Dives: LangChain, LlamaIndex, Haystack, LangGraph, Ollama, OpenAI, Hugging Face | [Explore ↗](11-Frameworks-and-Tools/README.md) |
| | [**12 · Curated Resources**](12-Resources/README.md) | Landmark Research Papers, Authoritative Blogs, Official Docs, Benchmark Datasets & Master Technical Glossary | [Explore ↗](12-Resources/README.md) |

### ⚡ Quick Navigation Matrix

```
┌───────────────────────────────┬────────────────────────────────────────────────────────────────────────┐
│ Quick Jump Category           │ Direct Markdown Jump Links                                             │
├───────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 🛠️  Framework Guides          │ • [LangChain](11-Frameworks-and-Tools/LangChain/README.md)             │
│                               │ • [LlamaIndex](11-Frameworks-and-Tools/LlamaIndex/README.md)           │
│                               │ • [LangGraph](11-Frameworks-and-Tools/LangGraph/README.md)             │
│                               │ • [Haystack](11-Frameworks-and-Tools/Haystack/README.md)               │
│                               │ • [Ollama (Local)](11-Frameworks-and-Tools/Ollama/README.md)           │
│                               │ • [HuggingFace](11-Frameworks-and-Tools/HuggingFace/README.md)         │
│                               │ • [OpenAI](11-Frameworks-and-Tools/OpenAI/README.md)                   │
├───────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 📚  Knowledge Base            │ • [Research Papers](12-Resources/Research-Papers.md)                   │
│                               │ • [Technical Blogs](12-Resources/Blogs.md)                             │
│                               │ • [Official Documentation](12-Resources/Documentation.md)              │
│                               │ • [YouTube Masterclasses](12-Resources/YouTube-Resources.md)           │
│                               │ • [Benchmark Datasets](12-Resources/Datasets.md)                       │
│                               │ • [Engineering Glossary](12-Resources/Glossary.md)                     │
├───────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 🚀  Flagship Projects         │ • [Codebase RAG (AST-Aware)](10-Hands-On-Projects/07-Codebase-RAG/)    │
│                               │ • [GraphRAG Knowledge Engine](10-Hands-On-Projects/09-GraphRAG-Application/) │
│                               │ • [Multi-Modal Vision RAG](10-Hands-On-Projects/10-Multi-Modal-RAG/)   │
│                               │ • [LangGraph Agentic Assistant](10-Hands-On-Projects/11-Agentic-RAG-Assistant/) │
│                               │ • [Production Microservice](10-Hands-On-Projects/12-Production-Ready-RAG/) │
└───────────────────────────────┴────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Architectural Pillars of this Repository

<table>
<tr>
<td width="33%" valign="top">

### 🔬 Mathematical Rigor
- High-dimensional vector space geometry
- Dot product, Cosine, L2 (Euclidean), and Manhattan distances
- Dimensionality reduction, quantization (PQ, SQ), and HNSW graph traversal mechanics

</td>
<td width="33%" valign="top">

### ⚡ Hybrid Search & RRF
- Dense vector semantic search combined with sparse lexical scoring (BM25)
- Reciprocal Rank Fusion (RRF) for unified ranking
- Cross-encoder rerankers and ColBERT late-interaction token scoring

</td>
<td width="33%" valign="top">

### 🕸️ Graph & Hierarchical RAG
- Neo4j entity-relation extraction and community summarization (GraphRAG)
- Recursive Abstractive Processing for Tree-Organized Retrieval (RAPTOR)
- Parent-Document and Small-to-Big contextual retrieval

</td>
</tr>
<tr>
<td width="33%" valign="top">

### 🤖 Agentic Reflection Loops
- ReAct (Reasoning + Acting) decision engines
- Self-RAG (retrieval on demand, self-critique tokens)
- Corrective RAG (CRAG) with autonomous web search fallbacks
- LangGraph cyclical state machines

</td>
<td width="33%" valign="top">

### 📊 Quantitative Evaluation
- Full RAG Triad assessment: Faithfulness, Answer Relevance, Context Precision, and Context Recall
- RAGAS and DeepEval automated test suites
- Synthetic test data generation with LLM-as-a-Judge guardrails

</td>
<td width="33%" valign="top">

### 🛡️ Production Engineering
- Asynchronous FastAPI endpoints with streaming responses
- Semantic caching with Redis to reduce token costs by up to 60%
- Dockerized multi-stage containers, RBAC security, and OpenTelemetry instrumentation

</td>
</tr>
</table>

---

## 📂 Repository Layout

```
RAG-Engineering/
├── README.md                                  # Repository overview and navigation hub
├── LICENSE                                    # Open-source MIT license
├── CONTRIBUTING.md                             # Contribution guidelines and styling rules
├── requirements.txt                           # Universal Python dependencies
├── .gitignore                                 # Git ignore patterns
│
├── 00-Introduction-to-RAG/                    # Stage 1: Concepts, lifecycle, vs fine-tuning
├── 01-RAG-Fundamentals/                       # Stage 1: Math, tokens, context, embeddings
├── 02-Document-Processing/                    # Stage 1: Parsers, cleaning, semantic chunking
├── 03-Embeddings-and-Vector-Databases/        # Stage 1: Vector engines (FAISS, Chroma, Qdrant...)
│
├── 04-Retrieval-Strategies/                   # Stage 2: Hybrid, BM25, RRF, MMR, contextual
├── 05-RAG-Pipeline/                           # Stage 2: Full pipeline, citations, grounding
│
├── 06-Advanced-RAG/                           # Stage 3: GraphRAG, RAPTOR, ColBERT, Self-RAG
├── 07-Agentic-RAG/                            # Stage 3: LangGraph, tool-calling, ReAct agents
│
├── 08-RAG-Evaluation-and-Optimization/        # Stage 4: RAGAS, DeepEval, latency & cost
├── 09-RAG-Deployment-and-Production/          # Stage 4: FastAPI, Docker, Redis cache, security
│
├── 10-Hands-On-Projects/                      # 12 End-to-end practical RAG applications
│   ├── 01-Simple-RAG-Chatbot/
│   ├── 02-PDF-Question-Answering/
│   ├── 03-Multiple-PDF-RAG/
│   ├── 04-Website-RAG-Chatbot/
│   ├── 05-Youtube-Video-RAG/
│   ├── 06-SQL-RAG-Assistant/
│   ├── 07-Codebase-RAG/
│   ├── 08-Hybrid-Search-RAG/
│   ├── 09-GraphRAG-Application/
│   ├── 10-Multi-Modal-RAG/
│   ├── 11-Agentic-RAG-Assistant/
│   └── 12-Production-Ready-RAG/
│
├── 11-Frameworks-and-Tools/                   # Deep dives on LangChain, LlamaIndex, Haystack...
└── 12-Resources/                              # Landmark papers, blogs, datasets, glossary
```

---

## 🚀 Quick Start Guide

Get up and running with the codebase in 4 straightforward steps:

### 1. Clone the Repository
```bash
git clone https://github.com/swapnil3104/RAG-Engineering.git
cd RAG-Engineering
```

### 2. Set Up a Virtual Environment
```bash
# Create environment
python -m venv .venv

# Activate on Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Activate on Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory:
```env
# Primary LLM & Embedding Providers
OPENAI_API_KEY=sk-proj-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
COHERE_API_KEY=your-cohere-key-here

# Vector Databases
PINECONE_API_KEY=your-pinecone-key-here
PINECONE_ENVIRONMENT=your-pinecone-env
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your-qdrant-key-here

# Graph & Search Tools
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
TAVILY_API_KEY=tvly-your-tavily-key-here
```

---

## 💻 12 Hands-On Production Projects

The repository includes 12 progressive, fully documented hands-on projects taking you from simple baseline prototypes to production microservices:

| # | Project Title | Architecture Highlights | Primary Stack | Documentation & Code |
| :---: | :--- | :--- | :--- | :---: |
| **01** | **[Simple RAG Chatbot](10-Hands-On-Projects/01-Simple-RAG-Chatbot/README.md)** | Baseline document QA assistant with conversational memory | LangChain, ChromaDB, Streamlit | `✅ Complete` |
| **02** | **[PDF Question Answering](10-Hands-On-Projects/02-PDF-Question-Answering/README.md)** | Deep parsing, layout analysis, table extraction & exact page citations | PyMuPDF, pdfplumber, FAISS | `✅ Complete` |
| **03** | **[Multiple PDF RAG](10-Hands-On-Projects/03-Multiple-PDF-RAG/README.md)** | Cross-document research engine with metadata filtering & source attribution | ChromaDB, Metadata Filters, Streamlit | `✅ Complete` |
| **04** | **[Website RAG Chatbot](10-Hands-On-Projects/04-Website-RAG-Chatbot/README.md)** | Automated web crawler, HTML boilerplate cleaning & documentation Q&A | Trafilatura, Playwright, Qdrant | `✅ Complete` |
| **05** | **[YouTube Video RAG](10-Hands-On-Projects/05-Youtube-Video-RAG/README.md)** | Video transcript extraction, timestamp chunking & direct video jump links | youtube-transcript-api, ChromaDB | `✅ Complete` |
| **06** | **[SQL RAG Assistant](10-Hands-On-Projects/06-SQL-RAG-Assistant/README.md)** | Text-to-SQL translator with dynamic schema retrieval & automated verification | PostgreSQL, SQLAlchemy, OpenAI | `✅ Complete` |
| **07** | **[Codebase RAG](10-Hands-On-Projects/07-Codebase-RAG/README.md)** | AST-aware code chunking across classes, functions, and cross-file dependencies | Tree-sitter, Qdrant, LangChain | `✅ Complete` |
| **08** | **[Hybrid Search RAG](10-Hands-On-Projects/08-Hybrid-Search-RAG/README.md)** | Dense semantic embeddings + BM25 keyword matching fused via Reciprocal Rank Fusion | Rank-BM25, ChromaDB, Cross-Encoder | `✅ Complete` |
| **09** | **[GraphRAG Application](10-Hands-On-Projects/09-GraphRAG-Application/README.md)** | Neo4j knowledge graph extraction, entity resolution & community summaries | Neo4j, NetworkX, LangChain | `✅ Complete` |
| **10** | **[Multi-Modal RAG](10-Hands-On-Projects/10-Multi-Modal-RAG/README.md)** | Multi-modal ingestion of text, charts, diagrams, tables, and infographics | ColPali, GPT-4o Vision, Unstructured | `✅ Complete` |
| **11** | **[Agentic RAG Assistant](10-Hands-On-Projects/11-Agentic-RAG-Assistant/README.md)** | Cyclical ReAct loops, dynamic query decomposition & autonomous search fallbacks | LangGraph, Tavily Search, ChromaDB | `✅ Complete` |
| **12** | **[Production-Ready RAG](10-Hands-On-Projects/12-Production-Ready-RAG/README.md)** | High-throughput async microservice with Redis semantic cache, Docker & RAGAS CI | FastAPI, Redis, Docker, RAGAS | `✅ Complete` |

---

## 🛠️ Frameworks & Tools Deep Dives

Each major framework in the modern RAG ecosystem is covered with architecture teardowns, idiomatic design patterns, and code recipes in [`11-Frameworks-and-Tools`](11-Frameworks-and-Tools/README.md):

- 🦜 **[LangChain](11-Frameworks-and-Tools/LangChain/README.md)**: LCEL (LangChain Expression Language), Document Loaders, VectorStores, and Retrievable Chains.
- 🦙 **[LlamaIndex](11-Frameworks-and-Tools/LlamaIndex/README.md)**: Data Connectors, Hierarchical Node Parsers, Vector/Summary/Keyword Indices, and Query Engines.
- 🔀 **[LangGraph](11-Frameworks-and-Tools/LangGraph/README.md)**: Stateful graphs, cyclical conditional edges, checkpoint persistence, and multi-agent coordination.
- 🌾 **[Haystack](11-Frameworks-and-Tools/Haystack/README.md)**: Directed Acyclic Graph (DAG) pipelines, modular components, and document stores.
- 🦙 **[Ollama](11-Frameworks-and-Tools/Ollama/README.md)**: Zero-cloud, local inference with Llama 3, Mistral, and local embeddings (Nomic, BGE).
- 🤗 **[Hugging Face](11-Frameworks-and-Tools/HuggingFace/README.md)**: Transformers, `sentence-transformers`, TEI (Text Embeddings Inference), and TGI.
- 🤖 **[OpenAI](11-Frameworks-and-Tools/OpenAI/README.md)**: Assistants API, Tool-Calling, Structured Outputs, and text-embedding-3 models.

---

## 📚 Curated Resources & Knowledge Base

Accelerate your learning journey with our vetted repository knowledge base in [`12-Resources`](12-Resources/README.md):

- 📄 **[Research Papers](12-Resources/Research-Papers.md)**: Annotated summaries and key findings of foundational papers (Lewis et al. 2020, ColBERT, RETRO, Self-RAG, CRAG, RAPTOR, GraphRAG, ColPali).
- 📰 **[Technical Blogs](12-Resources/Blogs.md)**: Engineering deep dives from OpenAI, Anthropic, Pinecone, Cohere, and LlamaIndex.
- 📖 **[Documentation Hub](12-Resources/Documentation.md)**: Authoritative links and best practices for libraries and vector engines.
- 🎥 **[YouTube Masterclasses](12-Resources/YouTube-Resources.md)**: Curated video lectures, architectural walk-throughs, and code sessions.
- 📊 **[Benchmark Datasets](12-Resources/Datasets.md)**: MS MARCO, BEIR, HotpotQA, MultiHop-RAG, Natural Questions, and TriviaQA.
- 🔤 **[Engineering Glossary](12-Resources/Glossary.md)**: Over 100+ formal definitions of key retrieval, embedding, and LLM terms.

---

## 🤝 Contributing

Contributions are what make the open-source AI community such an incredible place to learn, inspire, and create. Any contributions you make are **warmly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingRAGFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingRAGFeature'`)
4. Push to the Branch (`git push origin feature/AmazingRAGFeature`)
5. Open a Pull Request

Please review our [CONTRIBUTING.md](CONTRIBUTING.md) for full style guidelines and code formatting standards.

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

---

<div align="center">
  <sub>Built with ❤️ for AI Engineers and the open-source Generative AI community.</sub>
</div>

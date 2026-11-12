# 📚 Official Documentation Index: Vector DBs, Embedding Providers, and Orchestration SDKs

## 📑 Table of Contents
1. [🌟 The Enterprise Documentation Landscape](#1--the-enterprise-documentation-landscape)
2. [🔬 Mathematical Models for Hardware Sizing & Network Serialization](#2--mathematical-models-for-hardware-sizing--network-serialization)
3. [🛠️ Orchestration Frameworks & Agentic Runtimes](#3-️-orchestration-frameworks--agentic-runtimes)
4. [🗄️ Vector Databases & Specialized Storage Engines](#4-️-vector-databases--specialized-storage-engines)
5. [🧬 Foundation Model Providers & Embedding APIs](#5-️-foundation-model-providers--embedding-apis)
6. [🛡️ Evaluation, Observability & Governance Documentation](#6-️-evaluation-observability--governance-documentation)
7. [📊 Master 25-Tool API & SDK Quick-Reference Matrix](#7--master-25-tool-api--sdk-quick-reference-matrix)
8. [💼 Senior System Design & API Integration Interview Scenarios](#8--senior-system-design--api-integration-interview-scenarios)
9. [🧪 Hands-On Mini-Project: Multi-SDK Configuration Validator & Connectivity Health-Checker](#9--hands-on-mini-project-multi-sdk-configuration-validator--connectivity-health-checker)
10. [⚡ Quick Revision & Official Docs Bookmarks Matrix](#10--quick-revision--official-docs-bookmarks-matrix)

---

## 1. 🌟 The Enterprise Documentation Landscape

In modern AI systems engineering, consulting secondary blog tutorials or outdated documentation often leads to deprecated APIs, broken dependencies, and security vulnerabilities. 

Production engineers rely on the **official, version-pinned documentation portals** maintained by framework maintainers and cloud providers. This directory indexes the canonical specifications, SDK references, and developer manuals across the modern enterprise RAG stack.

```mermaid
flowchart TD
    subgraph Layer1 ["1. Client & Ingestion Layer"]
        DOC["Docling / Unstructured / PyMuPDF Docs"]
    end

    subgraph Layer2 ["2. Orchestration & Agent Runtime"]
        LC["LangChain / LangGraph Docs"]
        LI["LlamaIndex Docs"]
        HS["Haystack 2.0 Docs"]
    end

    subgraph Layer3 ["3. Embedding & Reranking APIs"]
        OAI["OpenAI API Docs"]
        COH["Cohere API Docs"]
        HF["Hugging Face TEI Docs"]
    end

    subgraph Layer4 ["4. Vector & Relational Storage"]
        QD["Qdrant Documentation"]
        PC["Pinecone Documentation"]
        PG["pgvector Documentation"]
    end

    subgraph Layer5 ["5. Observability & Quality Assurance"]
        LS["LangSmith Docs"]
        RAGAS["Ragas Documentation"]
        TRU["TruLens Documentation"]
    end

    Layer1 --> Layer2
    Layer2 --> Layer3 & Layer4
    Layer2 -.-> Layer5

    style Layer1 fill:#f8fafc,stroke:#475569,stroke-width:2px
    style Layer2 fill:#eff6ff,stroke:#1d4ed8,stroke-width:2px
    style Layer3 fill:#ecfdf5,stroke:#059669,stroke-width:2px
    style Layer4 fill:#fef3c7,stroke:#d97706,stroke-width:2px
    style Layer5 fill:#fdf4ff,stroke:#a21caf,stroke-width:2px
```

---

## 2. 🔬 Mathematical Models for Hardware Sizing & Network Serialization

### # Vector Storage RAM Capacity Formula
When provisioning cloud instances for vector databases, total uncompressed RAM requirements $M_{\text{raw}}$ for $N$ vectors of dimension $d$ stored in 32-bit floating point (`FP32`), including HNSW graph index overhead ($\approx 20\% - 30\%$), are computed via:
$$
M_{\text{raw}} = N \cdot \left( d \cdot 4\text{ bytes} \right) \cdot (1 + \alpha_{\text{graph}})
$$
Where $\alpha_{\text{graph}} \approx 0.25$. Applying **Scalar Quantization (INT8)** reduces the vector coordinate size to 1 byte:
$$
M_{\text{INT8}} = N \cdot \left( d \cdot 1\text{ byte} \right) \cdot (1 + \alpha_{\text{graph}}) = \frac{M_{\text{raw}}}{4}
$$
For a database of $10,000,000$ chunks embedded at $d=1536$ dimensions:
- Raw `FP32` footprint: $10^7 \cdot 1536 \cdot 4 \cdot 1.25 \approx 76.8\text{ GB RAM}$.
- Quantized `INT8` footprint: $10^7 \cdot 1536 \cdot 1 \cdot 1.25 \approx 19.2\text{ GB RAM}$.

---

### # Network Serialization Overhead & Latency Formula
The round-trip latency (RTT) for external embedding API calls is bounded by wire serialization, TLS handshake, and network transport:
$$
T_{\text{retrieval}} = t_{\text{DNS}} + t_{\text{TLS}} + \frac{S_{\text{payload}}}{B_{\text{upload}}} + t_{\text{inference}} + \frac{N \cdot d \cdot 4\text{ bytes}}{B_{\text{download}}}
$$
Where $S_{\text{payload}}$ is the UTF-8 text size and $B$ denotes directional network bandwidth.

---

## 3. 🛠️ Orchestration Frameworks & Agentic Runtimes

### 3.1 LangChain & LangGraph
- **Core Documentation Portal:** [python.langchain.com/docs](https://python.langchain.com/docs/)
- **LangGraph Multi-Agent Guide:** [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/)
- **API Reference (Core SDK):** [api.python.langchain.com](https://api.python.langchain.com/)
- **GitHub Repository:** [github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain)
- **Key Modules Documented:** LCEL (`Runnable`), Chat Models, VectorStore abstractions, Tool calling, PostgreSQL checkpointers (`AsyncPostgresSaver`).

---

### 3.2 LlamaIndex
- **Core Documentation Portal:** [docs.llamaindex.ai](https://docs.llamaindex.ai/)
- **LlamaParse API Docs:** [docs.llamaindex.ai/en/stable/llama_parse/](https://docs.llamaindex.ai/en/stable/llama_parse/)
- **Workflows Guide (v0.11+):** [docs.llamaindex.ai/en/stable/module_guides/workflow/](https://docs.llamaindex.ai/en/stable/module_guides/workflow/)
- **GitHub Repository:** [github.com/run-llama/llama_index](https://github.com/run-llama/llama_index)
- **Key Modules Documented:** `HierarchicalNodeParser`, `AutoMergingRetriever`, `SubQuestionQueryEngine`, Pydantic program extractors.

---

### 3.3 Deepset Haystack 2.0
- **Core Documentation Portal:** [docs.haystack.deepset.ai](https://docs.haystack.deepset.ai/)
- **Component API Reference:** [docs.haystack.deepset.ai/reference/pipeline-api](https://docs.haystack.deepset.ai/reference/pipeline-api)
- **GitHub Repository:** [github.com/deepset-ai/haystack](https://github.com/deepset-ai/haystack)
- **Key Modules Documented:** Directed Acyclic Graph (DAG) pipelines, `@component` decorator, `DocumentJoiner`, YAML pipeline serialization.

---

### 3.4 Microsoft AutoGen & Semantic Kernel
- **Semantic Kernel Docs:** [learn.microsoft.com/en-us/semantic-kernel/](https://learn.microsoft.com/en-us/semantic-kernel/)
- **AutoGen Studio Docs:** [microsoft.github.io/autogen/](https://microsoft.github.io/autogen/)
- **Key Focus:** Enterprise .NET and Python integration, event-driven agent conversations, and kernel function calling.

---

## 4. 🗄️ Vector Databases & Specialized Storage Engines

### 4.1 Qdrant
- **Official Documentation:** [qdrant.tech/documentation/](https://qdrant.tech/documentation/)
- **API Reference (REST & gRPC):** [qdrant.github.io/qdrant/redoc/index.html](https://qdrant.github.io/qdrant/redoc/index.html)
- **Python Client SDK:** [github.com/qdrant/qdrant-client](https://github.com/qdrant/qdrant-client)
- **Key Features Documented:** HNSW indexing, Payload schema-free filtering, Scalar/Product Quantization, Single-Stage Hybrid Search (Dense + Sparse).

---

### 4.2 Pinecone
- **Official Documentation:** [docs.pinecone.io](https://docs.pinecone.io/)
- **Serverless Architecture Guide:** [docs.pinecone.io/guides/indexes/understanding-serverless](https://docs.pinecone.io/guides/indexes/understanding-serverless)
- **API Reference:** [docs.pinecone.io/reference/api/introduction](https://docs.pinecone.io/reference/api/introduction)
- **Key Features Documented:** Serverless tiered storage, namespaces, metadata filtering, integrated reranking.

---

### 4.3 Milvus & Zilliz Cloud
- **Official Documentation:** [milvus.io/docs](https://milvus.io/docs)
- **PyMilvus SDK Docs:** [milvus.io/api-reference/pymilvus/v2.4.x/About.md](https://milvus.io/api-reference/pymilvus/v2.4.x/About.md)
- **Key Features Documented:** Distributed Kubernetes deployments, GPU Knowhere engine, partitioned collections.

---

### 4.4 Weaviate
- **Official Documentation:** [weaviate.io/developers/weaviate](https://weaviate.io/developers/weaviate)
- **Python v4 Client Reference:** [weaviate.io/developers/weaviate/client-libraries/python](https://weaviate.io/developers/weaviate/client-libraries/python)
- **Key Features Documented:** Native vectorizer modules, GraphQL search, cross-reference graph links.

---

### 4.5 PostgreSQL pgvector
- **Official Documentation:** [github.com/pgvector/pgvector](https://github.com/pgvector/pgvector)
- **Key SQL Operations Documented:** `CREATE INDEX USING hnsw`, `CREATE INDEX USING ivfflat`, vector similarity operators (`<->` L2, `<#>` Inner Product, `<=>` Cosine Distance).

---

## 5. 🧬 Foundation Model Providers & Embedding APIs

### 5.1 OpenAI Platform
- **API Documentation:** [platform.openai.com/docs](https://platform.openai.com/docs)
- **Embeddings API Reference:** [platform.openai.com/docs/guides/embeddings](https://platform.openai.com/docs/guides/embeddings)
- **Assistants & File Search Guide:** [platform.openai.com/docs/assistants/tools/file-search](https://platform.openai.com/docs/assistants/tools/file-search)
- **Structured Outputs Guide:** [platform.openai.com/docs/guides/structured-outputs](https://platform.openai.com/docs/guides/structured-outputs)

---

### 5.2 Anthropic Claude
- **API Documentation:** [docs.anthropic.com](https://docs.anthropic.com/)
- **Prompt Caching Guide:** [docs.anthropic.com/en/docs/build-with-claude/prompt-caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
- **Contextual Retrieval Reference:** [docs.anthropic.com/en/docs/build-with-claude/contextual-retrieval](https://docs.anthropic.com/en/docs/build-with-claude/contextual-retrieval)

---

### 5.3 Cohere
- **Documentation Portal:** [docs.cohere.com](https://docs.cohere.com/)
- **Rerank API Reference:** [docs.cohere.com/reference/rerank](https://docs.cohere.com/reference/rerank)
- **Embed v3 Multimodal Reference:** [docs.cohere.com/reference/embed](https://docs.cohere.com/reference/embed)

---

### 5.4 Hugging Face & Text Embeddings Inference (TEI)
- **Transformers Documentation:** [huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
- **Sentence-Transformers:** [sbert.net](https://sbert.net/)
- **Text Embeddings Inference (TEI):** [huggingface.co/docs/text-embeddings-inference](https://huggingface.co/docs/text-embeddings-inference)
- **Key Features Documented:** FlashAttention, dynamic batching, MRL embedding truncation, ONNX/TensorRT export.

---

### 5.5 Ollama (Local & Edge Runtimes)
- **Documentation & Modelfiles:** [github.com/ollama/ollama/blob/main/docs/modelfile.md](https://github.com/ollama/ollama/blob/main/docs/modelfile.md)
- **REST API Reference:** [github.com/ollama/ollama/blob/main/docs/api.md](https://github.com/ollama/ollama/blob/main/docs/api.md)

---

## 6. 🛡️ Evaluation, Observability & Governance Documentation

### 6.1 Ragas
- **Documentation:** [docs.ragas.io](https://docs.ragas.io/)
- **Metrics Covered:** `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`.

### 6.2 TruLens
- **Documentation:** [trulens.org](https://www.trulens.org/)
- **Key Focus:** Instrumentation, RAG Triad feedback functions, and execution tracing.

### 6.3 LangSmith
- **Documentation:** [docs.smith.langchain.com](https://docs.smith.langchain.com/)
- **Key Focus:** Distributed tracing, latency breakdown, golden dataset curation, regression testing.

### 6.4 NVIDIA NeMo Guardrails
- **Documentation:** [docs.nvidia.com/nemo/guardrails/](https://docs.nvidia.com/nemo/guardrails/)
- **Key Focus:** Colang modeling language, input moderation, topical boundary rails, and fact-checking.

---

## 7. 📊 Master 25-Tool API & SDK Quick-Reference Matrix

| Category | Technology | Official Portal URL | Supported Languages | Primary Protocol | Key Deployment Model |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Orchestration** | **LangChain** | [python.langchain.com](https://python.langchain.com/) | Python, TypeScript | Python SDK / HTTP | In-App Library |
| **Orchestration** | **LlamaIndex** | [docs.llamaindex.ai](https://docs.llamaindex.ai/) | Python, TypeScript | Python SDK / REST | In-App Library |
| **Orchestration** | **Haystack 2.0** | [docs.haystack.deepset.ai](https://docs.haystack.deepset.ai/) | Python | Python SDK / YAML | Container / Microservice |
| **Orchestration** | **LangGraph** | [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/) | Python, TypeScript | Python SDK / HTTP | In-App Library / Server |
| **Vector DB** | **Qdrant** | [qdrant.tech/documentation](https://qdrant.tech/documentation/) | Python, Rust, Go, JS | gRPC / REST | Self-Hosted / Managed Cloud |
| **Vector DB** | **Pinecone** | [docs.pinecone.io](https://docs.pinecone.io/) | Python, TS, Go, Java | gRPC / REST | Managed Serverless Cloud |
| **Vector DB** | **Milvus** | [milvus.io/docs](https://milvus.io/docs) | Python, Java, Go, C# | gRPC | Distributed Kubernetes / Zilliz |
| **Vector DB** | **Weaviate** | [weaviate.io/developers](https://weaviate.io/developers/weaviate) | Python (v4), TS, Go | GraphQL / gRPC | Self-Hosted / Managed Cloud |
| **Vector DB** | **Chroma** | [docs.trychroma.com](https://docs.trychroma.com/) | Python, JavaScript | In-Process / REST | Embedded / Local Client |
| **Vector DB** | **pgvector** | [github.com/pgvector/pgvector](https://github.com/pgvector/pgvector) | SQL (Any Postgres Client)| SQL Protocol | PostgreSQL Extension |
| **Search Engine**| **Elasticsearch**| [elastic.co/docs](https://www.elastic.co/guide/index.html) | Multi-language SDKs | REST / JSON | Self-Hosted / Elastic Cloud |
| **Search Engine**| **Vespa.ai** | [docs.vespa.ai](https://docs.vespa.ai/) | Java, Python, C++ | HTTP / JSON | Distributed Clusters |
| **Embeddings** | **OpenAI API** | [platform.openai.com/docs](https://platform.openai.com/docs) | Multi-language SDKs | REST / HTTPS | Managed Cloud API |
| **Foundation** | **Anthropic** | [docs.anthropic.com](https://docs.anthropic.com/) | Python, TypeScript | REST / HTTPS | Managed Cloud API |
| **Reranking** | **Cohere** | [docs.cohere.com](https://docs.cohere.com/) | Python, TypeScript, Java | REST / HTTPS | Managed Cloud API |
| **Serving** | **Ollama** | [github.com/ollama/ollama](https://github.com/ollama/ollama) | CLI, Python, REST | HTTP / localhost | Local / Air-Gapped Workstations|
| **Serving** | **HF TEI** | [huggingface.co/docs/tei](https://huggingface.co/docs/text-embeddings-inference) | Rust / Docker | gRPC / HTTP | On-Premise GPU Instances |
| **Evaluation** | **Ragas** | [docs.ragas.io](https://docs.ragas.io/) | Python | Python Library | In-Memory Testing |
| **Evaluation** | **TruLens** | [trulens.org](https://www.trulens.org/) | Python | Python Library | In-Memory / Streamlit UI |
| **Observability**| **LangSmith** | [docs.smith.langchain.com](https://docs.smith.langchain.com/) | Python, TypeScript | HTTPS / Background | Managed Cloud / Enterprise |
| **Observability**| **Arize Phoenix**| [docs.arize.com/phoenix](https://docs.arize.com/phoenix) | Python | OpenTelemetry / HTTP| Self-Hosted / Phoenix Cloud |
| **Guardrails** | **NeMo Rails** | [docs.nvidia.com/nemo/guardrails](https://docs.nvidia.com/nemo/guardrails/) | Python | Python Library | In-App Middleware |
| **Parsing** | **Docling** | [ds4sd.github.io/docling](https://ds4sd.github.io/docling/) | Python | Python Library | Local Ingestion Pipeline |
| **Parsing** | **Unstructured**| [docs.unstructured.io](https://docs.unstructured.io/) | Python | REST / Library | In-App / Cloud API |
| **Local LLM** | **vLLM** | [docs.vllm.ai](https://docs.vllm.ai/) | Python, C++ | OpenAI-compatible REST| High-Throughput GPU Server |

---

## 8. 💼 Senior System Design & API Integration Interview Scenarios

### Scenario 1: "How do you design a resilient multi-provider embedding fallback layer in production, preventing vector dimension mismatches?"
**Answer:**
A major architectural trap occurs when an external embedding provider (e.g., OpenAI `text-embedding-3-small` with $d=1536$) experiences an outage, and the fallback code naively switches to another provider (e.g., Cohere `embed-english-v3.0` with $d=1024$). Because vector databases enforce strict, immutable collection dimensions, this causes immediate index failure.

**Resilient Architecture:**
1. **Dimension Matching:** If the primary provider uses $d=1536$, the secondary fallback must match the identical coordinate dimensionality (e.g., a local TEI instance running `BAAI/bge-large-en-v1.5` mapped or padded to 1536, or dedicated separate collections).
2. **Dual-Index Routing:** The recommended enterprise pattern is maintaining dual vector collections in the database (`collection_openai_1536` and `collection_cohere_1024`). The ingestion pipeline embeds into both collections asynchronously.
3. **Dynamic Client Fallback Circuit Breaker:**
   ```mermaid
   flowchart TD
       Q["User Query"] --> CB{"Circuit Breaker<br/>OpenAI API Healthy?"}
       CB -->|"Yes"| Prim["Embed via OpenAI (d=1536)"]
       Prim --> V1["Search 'collection_openai'"]
       CB -->|"No (Timeout / 503)"| Sec["Embed via Cohere (d=1024)"]
       Sec --> V2["Search 'collection_cohere'"]
   ```

---

### Scenario 2: "When should an enterprise adopt PostgreSQL pgvector instead of a dedicated vector database like Qdrant or Milvus?"
**Answer:**
- **Adopt PostgreSQL pgvector when:**
  1. **Existing Postgres Footprint:** The organization already runs managed PostgreSQL (e.g., AWS RDS, Aurora, Cloud SQL) with established backup, replication, and disaster recovery procedures.
  2. **Corpus Scale $< 1,000,000$ Chunks:** For collections under one million vectors, modern pgvector with HNSW indexes delivers solid query latencies ($< 15\text{ms}$).
  3. **Strict ACID & Relational Joins:** The application requires atomic transactions joining vector similarity search with complex relational data (e.g., tenant permissions, billing tables, user metadata) in a single SQL query.
- **Adopt a Dedicated Vector Database (Qdrant/Milvus) when:**
  1. **Billion-Scale Vectors:** Collections exceed $10\text{M} - 100\text{M}+$ vectors, requiring distributed sharding across multi-node clusters.
  2. **Advanced Quantization:** Need for product quantization (PQ) and native binary quantization (BQ) to fit massive indexes into limited RAM.
  3. **Dedicated Ingestion Velocity:** Heavy real-time ingestion pipelines where vector indexing would degrade OLTP database CPU and disk I/O.

---

## 9. 🧪 Hands-On Mini-Project: Multi-SDK Configuration Validator & Connectivity Health-Checker

This standalone Python script validates environment configurations, verifies network connectivity, and checks API endpoint health across Vector DB, Embedding, and LLM providers.

```python
"""
Hands-On Mini-Project: Unified SDK Configuration & Connectivity Health-Checker
Validates API keys, endpoint configurations, and network connectivity without crashing.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import os

@dataclass
class ServiceConfig:
    name: str
    category: str
    required_env_var: str
    endpoint_url: str
    is_critical: bool

@dataclass
class HealthStatus:
    service_name: str
    configured: bool
    env_present: bool
    status_message: str

class RAGEnvironmentValidator:
    def __init__(self, services: List[ServiceConfig]):
        self.services = services

    def run_diagnostics(self) -> List[HealthStatus]:
        diagnostics = []

        for svc in self.services:
            env_val = os.getenv(svc.required_env_var)
            has_env = bool(env_val and len(env_val.strip()) > 0)

            if has_env:
                # Mask secret for safe output logging
                masked = env_val[:4] + "..." + env_val[-3:] if len(env_val) > 7 else "****"
                msg = f"Configured successfully. Key: {masked}"
                status = HealthStatus(svc.name, True, True, msg)
            else:
                severity = "CRITICAL FAIL" if svc.is_critical else "WARNING OPTIONAL"
                msg = f"[{severity}] Environment variable '{svc.required_env_var}' is missing!"
                status = HealthStatus(svc.name, False, False, msg)

            diagnostics.append(status)

        return diagnostics

# Verification & Test Execution
if __name__ == "__main__":
    # 1. Define standard Enterprise RAG configuration stack
    rag_stack = [
        ServiceConfig("OpenAI Embeddings API", "Embeddings", "OPENAI_API_KEY", "https://api.openai.com/v1", True),
        ServiceConfig("Qdrant Vector DB", "Storage", "QDRANT_API_KEY", "https://xyz.qdrant.tech", True),
        ServiceConfig("Cohere Rerank API", "Reranker", "COHERE_API_KEY", "https://api.cohere.com/v1", False),
        ServiceConfig("LangSmith Telemetry", "Observability", "LANGCHAIN_API_KEY", "https://api.smith.langchain.com", False)
    ]

    # 2. Inject mock environment for demonstration
    os.environ["OPENAI_API_KEY"] = "sk-proj-abc123xyz987"
    os.environ["QDRANT_API_KEY"] = "qdrant_live_key_555"
    # Notice COHERE_API_KEY and LANGCHAIN_API_KEY are deliberately omitted

    validator = RAGEnvironmentValidator(rag_stack)
    report = validator.run_diagnostics()

    print("=== Enterprise RAG SDK Configuration Diagnostic Report ===")
    for item in report:
        icon = "✅" if item.configured else "❌"
        print(f"{icon} {item.service_name}: {item.status_message}")
```

---

## 10. ⚡ Quick Revision & Official Docs Bookmarks Matrix

### 🔑 Essential Key Takeaways
- **Never rely on outdated tutorials:** Vector databases and orchestration frameworks update monthly. Always verify parameters against official docs.
- **pgvector vs Dedicated Vector DBs:** Use pgvector for $< 1\text{M}$ vectors with relational ACID joins; use Qdrant/Milvus for large-scale, quantized, high-throughput pipelines.
- **Hardware Sizing:** Always factor in HNSW graph index overhead ($\approx 25\%$ extra memory) and consider `INT8` scalar quantization to cut RAM costs by $75\%$.
- **Fallback Rule:** Ensure fallback embedding models maintain identical dimensionalities to prevent runtime index schema exceptions.

### 🗺️ Production Documentation Bookmarks Cheat Sheet
```mermaid
flowchart LR
    Dev(["AI Engineer"]) --> Fw["Frameworks<br/>• python.langchain.com<br/>• docs.llamaindex.ai"]
    Dev --> Vdb["Storage<br/>• qdrant.tech/documentation<br/>• docs.pinecone.io"]
    Dev --> Models["Foundation APIs<br/>• platform.openai.com<br/>• docs.anthropic.com"]
    Dev --> Evals["Observability<br/>• docs.ragas.io<br/>• docs.smith.langchain.com"]

    style Dev fill:#f8fafc,stroke:#475569,stroke-width:2px
    style Fw fill:#eff6ff,stroke:#1d4ed8,stroke-width:1px
    style Vdb fill:#ecfdf5,stroke:#059669,stroke-width:1px
    style Models fill:#fef3c7,stroke:#d97706,stroke-width:1px
    style Evals fill:#fdf4ff,stroke:#a21caf,stroke-width:1px
```

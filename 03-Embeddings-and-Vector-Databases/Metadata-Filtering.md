# 🎯 Metadata Filtering in Vector Databases: Single-Stage Pre-Filtering vs. Two-Stage Post-Filtering

## 📑 Table of Contents
1. [🌟 Fundamentals: The Need for Metadata Filtering in Enterprise RAG](#1--fundamentals-the-need-for-metadata-filtering-in-enterprise-rag)
2. [⚠️ The Three Filtering Paradigms & The Recall Collapse Failure](#2-️-the-three-filtering-paradigms--the-recall-collapse-failure)
3. [🔬 Mathematical Mechanics of Filter Selectivity](#3--mathematical-mechanics-of-filter-selectivity)
4. [🏗️ Engine Storage Architecture: Roaring Bitmaps & Inverted Payloads](#4-️-engine-storage-architecture-roaring-bitmaps--inverted-payloads)
5. [⚖️ Architectural Trade-Off Analysis](#5-️-architectural-trade-off-analysis)
6. [💻 Practical Implementation: Single-Stage Filtered Graph Engine](#6--practical-implementation-single-stage-filtered-graph-engine)
7. [🌍 Real-World Enterprise Scenarios: RBAC & Multi-Tenancy](#7--real-world-enterprise-scenarios-rbac--multi-tenancy)
8. [🛠️ Best Practices & ❌ Common Production Pitfalls](#8-️-best-practices---common-production-pitfalls)
9. [💼 Senior Interview Questions & System Design Scenarios](#9--senior-interview-questions--system-design-scenarios)
10. [🧪 Hands-On Mini-Project: Post-Filtering Collapse & Selectivity Profiler](#10--hands-on-mini-project-post-filtering-collapse--selectivity-profiler)
11. [⚡ Quick Revision & Architectural Decision Roadmap](#11--quick-revision--architectural-decision-roadmap)

---

## 1. 🌟 Fundamentals: The Need for Metadata Filtering in Enterprise RAG

In real-world enterprise applications, vector similarity is rarely evaluated in isolation. A semantically perfect document passage is completely useless—or actively dangerous—if it violates business constraints:

* **Security & Access Control (RBAC):** An employee in Marketing searching internal documentation must never retrieve confidential executive compensation files, regardless of cosine similarity.
* **Temporal Freshness:** A financial analyst querying quarterly revenue trends must restrict results to the current fiscal year (`year >= 2024`).
* **Multi-Tenant Isolation:** In SaaS architectures, queries must be strictly confined to the user's specific organization (`tenant_id = 'acme_corp'`).
* **Categorical Facets:** Restricting product search to specific price ranges, geographic regions, or inventory statuses.

In modern vector databases (e.g., Qdrant, Pinecone, Weaviate, Milvus), each record consists of a **Vector ID**, the **Dense Embedding**, and a structured **JSON Metadata Payload**:

```mermaid
flowchart LR
    Point["Vector Database Point"]
    Point --> ID["Point ID: 'doc_9481'"]
    Point --> Vec["Dense Vector: [0.12, -0.45, ..., 0.88]"]
    Point --> Payload["Metadata Payload:<br/>{<br/>  'tenant_id': 'org_42',<br/>  'department': 'engineering',<br/>  'security_clearance': 'tier_3',<br/>  'created_at': 1715421000<br/>}"]
```

The fundamental architectural challenge is **how to execute approximate nearest neighbor (ANN) graph traversal while strictly satisfying arbitrary boolean metadata filter predicates**.

---

## 2. ⚠️ The Three Filtering Paradigms & The Recall Collapse Failure

There are three primary architectural approaches to combining metadata filters with vector search:

```mermaid
flowchart TD
    subgraph PostFiltering [1. Two-Stage Post-Filtering]
        Q1["Search Query + Filter"] --> ANN1["Standard Unconstrained HNSW Search (Top-100)"]
        ANN1 --> Discard["Filter Out Mismatched Documents"]
        Discard --> Result1["⚠️ HIGH RISK: Recall Collapse if filter is selective!"]
    end

    subgraph NaivePreFiltering [2. Naive Pre-Filtering]
        Q2["Search Query + Filter"] --> RelFilter["Filter Database using B-Tree / Inverted Index"]
        RelFilter --> Scan["Scan Matching Subset via Brute-Force Vector Scan"]
        Scan --> Result2["⚠️ HIGH RISK: Linear latency if filter matches 500k docs!"]
    end

    subgraph SingleStageHNSW [3. Single-Stage Pre-Filtered HNSW]
        Q3["Search Query + Filter"] --> Mask["Build In-Memory Filter Bitmask (Roaring Bitmap)"]
        Mask --> HNSWTraverse["Traverse HNSW Graph:<br/>Ineligible nodes act as routing bridges;<br/>Only eligible nodes enter candidate heap!"]
        HNSWTraverse --> Result3["✅ Production Standard: Guaranteed Top-K & Sub-10ms Latency"]
    end
```

### 1. Two-Stage Post-Filtering (The Flawed Legacy Method)
1. The database executes an unconstrained ANN search across the entire graph to retrieve the top $K_{\text{fetch}}$ candidates (e.g., $K_{\text{fetch}} = 100$).
2. The candidates are inspected sequentially, and any document failing the metadata filter is discarded.
3. **The Catastrophic Failure: "Recall Collapse"**  
   If the metadata filter is highly selective (e.g., matching only $0.5\%$ of the total corpus), an unconstrained Top-100 vector search will almost certainly contain **zero matching documents**. The query returns an empty result set, even though thousands of highly relevant documents exist in the database!

### 2. Naive Pre-Filtering (The Brute-Force Subset Method)
1. The database scans its secondary metadata indices (B-Trees or inverted lists) to locate all document IDs matching the filter.
2. The database computes exact vector distances across this candidate subset.
3. **The Bottleneck:** If the filter matches $40\%$ of a 5-million vector database, the engine is forced to compute **2,000,000 exact distance dot products**, collapsing query latency from $5 \text{ ms}$ to over $500 \text{ ms}$.

### 3. Single-Stage Pre-Filtering (Iterative Graph Traversal with Masking)
The modern industry standard pioneered by **Qdrant** and modern vector engines:
1. The engine evaluates the metadata filter first, producing a compressed **Boolean Bitmask (Roaring Bitmap)** where bit $i = 1$ if document $i$ matches the filter.
2. The engine navigates the HNSW graph using this bitmask.
3. **The Key Innovation (Bridge Traversal):** Nodes that do *not* match the filter are **not discarded during graph routing**. They can still be traversed as geometric "stepping stones" (bridges) to reach distant clusters. However, **only nodes with bit $i = 1$ are eligible to enter the final Top-$K$ candidate return heap**.

---

## 3. 🔬 Mathematical Mechanics of Filter Selectivity

The performance and accuracy of vector filtering depend strictly on **Filter Selectivity** $\sigma$:

### # Filter Selectivity Ratio Formula
$$
\sigma = \frac{|\mathcal{D}_{\text{matching}}|}{|\mathcal{D}_{\text{total}}|}, \quad \text{where } \sigma \in (0, 1]
$$

```mermaid
flowchart LR
    LowSel["Low Selectivity (σ < 0.01)<br/>Matches < 1% of corpus<br/>(e.g., tenant_id = 'user_891')"]
    HighSel["High Selectivity (σ > 0.30)<br/>Matches > 30% of corpus<br/>(e.g., status = 'active')"]

    LowSel --> Opt1["Optimizer Strategy: Exact Subset Brute-Force Scan<br/>(Only 500 vectors; exact scan takes 0.2 ms!)"]
    HighSel --> Opt2["Optimizer Strategy: Single-Stage Filtered HNSW Graph<br/>(Dense connections allow rapid graph navigation)"]
```

### 🧮 Expected Yield in Post-Filtering
In two-stage post-filtering, the expected number of returned documents when fetching $K_{\text{fetch}}$ candidates is:

### # Post-Filtering Expected Yield Formula
$$
\mathbb{E}[\text{Matching Candidates}] = K_{\text{fetch}} \cdot \sigma
$$

To guarantee retrieving $K$ valid documents with probability $1 - \delta$, the database must over-fetch:

### # Required Over-Fetching Factor
$$
K_{\text{fetch}} \ge \frac{K + \sqrt{2K \ln(1/\delta)}}{\sigma}
$$

When $\sigma = 0.001$ ($0.1\%$ selectivity) and $K = 10$, $K_{\text{fetch}} \ge 10,000$! The database would have to evaluate tens of thousands of vector neighbors, completely destroying latency and memory buffers.

---

## 4. 🏗️ Engine Storage Architecture: Roaring Bitmaps & Inverted Payloads

To enable sub-millisecond filtering across hundreds of millions of points, vector databases index metadata payloads using **Roaring Bitmaps**.

```mermaid
flowchart TD
    subgraph PayloadIndices [Inverted Metadata Storage]
        FieldDept["Index: 'department'"]
        FieldYear["Index: 'year'"]
    end

    FieldDept -->|"Key: 'Engineering'"| BM1["Roaring Bitmap 1: [Bits: 1, 4, 9, 12, ...]"]
    FieldYear -->|"Key: '2024'"| BM2["Roaring Bitmap 2: [Bits: 4, 9, 15, 22, ...]"]

    BM1 & BM2 --> BitwiseAND["Hardware Vectorized SIMD Bitwise AND (∩)"]
    BitwiseAND --> UnifiedMask["Combined Execution Mask: [Bits: 4, 9, ...]"]
```

### ⚡ Why Roaring Bitmaps?
1. **Dynamic Chunk Partitioning:** Roaring Bitmaps split 32-bit integer IDs into chunks of $2^{16}$ (65,536 integers). Each chunk is stored as:
   * An **Array Container** (if $< 4,096$ IDs): Compact sorted array.
   * A **Bitset Container** (if $\ge 4,096$ IDs): 8 KB uncompressed bitset.
   * A **Run-Length Container**: Optimized for contiguous sequences.
2. **SIMD Hardware Acceleration:** Intersection (`AND`), union (`OR`), and difference (`AND NOT`) operations evaluate across 512 bits per CPU clock cycle using AVX-512 vector instructions.
3. Combining filters like `tenant_id == 'acme' AND role == 'admin' AND year >= 2023` takes **under 50 microseconds** over 10 million records.

---

## 5. ⚖️ Architectural Trade-Off Analysis

| Filtering Paradigm | Recall Guarantee | Latency at Low Selectivity ($\sigma < 0.1\%$) | Latency at High Selectivity ($\sigma > 50\%$) | RAM Footprint | Complexity |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Two-Stage Post-Filtering** | ❌ **Extremely Poor (Recall Collapse)** | Instant Failure (Returns 0 hits) | Fast ($5 - 12 \text{ ms}$) | Low | Very Low |
| **Naive Pre-Filtering** | ✅ 100% Exact | Fast (Small candidate scan) | ❌ **Terrible ($200 - 800 \text{ ms}$)** | Low | Low |
| **Single-Stage Filtered HNSW**| ✅ **High ($98\% - 99.5\%$)** | Fast ($4 - 15 \text{ ms}$) | **Ultra-Fast ($3 - 8 \text{ ms}$)** | Moderate (Payload index overhead) | High (Complex graph logic) |
| **Cost-Based Query Planner** | ✅ **Maximum ($99\%+$)** | **Ultra-Fast ($1 - 5 \text{ ms}$)** (Switches to exact subset) | **Ultra-Fast ($3 - 8 \text{ ms}$)** (Switches to filtered graph) | Moderate | Very High (Qdrant / Milvus engine) |

---

## 6. 💻 Practical Implementation: Single-Stage Filtered Graph Engine

The following standalone Python module implements both **Post-Filtering** and **Single-Stage Filtered Search** over a synthetic vector graph, demonstrating how single-stage filtering eliminates recall collapse.

```python
"""
Production Architectural Blueprint: Single-Stage Filtered Vector Graph Engine
Author: Enterprise RAG Curriculum
Requirements: pip install numpy tabulate
"""

from typing import Any, Callable, Dict, List, Set, Tuple
import numpy as np
from tabulate import tabulate

class FilteredVectorEngine:
    """Demonstrates Single-Stage Graph Filtering vs Post-Filtering Failure."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors: Dict[int, np.ndarray] = {}
        self.payloads: Dict[int, Dict[str, Any]] = {}
        # Graph adjacency: node_id -> list of neighbor IDs
        self.graph: Dict[int, List[int]] = {}

    def insert(self, point_id: int, vector: List[float], payload: Dict[str, Any]) -> None:
        v = np.array(vector, dtype=np.float32)
        v /= np.linalg.norm(v)
        self.vectors[point_id] = v
        self.payloads[point_id] = payload
        self.graph[point_id] = []

    def build_knn_graph(self, k_neighbors: int = 8) -> None:
        """Constructs a basic connected navigation graph."""
        point_ids = list(self.vectors.keys())
        matrix = np.array([self.vectors[pid] for pid in point_ids])
        # Compute all-pairs cosine similarity
        sims = np.dot(matrix, matrix.T)
        
        for i, pid in enumerate(point_ids):
            # Select top k nearest neighbors excluding self
            sorted_indices = np.argsort(-sims[i])
            neighbors = [point_ids[idx] for idx in sorted_indices if point_ids[idx] != pid][:k_neighbors]
            self.graph[pid] = neighbors

    # --- Flawed Approach: Post-Filtering ---
    def post_filtered_search(
        self, query_vec: List[float], filter_fn: Callable[[Dict[str, Any]], bool], top_k: int = 5, fetch_k: int = 20
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """Step 1: Retrieve top fetch_k unconstrained. Step 2: Discard non-matching."""
        qv = np.array(query_vec, dtype=np.float32)
        qv /= np.linalg.norm(qv)

        # Unconstrained score over all points
        all_hits = []
        for pid, vec in self.vectors.items():
            all_hits.append((pid, float(np.dot(qv, vec))))
        all_hits.sort(key=lambda x: x[1], reverse=True)
        
        top_unconstrained = all_hits[:fetch_k]

        # Filter post-retrieval
        passed = []
        for pid, score in top_unconstrained:
            meta = self.payloads[pid]
            if filter_fn(meta):
                passed.append((pid, score, meta))
                if len(passed) == top_k:
                    break
        return passed

    # --- Robust Approach: Single-Stage Filtered Graph Traversal ---
    def single_stage_filtered_search(
        self, query_vec: List[float], filter_fn: Callable[[Dict[str, Any]], bool], top_k: int = 5, ef: int = 30
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """Navigates graph: ineligibles act as bridges, only eligibles enter return heap."""
        qv = np.array(query_vec, dtype=np.float32)
        qv /= np.linalg.norm(qv)

        entry_id = list(self.vectors.keys())[0]
        visited: Set[int] = {entry_id}
        candidate_queue = [(entry_id, float(1.0 - np.dot(qv, self.vectors[entry_id])))]
        
        # Valid matching results heap
        eligible_matches: List[Tuple[int, float, Dict[str, Any]]] = []

        while candidate_queue:
            candidate_queue.sort(key=lambda x: x[1])
            curr_id, curr_dist = candidate_queue.pop(0)

            # Check if current node is eligible for return
            meta = self.payloads[curr_id]
            if filter_fn(meta):
                sim = float(np.dot(qv, self.vectors[curr_id]))
                eligible_matches.append((curr_id, sim, meta))
                eligible_matches.sort(key=lambda x: x[1], reverse=True)
                if len(eligible_matches) > top_k:
                    eligible_matches.pop()

            # Explore graph neighbors (Both eligible AND ineligible nodes are traversed!)
            for neighbor_id in self.graph.get(curr_id, []):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    n_dist = float(1.0 - np.dot(qv, self.vectors[neighbor_id]))
                    candidate_queue.append((neighbor_id, n_dist))
                    
            if len(visited) >= ef:
                break

        return eligible_matches[:top_k]


# Demonstration Execution
if __name__ == "__main__":
    np.random.seed(42)
    engine = FilteredVectorEngine(dimension=16)

    # Insert 1,000 points. Only 1% belong to tenant 'security_alpha'
    print("[*] Ingesting 1,000 vectors with 1% filter selectivity (10 matching docs)...")
    for i in range(1000):
        # Cluster matching docs slightly in a sub-dimension to simulate semantic correlation
        is_alpha = (i % 100 == 0)  # Exactly 10 documents
        v = np.random.randn(16)
        if is_alpha:
            v[0] += 2.5  # Distinct feature
        engine.insert(
            point_id=i, 
            vector=v.tolist(), 
            payload={"tenant": "security_alpha" if is_alpha else "general_public"}
        )

    engine.build_knn_graph(k_neighbors=10)

    # Target query searching for 'security_alpha' content
    query_vector = np.random.randn(16)
    query_vector[0] += 2.5
    
    alpha_filter = lambda p: p.get("tenant") == "security_alpha"

    # 1. Post-Filtering Search (fetch_k = 20)
    post_results = engine.post_filtered_search(query_vector.tolist(), alpha_filter, top_k=5, fetch_k=20)

    # 2. Single-Stage Filtered Search
    single_results = engine.single_stage_filtered_search(query_vector.tolist(), alpha_filter, top_k=5, ef=60)

    print("\n" + tabulate([
        ["Post-Filtering (fetch_k=20)", f"{len(post_results)} returned", "❌ RECALL COLLAPSE" if len(post_results) < 5 else "OK"],
        ["Single-Stage Filtered HNSW", f"{len(single_results)} returned", "✅ COMPLETE TOP-K FOUND"]
    ], headers=["Strategy", "Results Retrieved (Target=5)", "Health Status"], tablefmt="github"))
```

---

## 7. 🌍 Real-World Enterprise Scenarios: RBAC & Multi-Tenancy

```mermaid
flowchart TD
    UserQuery["User Request: 'Financial Q3 Statements'"]
    AuthContext["Auth Context: User=Bob, Roles=['compliance', 'auditor']"]
    
    UserQuery & AuthContext --> QueryGateway["API Security Gateway"]
    
    QueryGateway --> FilterSynthesis["Synthesize AST Filter:<br/>tenant_id == 'org_9' AND<br/>(acl_roles IN ['compliance', 'auditor'] OR is_public == true)"]
    
    FilterSynthesis --> VectorEngine["Vector DB (Single-Stage Filtered Engine)"]
    VectorEngine --> SafePassages["Returned Grounded Passages<br/>(100% Cryptographically & Logically Compliant)"]
```

### Case Study: Global Financial Accounting Multi-Tenant SaaS
* **Setup:** A multi-tenant tax analysis platform hosting 12,000 corporate clients on a shared cluster of 80 million vector chunks.
* **Security Mandate:** Strict tenant boundaries. If a query leaks even a single chunk from another corporate tenant, the company faces immediate regulatory revocation.
* **Failure of Post-Filtering:** When small tenants (accounting for $<0.01\%$ of vectors) searched their own documents, post-filtering returned 0 results $94\%$ of the time due to recall collapse.
* **Architectural Solution:**
  1. Enforced a mandatory `tenant_id` filter injected at the API Gateway layer (preventing prompt injection tampering).
  2. Implemented **Cost-Based Query Optimization**: If a tenant has fewer than 2,000 chunks, the database skips graph traversal and executes an exact brute-force vector scan across their isolated Roaring Bitmap mask in $<1.5 \text{ ms}$.
  3. For large enterprise tenants ($>500,000$ chunks), the engine executes Single-Stage Filtered HNSW.
* **Outcome:** Zero multi-tenant data leakage; $100\%$ retrieval recall across all tenant tiers.

---

## 8. 🛠️ Best Practices & ❌ Common Production Pitfalls

### ✅ Best Practices
1. **Always Enforce Hard Constraints at the Query Gateway:** Never trust user prompt strings to specify filters. Extract identity, organization ID, and role permissions cryptographically from authenticated JWT session tokens.
2. **Build Explicit Payload Indices on High-Selectivity Fields:** In Qdrant or Weaviate, metadata fields are unindexed by default. Always create payload schema indices on `tenant_id`, `status`, and `timestamp` fields.
3. **Use Roaring Bitmaps for High-Cardinality Enums:** For tags, categories, and role memberships, bitmap indices provide $50\times$ faster set evaluation than relational B-Trees.

### ❌ Common Production Pitfalls
* **Creating Payload Indices on High-Entropy Unique Fields:** Building a payload index on a unique document hash or user email string wastes RAM without providing filtering utility.
* **The "Disconnected Island" Problem in Filtered HNSW:** If an aggressive filter eliminates $99.9\%$ of nodes, the remaining eligible nodes may not have direct edges between them, trapping search in local minima. (Modern engines solve this by using ineligible nodes as bridges or falling back to flat subset scan).
* **Failing to Sanitize Temporal Ranges:** Running open-ended range filters (e.g., `timestamp > 0`) forces the engine to build a full bitmask over the entire database unnecessarily.

---

## 9. 💼 Senior Interview Questions & System Design Scenarios

### Q1: What is the "Recall Collapse" problem in vector database post-filtering, and what mathematical condition causes it?
> **Expert Answer:**  
> Recall Collapse occurs when a vector database attempts to combine metadata filtering with ANN search using a two-stage pipeline: first executing unconstrained nearest neighbor retrieval for $K_{\text{fetch}}$ items, and then discarding candidates that fail the metadata predicate.  
> Mathematically, let $\sigma = \frac{|\mathcal{D}_{\text{filter}}|}{|\mathcal{D}_{\text{total}}|}$ represent filter selectivity. The expected number of valid documents present in the unconstrained top-$K_{\text{fetch}}$ candidate pool is $\mathbb{E}[K_{\text{valid}}] = K_{\text{fetch}} \cdot \sigma$.  
> When $\sigma \ll 1$ (e.g., $\sigma = 0.002$ in a multi-tenant or filtered dataset), retrieving $K_{\text{fetch}} = 100$ candidates yields an expected matching count of $\mathbb{E}[K_{\text{valid}}] = 100 \times 0.002 = 0.2$ documents. Consequently, with $>80\%$ probability, the search returns **zero results**, collapsing retrieval recall to 0% even when hundreds of highly relevant matching documents exist in the corpus.

### Q2: How does Single-Stage Filtered HNSW avoid getting stuck in local minima when 95% of the graph nodes are filtered out?
> **Expert Answer:**  
> If an engine simply deleted or ignored ineligible nodes during graph traversal, the HNSW graph would fragment into isolated, disconnected sub-graphs. Search entering at the top layer would hit dead ends where all local neighbors are invalid, causing premature termination.  
> Single-Stage Filtered HNSW overcomes this by **decoupling graph navigation from candidate selection**:
> 1. Ineligible nodes are treated as **navigational bridges**: the algorithm computes distance to them and traverses their edges to continue navigating across Euclidean space toward the target neighborhood.
> 2. However, ineligible nodes are **strictly masked from the candidate priority queue**: only nodes passing the filter bitmask are pushed to the Top-$K$ output heap.  
> 3. If selectivity is extremely low ($\sigma < 0.001$), the query optimizer detects that the graph is too sparse for efficient traversal and dynamically switches to a flat brute-force scan over the pre-filtered ID list.

### Q3: How do modern vector databases optimize complex compound filters (AND, OR, NOT) over millions of documents?
> **Expert Answer:**  
> Modern vector databases utilize **Roaring Bitmaps** backed by hardware SIMD vectorization.  
> When a query includes a compound predicate like:
> `(category == 'legal' OR category == 'compliance') AND year >= 2023 AND NOT status == 'archived'`
> 1. The payload index retrieves or calculates the bitmap container for each clause.
> 2. The database evaluates bitwise operations (`OR`, `AND`, `AND NOT`) across these containers using 512-bit AVX-512 instructions, processing 512 boolean values per CPU cycle.
> 3. The resulting unified bitmask serves as an $O(1)$ constant-time lookup during graph traversal: checking whether node $i$ is eligible requires a single bit-test operation (`test_bit(mask, i)`), adding virtually zero latency to graph navigation hops.

---

## 10. 🧪 Hands-On Mini-Project: Post-Filtering Collapse & Selectivity Profiler

This self-contained Python script benchmarks how varying **Filter Selectivity ($\sigma$)** causes catastrophic recall failure in post-filtering, proving why single-stage filtering is essential.

```python
"""
Hands-On Diagnostic: Filter Selectivity vs. Post-Filtering Recall Collapse
Run directly: python selectivity_profiler.py
Requires: pip install tabulate
"""

import random
from typing import List, Tuple
from tabulate import tabulate

def simulate_selectivity_collapse():
    random.seed(42)
    corpus_size = 50000
    top_k_desired = 10
    fetch_k = 100  # Standard post-filtering candidate pool
    
    # Selectivity rates to evaluate: 50%, 10%, 1%, 0.1%, 0.01%
    selectivities = [0.50, 0.10, 0.05, 0.01, 0.001]
    
    results = []

    for sigma in selectivities:
        matching_count = int(corpus_size * sigma)
        # Create ground truth matching pool
        matching_ids = set(random.sample(range(corpus_size), matching_count))
        
        # Simulate 100 queries: Assume true relevant documents are distributed in top 500
        total_retrieved = 0
        zero_hit_queries = 0
        
        for _ in range(100):
            # Simulated top-100 unconstrained candidates from ANN search
            # Higher probability of matching items if topically correlated
            candidate_pool = random.sample(range(corpus_size), fetch_k)
            
            # Post-filter
            passed = [cid for cid in candidate_pool if cid in matching_ids]
            returned_count = min(top_k_desired, len(passed))
            total_retrieved += returned_count
            if returned_count == 0:
                zero_hit_queries += 1
                
        avg_retrieved = total_retrieved / 100.0
        recall_rate = (avg_retrieved / top_k_desired) * 100.0
        
        results.append([
            f"{sigma * 100:.2f}%",
            matching_count,
            f"{avg_retrieved:.1f} / {top_k_desired}",
            f"{zero_hit_queries}%",
            f"{recall_rate:.1f}%",
            "CRITICAL COLLAPSE" if recall_rate < 50 else ("DEGRADED" if recall_rate < 90 else "PASS")
        ])

    print("--- Post-Filtering Recall Collapse Across Selectivity Spectrum (fetch_k=100, target=10) ---")
    print(tabulate(
        results,
        headers=["Selectivity (σ)", "Matching Docs in DB", "Avg Returned", "Zero-Result Queries", "Effective Recall", "Verdict"],
        tablefmt="github"
    ))

if __name__ == "__main__":
    simulate_selectivity_collapse()
```

---

## 11. ⚡ Quick Revision & Architectural Decision Roadmap

### 📋 Key Takeaways Cheat-Sheet
* **Metadata is Essential for Enterprise RAG:** Enforces multi-tenancy, RBAC security, and temporal freshness filters.
* **Post-Filtering Suffers from Recall Collapse:** When selectivity is low ($\sigma < 1\%$), unconstrained top-100 search returns zero matching documents.
* **Naive Pre-Filtering Degenerates on High Selectivity:** Scanning millions of matching records via brute-force destroys search latency.
* **Single-Stage Filtered HNSW is the Gold Standard:** Uses compressed Roaring Bitmaps during graph traversal; ineligible nodes act as navigational bridges, while only eligible nodes enter the candidate return heap.
* **Cost-Based Query Planners Provide Safety:** Vector engines dynamically switch between exact subset scan (at low $\sigma$) and single-stage filtered graph search (at high $\sigma$).

### 🧭 Visual Decision Tree: Selecting Filter Execution Strategy

```mermaid
flowchart TD
    Start["Incoming Query with Metadata Filter"] --> Q1{"Calculate Filter Selectivity (σ)"}
    
    Q1 -->|"Extremely Low (σ < 0.1% or < 1,000 docs)"| PlanA["Execute Exact Brute-Force Scan over Filtered ID Subset<br/>(Sub-2ms execution, 100% exact recall)"]
    
    Q1 -->|"Moderate to High (σ ≥ 0.1%)"| PlanB["Execute Single-Stage Filtered HNSW Graph Traversal<br/>(Use Roaring Bitmask, traverse ineligibles as bridges)"]
    
    Q1 -->|"No Metadata Filter Present"| PlanC["Execute Pure Unconstrained HNSW Traversal"]
```

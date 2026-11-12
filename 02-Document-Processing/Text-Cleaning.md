# 🧹 Text Cleaning, Normalization, Unicode Sanitization, and Noise Reduction for RAG

## 📑 Table of Contents
1. [🌟 Fundamentals: The "Garbage In, Garbage Out" Law of Embeddings](#1--fundamentals-the-garbage-in-garbage-out-law-of-embeddings)
2. [🔤 Unicode Sanitization & Normalization Forms (NFC vs. NFKC)](#2--unicode-sanitization--normalization-forms-nfc-vs-nfkc)
3. [🧹 Whitespace Normalization & Structural Noise Reduction](#3--whitespace-normalization--structural-noise-reduction)
4. [🧮 Mathematical Formalisms & Information-Theoretic Noise Metrics](#4--mathematical-formalisms--information-theoretic-noise-metrics)
5. [⚙️ Normalization & Cleaning Rules Matrix](#5-️-normalization--cleaning-rules-matrix)
6. [💻 Production Implementation: Enterprise Text Sanitizer](#6--production-implementation-enterprise-text-sanitizer)
7. [🛡️ Security Sanitization: Defending Vector Stores & LLMs](#7-️-security-sanitization-defending-vector-stores--llms)
8. [🌍 Real-World Enterprise Case Studies](#8--real-world-enterprise-case-studies)
9. [✅ Advantages & ⚠️ Limitations of Aggressive Cleaning](#9--advantages--️-limitations-of-aggressive-cleaning)
10. [🛠️ Best Practices & ❌ Common Traps in Production](#10-️-best-practices---common-traps-in-production)
11. [💼 Senior Interview Preparation: High-Yield Questions](#11--senior-interview-preparation-high-yield-questions)
12. [🧪 Hands-On Mini-Project: Shannon Entropy OCR Noise Filter](#12--hands-on-mini-project-shannon-entropy-ocr-noise-filter)
13. [⚡ Quick Revision & Summary Roadmap](#13--quick-revision--summary-roadmap)

---

## 1. 🌟 Fundamentals: The "Garbage In, Garbage Out" Law of Embeddings

In traditional Natural Language Processing (NLP) circa 2012, preprocessing meant aggressive text stripping: lowercasing all words, removing punctuation, dropping stop-words, and stemming words down to morphological roots (`running` $\to$ `run`).

In modern **Retrieval-Augmented Generation**, applying those legacy techniques is catastrophic. Transformer embedding models (OpenAI `text-embedding-3`, BGE, Cohere, Voyage) and Large Language Models are **context-dependent, case-sensitive neural networks**. Indiscriminate lowercasing transforms `"US"` (United States) into `"us"` (pronoun), destroying semantic distinction.

However, uncleaned text contains invisible formatting artifacts that degrade vector retrieval:
* **Inconsistent Unicode encodings:** Visual duplicates having different byte sequences.
* **Invisible control characters:** Zero-width spaces (`\u200b`) and soft hyphens (`\u00ad`) that fracture subword tokenization.
* **Repetitive layout noise:** Running headers, footers, and page numbers that pollute vector similarity.
* **Null bytes (`\x00`):** Binary artifacts that crash vector databases like PostgreSQL (`pgvector`).

```mermaid
flowchart LR
    Raw["Raw Extracted Text<br/>• Mixed Unicode<br/>• \u200b Zero-Width Spaces<br/>• Broken hyphen-\nations<br/>• Running Headers"] --> Pipe["Sanitization Pipeline<br/>• NFKC Normalization<br/>• Control Char Stripping<br/>• De-hyphenation<br/>• Header De-duplication"]
    Pipe --> Clean["High-Fidelity Corpus<br/>• Canonical Unicode<br/>• Intact Subword Tokens<br/>• Clean Markdown Layout"]
    Clean --> Embed["Dense Embedding Model"]
```

---

## 2. 🔤 Unicode Sanitization & Normalization Forms (NFC vs. NFKC)

Unicode allows multiple distinct character sequences to represent the exact same visual glyph. This introduces the **Canonical Equivalence** problem.

For example, the character **é** can be represented as:
1. A single precomposed code point: `\u00e9` (Latin Small Letter E with Acute).
2. A decomposed sequence: `\u0065` (Letter E) + `\u0301` (Combining Acute Accent).

Subword tokenizers (like Byte-Pair Encoding) treat these two encodings as completely different token sequences, yielding divergent vector embeddings for identical words!

```mermaid
flowchart TD
    subgraph DecomposedInput [Decomposed Unicode]
        D1["'e' (\u0065) + '\u0301' (\u0301)"]
    end

    subgraph PrecomposedInput [Precomposed Unicode]
        P1["'é' (\u00e9)"]
    end

    subgraph NFKCTransformation [NFKC Normalization]
        Norm["unicodedata.normalize('NFKC', text)"]
    end

    subgraph CanonicalOutput [Single Canonical Token]
        C1["'é' (\u00e9)"]
    end

    D1 --> Norm
    P1 --> Norm
    Norm --> C1
```

### 🗄️ Unicode Normalization Forms

| Form | Name | Mechanics | Compatibility Handling | Best Fit for RAG? |
| :--- | :--- | :--- | :--- | :---: |
| **NFC** | Canonical Decomposition + Canonical Composition | Decomposes characters then recomposes them to precomposed forms | Preserves typographic ligatures (`ﬁ`, `™`, `²`) | ⚠️ Partial |
| **NFD** | Canonical Decomposition | Breaks all composite glyphs into base characters + combining marks | Preserves typographic ligatures | ❌ No |
| **NFKC** | Compatibility Decomposition + Canonical Composition | Decomposes compatibility characters and recomposes to canonical equivalents | **Expands ligatures (`ﬁ` $\to$ `fi`), replaces superscripts (`²` $\to$ `2`), normalizes symbols** | ✅ **Gold Standard** |
| **NFKD** | Compatibility Decomposition | Decomposes both canonical and compatibility characters | Completely splits all accents and symbols | ❌ No |

---

## 3. 🧹 Whitespace Normalization & Structural Noise Reduction

### 1. Zero-Width & Non-Breaking Character Removal
PDFs and web pages are littered with non-printing Unicode characters used for visual justification. These characters must be scrubbed:
* Zero-width space: `\u200b`
* Zero-width non-joiner: `\u200c`
* Zero-width joiner: `\u200d`
* Non-breaking space: `\u00a0` $\to$ standard space ` `

### 2. Hyphenation Repair
At line wraps, words are often split with a soft hyphen (`\u00ad`) or hard hyphen:
```text
The retrieval-aug-
mented generation system...
```
A regex normalization replaces end-of-line hyphens when followed by lowercase characters, merging them back into single semantic units (`retrieval-augmented`).

### 3. Repetitive Running Header/Footer Detection
If a 200-page document repeats `"ACME CORP INTERNAL - CONFIDENTIAL - PAGE N"` at the top of every page, the vector retriever will match this boilerplate for queries regarding "internal policies".
* **Algorithmic Fix:** Hash or compute string similarity across the first and last two lines of all pages; if identical lines appear on $>20\%$ of pages, prune them automatically.

---

## 4. 🧮 Mathematical Formalisms & Information-Theoretic Noise Metrics

### # Shannon Entropy for OCR Noise Detection
Low-quality OCR scans often produce scrambled strings of unreadable punctuation and characters (e.g., `^%#$$@!a1~`). We detect and discard corrupted text blocks using **Character-Level Shannon Entropy** $H(X)$:

$$
H(X) = -\sum_{i=1}^V P(c_i) \log_2 P(c_i)
$$

Where:
* $V$ is the alphabet of distinct characters appearing in the text string.
* $P(c_i) = \frac{\text{Count}(c_i)}{N}$ is the empirical probability of character $c_i$ in the string of length $N$.

Standard natural language English text exhibits an entropy range of $3.5 \le H(X) \le 4.8$. Blocks with $H(X) > 5.5$ (random character noise) or $H(X) < 1.5$ (repetitive punctuation like `........`) are automatically dropped.

### # Jaccard Similarity for Running Header Pruning
To identify repetitive running headers across pages $P_A$ and $P_B$ despite varying page numbers, we compute the **Jaccard Similarity** $J(S_A, S_B)$ over token sets:

$$
J(S_A, S_B) = \frac{|S_A \cap S_B|}{|S_A \cup S_B|} = \frac{|S_A \cap S_B|}{|S_A| + |S_B| - |S_A \cap S_B|}
$$

When comparing top lines of adjacent pages, a similarity $J(S_A, S_B) > 0.80$ identifies repetitive running header patterns.

---

## 5. ⚙️ Normalization & Cleaning Rules Matrix

### 📊 Text Cleaning Rule Pipeline

| Pipeline Stage | Target Pattern | Transformation Rule | Impact on Embeddings | Risk of Over-Cleaning |
| :--- | :--- | :--- | :--- | :--- |
| **1. Unicode Normalization** | Mixed precomposed/decomposed glyphs, typographic ligatures (`ﬁ`, `œ`) | Apply `unicodedata.normalize('NFKC', text)` | Guarantees identical token IDs for semantically identical terms | Zero (Preserves meaning) |
| **2. Control Character Purge** | Non-printable ASCII (`\x00-\x08`, `\x0B-\x0C`, `\x0E-\x1F`, `\x7F`) | Strip characters; keep `\n`, `\r`, `\t` | Prevents database insert failures in `pgvector` | Zero |
| **3. Zero-Width Stripping** | `\u200b`, `\u200c`, `\u200d`, `\ufeff` (BOM) | Replace with empty string `""` | Restores standard subword tokenizer boundaries | Zero |
| **4. De-Hyphenation** | End-of-line word splits (`word-\nnext`) | Replace with `wordnext` (if next is lowercase) | Prevents vocab fragmentation in tokenizer | Low (Preserve hard hyphens in `state-of-the-art`) |
| **5. Whitespace Collapse** | Consecutive tabs and spaces (`[ \t]{2,}`) | Replace with single space `' '` | Saves context tokens without altering syntax | Keep code block indentation intact |
| **6. Newline Normalization** | Excessive line breaks (`\n{3,}`) | Collapse to `\n\n` (paragraph separator) | Maintains paragraph boundaries cleanly | Zero |

---

## 6. 💻 Production Implementation: Enterprise Text Sanitizer

This production-grade Python 3.10+ class implements complete Unicode sanitization, de-hyphenation, null byte removal, and code-block preservation:

```python
"""
Production Enterprise Text Sanitizer for RAG Pipelines
Applies NFKC normalization, whitespace sanitization, and protects code blocks.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math
import re
import unicodedata
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TextSanitizer")


@dataclass
class CleaningMetrics:
    original_chars: int
    cleaned_chars: int
    shannon_entropy: float
    is_valid: bool


class EnterpriseTextSanitizer:
    def __init__(self, min_entropy: float = 2.0, max_entropy: float = 5.6):
        self.min_entropy = min_entropy
        self.max_entropy = max_entropy

        # Precompiled regex patterns
        self.re_null_bytes = re.compile(r"\x00")
        self.re_control_chars = re.compile(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]")
        self.re_zero_width = re.compile(r"[\u200b\u200c\u200d\u200e\u200f\ufeff]")
        self.re_soft_hyphens = re.compile(r"\u00ad")
        self.re_eol_hyphen = re.compile(r"(\b[a-zA-Z]+)-\n([a-zA-Z]+\b)")
        self.re_multispaces = re.compile(r"[^\S\r\n]{2,}")
        self.re_multinewlines = re.compile(r"\n{3,}")

    @staticmethod
    def calculate_shannon_entropy(text: str) -> float:
        """Calculates character-level Shannon entropy to detect OCR noise or corruption."""
        if not text:
            return 0.0

        length = len(text)
        freqs: Dict[str, int] = {}
        for char in text:
            freqs[char] = freqs.get(char, 0) + 1

        entropy = 0.0
        for count in freqs.values():
            p = count / length
            entropy -= p * math.log2(p)

        return round(entropy, 4)

    def _mask_code_blocks(self, text: str) -> Tuple[str, List[str]]:
        """Temporarily extracts and masks markdown code blocks to prevent ruining indentation."""
        code_blocks: List[str] = []

        def replacer(match: re.Match) -> str:
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_PLACEHOLDER_{len(code_blocks) - 1}__"

        masked_text = re.sub(r"```[\s\S]*?```", replacer, text)
        return masked_text, code_blocks

    def _unmask_code_blocks(self, text: str, code_blocks: List[str]) -> str:
        """Restores protected code blocks back into the cleaned text."""
        for idx, block in enumerate(code_blocks):
            text = text.replace(f"__CODE_BLOCK_PLACEHOLDER_{idx}__", block)
        return text

    def sanitize(self, raw_text: str) -> Tuple[str, CleaningMetrics]:
        orig_len = len(raw_text)

        if not raw_text or not raw_text.strip():
            return "", CleaningMetrics(orig_len, 0, 0.0, False)

        # 1. Mask code blocks
        masked_text, code_blocks = self._mask_code_blocks(raw_text)

        # 2. Unicode NFKC Normalization (expands ligatures, unifies character variants)
        text = unicodedata.normalize("NFKC", masked_text)

        # 3. Strip null and illegal control characters
        text = self.re_null_bytes.sub("", text)
        text = self.re_control_chars.sub("", text)
        text = self.re_zero_width.sub("", text)
        text = self.re_soft_hyphens.sub("", text)

        # 4. Fix split line-wrap hyphens
        text = self.re_eol_hyphen.sub(r"\1\2", text)

        # 5. Normalize whitespace outside code blocks
        text = self.re_multispaces.sub(" ", text)
        text = self.re_multinewlines.sub("\n\n", text)
        text = text.strip()

        # 6. Restore code blocks
        final_text = self._unmask_code_blocks(text, code_blocks)
        cleaned_len = len(final_text)

        # 7. Entropy validation
        entropy = self.calculate_shannon_entropy(final_text)
        is_valid = self.min_entropy <= entropy <= self.max_entropy and cleaned_len > 10

        metrics = CleaningMetrics(
            original_chars=orig_len,
            cleaned_chars=cleaned_len,
            shannon_entropy=entropy,
            is_valid=is_valid
        )

        return final_text, metrics


if __name__ == "__main__":
    sanitizer = EnterpriseTextSanitizer()

    dirty_sample = (
        "Enterprise\u00ad RAG sys\u200btems require\n\n\n\n"
        "retriev-\nal of canonical\u00a0data.\x00 "
        "Typographic ﬁxture test.\n\n"
        "```python\n"
        "def query():\n"
        "    # Indentation must remain intact\n"
        "    return 42\n"
        "```\n\n"
        "Noise footer: ........~~~~~~~~~~~~"
    )

    clean_output, stats = sanitizer.sanitize(dirty_sample)
    print("--- Cleaned Output ---")
    print(clean_output)
    print("\n--- Telemetry Metrics ---")
    print(stats)
```

---

## 7. 🛡️ Security Sanitization: Defending Vector Stores & LLMs

### 🚨 Null Byte Poisoning (`\x00`)
In PostgreSQL `pgvector`, raw strings containing `\x00` trigger an immediate database crash (`psycopg2.errors.UntranslatableCharacter`). Ensure every input passes through null-byte filtering.

### 🎭 Bidirectional Unicode Spoofing (Trojan Source)
Adversaries use Unicode directional override characters (`\u202e` Right-to-Left Override) to reverse the appearance of text on human screens while passing opposite instructions to LLMs:
```python
# Trojan Source characters must be strictly rejected:
BIDI_CHARS = ["\u202a", "\u202b", "\u202c", "\u202d", "\u202e"]
for char in BIDI_CHARS:
    text = text.replace(char, "")
```

---

## 8. 🌍 Real-World Enterprise Case Studies

### Case Study 1: Multilingual European Legal Discovery Platform
* **Problem:** Searching through millions of French and German contracts. Queries for `"crédit"` failed to match documents containing decomposed `"credit"` + combining accent `\u0301`, causing a $23\%$ retrieval failure rate.
* **Architecture:** Ingested all documents through an automated NFKC normalization filter. Re-indexed 10 million passages.
* **Result:** Zero-recall query errors dropped from $23\%$ to $0.02\%$.

### Case Study 2: Historical Digitization Project (OCR Garbage Filtering)
* **Problem:** Scraping 500,000 scanned archival medical records. Smudged pages produced lines of gibberish punctuation that clogged vector databases and caused LLMs to hallucinate bizarre answers.
* **Architecture:** Deployed a Shannon Entropy filter ($H(X) > 5.5$) that automatically routed high-entropy garbage lines to a triage discard queue.
* **Result:** Reduced index storage footprint by $14\%$ and eliminated ungrounded hallucinations on historical queries.

---

## 9. ✅ Advantages & ⚠️ Limitations of Aggressive Cleaning

### 📊 Comparative Analysis

| Preprocessing Approach | Advantages | Major Risks / Limitations | Recommended Context |
| :--- | :--- | :--- | :--- |
| **Minimal Sanitization**<br/>(NFKC + Null bytes only) | • Preserves $100\%$ of original casing, syntax, and punctuation<br/>• Zero risk of altering legal/code meaning | • Retains running headers and footers<br/>• Higher context token count | Legal discovery, source code repositories |
| **Aggressive Rule-Based**<br/>(Header pruning + Entropy filtering) | • Removes $15 - 30\%$ of token bloat<br/>• Dramatically improves dense retrieval recall | • Risk of stripping ASCII art, dense mathematical equations, or cryptic passwords | Enterprise documentation, knowledge bases, news |
| **Legacy NLP Stripping**<br/>(Stop-word removal + Lowercasing) | • Ultra-compact lexical indexes | • **Destroys Transformer attention**<br/>• Breaks syntax, acronyms, and semantic nuances | ❌ **Do NOT use in modern RAG** |

---

## 10. 🛠️ Best Practices & ❌ Common Traps in Production

### 🛠️ Production Best Practices
1. **Always Protect Code Blocks:** Mask markdown triple backtick blocks (````python...```) before regex whitespace normalization to avoid destroying syntax indentation.
2. **Apply NFKC Universally:** Always normalize both the indexing corpus and the user's runtime query with `unicodedata.normalize('NFKC', text)`.
3. **Preserve Case and Punctuation:** Modern embedding models rely heavily on question marks, capitalization, and syntax for semantic disambiguation.

### ❌ Common Traps to Avoid
* ❌ **The Blind Lowercase Trap:** Running `.lower()` on all text, converting `"IT"` (Information Technology) to `"it"` (pronoun).
* ❌ **Stripping Markdown Structural Markers:** Removing `#` or `|` characters, turning headers and Markdown tables into unstructured run-on text.
* ❌ **Deleting Necessary Hyphens:** Blindly replacing all hyphens with empty strings, turning `"high-performance"` into `"highperformance"`.

---

## 11. 💼 Senior Interview Preparation: High-Yield Questions

### Q1: Why is traditional NLP preprocessing (lowercasing, stop-word removal, stemming) considered an anti-pattern in modern RAG pipelines?
> **Answer:**
> * Modern RAG systems rely on **Transformer models** that compute contextualized self-attention across complete sentences.
> * **Stop-words provide syntactic structure:** Words like "not", "between", or "under" drastically alter the meaning of a clause. Stripping them causes the attention mechanism to lose relational context.
> * **Casing encodes named entity semantics:** Transformer models use casing to distinguish proper nouns and acronyms (`Apple` the company vs. `apple` the fruit; `US` the country vs. `us` the pronoun).
> * **Stemming destroys subword token efficiency:** Stemming alters word endings, causing subword tokenizers (like Byte-Pair Encoding) to split known words into disjoint subword fragments.

### Q2: Explain the technical difference between Unicode NFC and NFKC normalization, and why NFKC is critical for vector search.
> **Answer:**
> * **NFC (Canonical Decomposition followed by Canonical Composition):** Only composes characters that are visually and semantically identical in standard typography (e.g., combining `e` + `\u0301` into `é`).
> * **NFKC (Compatibility Decomposition followed by Canonical Composition):** Replaces "compatibility characters" with their standard equivalents. It expands typographic ligatures (`ﬁ` $\to$ `fi`), normalizes superscripts (`m²` $\to$ `m2`), converts full-width Asian characters into standard ASCII, and standardizes circled numbers.
> * **Why it matters:** If an author typed a PDF using the ligature `ﬁ`, an NFC-only or unnormalized search for `"finance"` using standard `fi` letters will fail to match at the byte level, breaking lexical search and fragmenting subword embedding tokens. NFKC unifies them into standard characters.

---

## 12. 🧪 Hands-On Mini-Project: Shannon Entropy OCR Noise Filter

### Objective
Implement a line-by-line Shannon entropy calculator to isolate and prune corrupted OCR lines from a document without affecting valid prose or code snippets.

```python
import math
from typing import List, Tuple

def compute_entropy(text: str) -> float:
    if not text:
        return 0.0
    freqs = {}
    for c in text:
        freqs[c] = freqs.get(c, 0) + 1
    entropy = 0.0
    for count in freqs.values():
        p = count / len(text)
        entropy -= p * math.log2(p)
    return round(entropy, 3)

def filter_ocr_noise(lines: List[str], min_e: float = 2.5, max_e: float = 5.2) -> Tuple[List[str], List[str]]:
    valid_lines = []
    discarded_lines = []

    for line in lines:
        cleaned = line.strip()
        if not cleaned:
            continue
        e = compute_entropy(cleaned)
        # Short lines (<15 chars) can have lower entropy naturally, skip strict bounds
        if len(cleaned) < 15 or (min_e <= e <= max_e):
            valid_lines.append(cleaned)
        else:
            discarded_lines.append(f"[Entropy: {e}] {cleaned}")

    return valid_lines, discarded_lines

# Lab Verification
if __name__ == "__main__":
    sample_document_lines = [
        "# Chapter 1: Introduction to Semantic RAG",
        "Retrieval-Augmented Generation improves the factual accuracy of LLMs.",
        "@#$!*&^%#@!*&^%#@!*&^%",  # OCR Glitch / High-entropy garbage
        "The model queries an external vector database for non-parametric knowledge.",
        "......................",  # Repeated punctuation / Low-entropy noise
        "Enterprise search demands robust Unicode sanitization and clean data.",
    ]

    valid, discarded = filter_ocr_noise(sample_document_lines)

    print("--- Retained Valid Lines ---")
    for v in valid:
        print(f"  ✓ {v}")

    print("\n--- Discarded Noise Lines ---")
    for d in discarded:
        print(f"  ✗ {d}")

    assert len(valid) == 4
    assert len(discarded) == 2
    print("\n✅ Verification Successful: Noise successfully separated from valid text!")
```

---

## 13. ⚡ Quick Revision & Summary Roadmap

### 📌 Core Key Takeaways
* **NFKC is Non-Negotiable:** Normalize all text to Unicode NFKC to unify decomposed accents, compatibility glyphs, and ligatures (`ﬁ` $\to$ `fi`).
* **Never Lowercase Indiscriminately:** Casing provides critical semantic context for Transformer self-attention.
* **Scrub Control & Null Bytes:** Remove `\x00` and non-printing control characters to avoid crashes in database vector engines like `pgvector`.
* **Entropy Pruning:** Filter low-quality OCR text using Shannon entropy ($3.5 \le H(X) \le 4.8$ for natural English).

### 🗺️ Text Sanitization Roadmap

```mermaid
flowchart TD
    In["Raw Extracted Text Stream"] --> Mask["Mask Code Blocks (```)"]
    Mask --> NFKC["Unicode NFKC Normalization"]
    NFKC --> Scrub["Null Byte & Control Character Purge"]
    Scrub --> Hyphen["Line-Wrap De-hyphenation"]
    Hyphen --> WS["Whitespace & Newline Normalization"]
    WS --> Unmask["Restore Code Blocks"]
    Unmask --> Entropy{"Shannon Entropy Check"}
    Entropy -- "Valid (3.5 - 4.8)" --> Out["Clean Document Corpus"]
    Entropy -- "Anomaly (>5.5 or <1.5)" --> DLQ["Triage / Discard Queue"]

    style In fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    style NFKC fill:#fff8e1,stroke:#f57f17,stroke-width:2px
    style Out fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

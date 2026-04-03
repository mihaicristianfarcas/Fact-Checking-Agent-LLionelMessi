# Milestones

Project name: Automated Fact-Checking LLM Agent
Github repository: https://github.com/mihaicristianfarcas/Fact-Checking-Agent-LLionelMessi.git
Team name: LLionelMessi
Team members and their respective group: Mihai-Cristian Farcaș (933), David Croitor (933), Vasile Drăguța (933), Andrei Ungureanu (937)

## 1. Data & Ingestion — Implementation Report

### 1.1 Datasets

We are working with two complementary datasets that together cover both open-domain and political fact-checking.

**FEVER (Fact Extraction and VERification)** is our primary corpus. It contains about 185,000 claims, each linked to one or more Wikipedia sentences that either support, refute, or are insufficient to verify the claim. Labels map directly to the three-class scheme our pipeline targets: SUPPORTED, REFUTED, and NOT_ENOUGH_INFO. We load it from HuggingFace (`fever/fever` v1.0). FEVER also ships a companion Wikipedia dump of ~5.4 million pages, which we use as the evidence corpus for retrieval.

One issue we encountered: the HuggingFace version stores FEVER in a flat format, with one row per evidence sentence rather than one row per claim. Our initial loader assumed a nested structure and was silently producing zero evidence passages per claim. We rewrote the loader to group rows by claim ID before processing, which resolved it.

**LIAR (PolitiFact)** contributes 12,836 political claims scraped from PolitiFact.com across train, validation, and test splits. The original dataset uses six fine-grained labels (ranging from "pants-fire" to "true"), which we collapsed into the same three-class scheme as FEVER. This gives us broader domain coverage — particularly for political claims — while keeping the label space consistent. LIAR does not include evidence text, only the claim and verdict, so its entries serve primarily as training signal for verdict prediction rather than evidence retrieval.

---

### 1.2 Preprocessing

For Wikipedia passages we apply a standard cleaning pipeline: HTML tag removal (BeautifulSoup), Unicode normalization (ftfy + unidecode), and whitespace normalization. We then chunk page content into 500-character overlapping passages (50-character overlap), breaking at sentence boundaries where possible to avoid cutting mid-sentence. This chunk size is a deliberate tradeoff: short enough to fit comfortably within the embedding model's effective context window, long enough to preserve the surrounding context a stance classifier needs.

PolitiFact claims are short standalone statements and are indexed as-is without chunking.

---

### 1.3 Embedding Model

We use `sentence-transformers/all-MiniLM-L6-v2` to produce 384-dimensional dense vector representations. We chose it primarily for its balance between speed and quality: it performs well on semantic similarity benchmarks, has a small footprint (~80 MB), and runs fast enough for batch-encoding hundreds of thousands of passages without requiring a GPU. Larger models such as `all-mpnet-base-v2` would improve retrieval quality at the margins, but the latency and memory cost did not seem justified at this stage of the project.

---

### 1.4 Vector Database

We use **ChromaDB** as our vector store, persisted to disk at `data/index/chroma/`. The collection is configured with cosine similarity and an HNSW index. Each entry stores the passage text along with metadata: the Wikipedia article title (or "politifact"), the source dataset, and chunk position.

The index is not committed to the repository as it is too large for version control. Every team member builds it locally from the downloaded corpora using the provided script.

---

### 1.5 Evidence Corpus Strategy and RAG Design

This is the most significant design decision in our implementation, and it went through a few iterations worth explaining.

Our initial approach indexed only the ~32,000 Wikipedia sentences that FEVER annotators manually cited as gold evidence. This worked well for evaluating retrieval recall against known gold labels, but is fundamentally wrong for inference: if a new claim arrives about a political topic not covered by those specific sentences, the retriever returns nothing relevant. The corpus is too narrow.

The other extreme — indexing all ~25 million sentences from the full Wikipedia dump — is impractical to build and would introduce substantial noise for a system focused on political claims.

We settled on a **political keyword filter over page titles**. Before indexing, we stream all 5.4 million Wikipedia pages and retain only those whose titles contain keywords associated with political content: politician roles (senator, president, governor, minister), institutions (congress, senate, parliament, cabinet, white house), processes (election, campaign, legislation, referendum, impeachment), policy topics (immigration, healthcare, budget, sanctions, diplomacy), and security topics (military, terrorism, intelligence). Pages matching any of these keywords have their full content chunked and indexed.

This approach gives us broad evidence coverage for political claims — the domain our system targets — without indexing content that is irrelevant to fact-checking political statements. The resulting index contains 675,642 passages — 662,806 from 262,993 matched Wikipedia pages and 12,836 PolitiFact claim texts — built in approximately 25 minutes. This is large enough to serve meaningful RAG at inference time while remaining buildable in under an hour on a standard laptop.

The RAG strategy itself is single-stage dense retrieval: a claim is embedded with the same sentence-transformer, a cosine similarity search is run against ChromaDB, and the top-k most relevant passages are returned. Batch retrieval is supported for evaluating large claim sets efficiently. An optional dataset filter allows downstream components to restrict retrieval to Wikipedia passages or PolitiFact entries independently.

---

### 1.6 Current State and Evaluation

We exported claim-evidence-verdict triples to JSONL files for use in fine-tuning:

| Split | Triples |
|-------|---------|
| Train | 126,628 |
| Val   | 15,828  |
| Test  | 15,829  |

Approximately 51% of triples include gold evidence passages. The remainder — NOT_ENOUGH_INFO claims and all PolitiFact entries — have empty evidence lists and serve as negative examples during training.

We also ran a retrieval evaluation baseline against the FEVER labelled_dev set. The primary metric is Recall@k, which measures whether the correct Wikipedia article appears among the top-k retrieved passages. Without a stance classifier, we can only predict SUPPORTED vs NOT_ENOUGH_INFO using a confidence threshold on the top retrieved passage, giving a macro F1 floor of around 0.24 since REFUTED is never predicted. This gap is the explicit handoff point to the next component in the pipeline — the stance classifier will operate on the passages our retriever surfaces.

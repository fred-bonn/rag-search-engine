# rag-search-engine

A small playground for learning how retrieval-augmented generation (RAG) works, built around a
movie dataset. It implements the retrieval half of RAG from scratch and compares the tradeoffs
between the common approaches:

- **Keyword search** — an inverted index with TF, IDF, TF-IDF, and BM25 scoring.
- **Semantic search** — sentence-transformer embeddings with cosine similarity, on whole documents
  or on chunks.
- **Hybrid search** — weighted score fusion and Reciprocal Rank Fusion (RRF) over keyword and
  semantic results.
- **Query enhancement** — LLM-based spelling correction, rewriting, and expansion.
- **Re-ranking** — LLM scoring (individual or batched) and cross-encoder re-ranking.

Each stage is exposed through a CLI so you can poke at scores, compare result sets, and see how
changes to the pipeline affect ranking. Indexes and embeddings are cached under `cache/` so repeated
runs stay fast.

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/). The semantic search components pull in
`sentence-transformers`, and the first run will download model weights.

```sh
uv sync
```

The repo expects the movie data and stopword list in `data/` (`data/movies.json` and
`data/stopwords.txt`). Those files are gitignored, so provide your own copy — the dataset from the
Boot.dev RAG course works as-is.

LLM-backed features read an OpenRouter key from the environment (a `.env` file is loaded
automatically at the repo root):

```sh
echo 'OPENROUTER_API_KEY=your-key-here' > .env
```

## CLI entry points

Run any CLI with `uv run python <script> <command> ...`. Use `<command> --help` for the full flag
list.

### `cli/keyword_search_cli.py` — keyword search

```sh
uv run python cli/keyword_search_cli.py build          # build the inverted index
uv run python cli/keyword_search_cli.py search "bear"  # plain keyword matching
uv run python cli/keyword_search_cli.py bm25search "scary bear movie"
```

| Command | Description |
| --- | --- |
| `build` | Tokenize the movies, build the inverted index, and cache it |
| `search` | Return movies containing any query term |
| `bm25search` | Rank movies by summed BM25 score |
| `tf`, `idf`, `tfidf` | Inspect term frequency, inverse document frequency, and TF-IDF for a doc/term |
| `bm25idf`, `bm25tf` | Inspect BM25's IDF and length-normalized TF components |

### `cli/semantic_search_cli.py` — semantic search

```sh
uv run python cli/semantic_search_cli.py verify
uv run python cli/semantic_search_cli.py search "a movie about grief"
uv run python cli/semantic_search_cli.py search_chunked "a movie about grief"
```

| Command | Description |
| --- | --- |
| `verify` | Load the embedding model and report its max sequence length |
| `embed_text`, `embed_query` | Print embedding dimensions for arbitrary text |
| `verify_embeddings` | Load or build whole-document embeddings and show their shape |
| `embed_chunks` | Chunk descriptions and cache chunk embeddings |
| `search` | Cosine similarity over whole-movie embeddings |
| `search_chunked` | Cosine similarity over chunk embeddings, best chunk per movie |
| `chunk` | Demo fixed-size word chunking with optional overlap |
| `semantic_chunk` | Demo sentence-based chunking with optional overlap |

### `cli/hybrid_search_cli.py` — hybrid search

```sh
uv run python cli/hybrid_search_cli.py weighted-search "scary bear movie" --alpha 0.5
uv run python cli/hybrid_search_cli.py rrf-search "scary bear movie" -k 60
uv run python cli/hybrid_search_cli.py normalize 1 2 3 4
```

| Command | Description |
| --- | --- |
| `weighted-search` | Min-max normalize BM25 and semantic scores, then blend them by `--alpha` |
| `rrf-search` | Fuse results by rank with `1 / (k + rank)` |
| `normalize` | Min-max normalize a list of numbers |

Hybrid search needs the keyword index built first (`build` in the keyword CLI); chunk embeddings are
created on demand.

### `cli/test_llm.py` — OpenRouter smoke test

Sends a single prompt to the configured LLM and prints the response plus token usage. Useful for
confirming that `OPENROUTER_API_KEY` works before using `--enhance` or `--rerank-method`.

```sh
uv run python cli/test_llm.py
```

## Common options

The search commands share a few flags:

- `--limit N` — number of results to return (default `5`).
- `--enhance {spell,rewrite,expand}` — enhance the query with an LLM before searching.
- `--rerank-method {individual,batch,cross_encoder}` — over-fetch results and re-rank them.
  `individual` and `batch` call the LLM; `cross_encoder` runs a local cross-encoder model.

## Project layout

```text
cli/
  keyword_search_cli.py   # keyword/BM25 CLI
  semantic_search_cli.py  # embeddings/chunking CLI
  hybrid_search_cli.py    # hybrid fusion CLI
  test_llm.py             # OpenRouter connectivity check
  lib/
    keyword_search.py     # inverted index, TF-IDF, BM25
    semantic_search.py    # embeddings, cosine similarity, chunking
    hybrid_search.py      # normalization, weighted fusion, RRF
    query_enhancement.py  # LLM query rewriting
    reranking.py          # LLM and cross-encoder rerankers
    formatting.py         # CLI result formatting
    search_utils.py       # paths, defaults, and shared types
data/                     # movies.json + stopwords.txt (not tracked)
cache/                    # generated indexes and embeddings (not tracked)
```

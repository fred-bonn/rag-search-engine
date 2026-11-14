# rag-search-engine

A small playground for learning how retrieval-augmented generation (RAG) works, built around a
movie dataset. It implements the retrieval and generation pieces of RAG from scratch and compares
the tradeoffs between the common approaches:

- **Keyword search** — an inverted index with TF, IDF, TF-IDF, and BM25 scoring.
- **Semantic search** — sentence-transformer embeddings with cosine similarity, on whole documents
  or on chunks.
- **Hybrid search** — weighted score fusion and Reciprocal Rank Fusion (RRF) over keyword and
  semantic results.
- **Query enhancement** — LLM-based spelling correction, rewriting, and expansion.
- **Re-ranking** — LLM scoring (individual or batched) and cross-encoder re-ranking.
- **Answer generation** — RAG answers, summaries, cited responses, and conversational Q&A grounded
  in retrieved documents.

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

### `cli/evaluation_cli.py` — retrieval evaluation

```sh
uv run python cli/evaluation_cli.py --limit 5
```

Runs every query in `data/golden_dataset.json` through RRF search and reports precision@k,
recall@k, and F1 for the retrieved titles.

### `cli/augmented_generation_cli.py` — answer generation

```sh
uv run python cli/augmented_generation_cli.py rag "a movie about grief"
uv run python cli/augmented_generation_cli.py summarize "dinosaur movies" --limit 3
uv run python cli/augmented_generation_cli.py citations "dinosaur movies" --limit 3
uv run python cli/augmented_generation_cli.py question "Which dinosaur movie is best?"
```

| Command | Description |
| --- | --- |
| `rag` | Answer a query using the top RRF results |
| `summarize` | Synthesize an information-dense summary across multiple results |
| `citations` | Answer with numbered `[1]`, `[2]`, etc. source citations |
| `question` | Answer a conversational question using the retrieved movies |

These commands run an RRF hybrid search and pass the results to the LLM. `summarize`, `citations`,
and `question` accept `--limit`; `rag` always uses the default limit. They require
`OPENROUTER_API_KEY`.

## Common options

The search commands share a few flags:

- `--limit N` — number of results to return (default `5`).
- `--enhance {spell,rewrite,expand}` — enhance the query with an LLM before searching.
- `--rerank-method {individual,batch,cross_encoder}` — over-fetch results and re-rank them.
  `individual` and `batch` call the LLM; `cross_encoder` runs a local cross-encoder model.
- `--evaluate` — after printing results, ask the LLM to rate each one from `0` (not relevant) to
  `3` (highly relevant) against the original query.

## Search logs

Every interactive search appends structured events to `logs/search.jsonl` (gitignored), one JSON
object per line. Each search writes a `search` event with the original query, engine, search type,
enhancement details, and the results before re-ranking; when a re-rank method is used, a matching
`rerank` event follows, correlated by `search_id`. Failures write an `error` event instead. Results
are logged with `id`, `title`, and scores only — look up descriptions in `data/movies.json`.

```json
{"schema_version":1,"timestamp":"2026-11-13T21:47:12.481Z","event":"search","search_id":"8f14e45f","query":"scary bear movie","engine":"hybrid","search_type":"rrf_search","enhancement":"expand","enhanced_query":"scary horror grizzly bear movie","limit":5,"params":{"k":60},"results":[{"rank":1,"id":12,"title":"Grizzly","rrf_score":0.0328,"bm25_rank":3,"semantic_rank":1}]}
```

Engine values are `keyword`, `semantic`, and `hybrid`; search types are `search`, `bm25search`,
`chunked_search`, `weighted_search`, and `rrf_search`. With re-ranking, the `search` event contains
the over-fetched pool (up to `limit * 5` results), matching what the reranker saw.

Since logs are JSONL, `jq` makes them human-readable:

```sh
# One line per search: engine/type, query, and result titles
jq -r 'select(.event=="search") | "\(.timestamp) [\(.engine)/\(.search_type)] \(.query) -> \(.results|map(.title)|join(", "))"' logs/search.jsonl

# All events for the most recent search
last_id=$(jq -r 'select(.event=="search") | .search_id' logs/search.jsonl | tail -1)
jq -c --arg id "$last_id" 'select(.search_id == $id)' logs/search.jsonl
```

The evaluation CLI does not write logs.

## Project layout

```text
cli/
  keyword_search_cli.py       # keyword/BM25 CLI
  semantic_search_cli.py      # embeddings/chunking CLI
  hybrid_search_cli.py        # hybrid fusion CLI
  evaluation_cli.py           # retrieval evaluation CLI
  augmented_generation_cli.py # RAG answer generation CLI
  lib/
    keyword_search.py         # inverted index, TF-IDF, BM25
    semantic_search.py        # embeddings, cosine similarity, chunking
    hybrid_search.py          # normalization, weighted fusion, RRF
    query_enhancement.py      # LLM query rewriting
    reranking.py              # LLM and cross-encoder rerankers
    augmented_generation.py   # RAG prompts and LLM answer generation
    llm_client.py             # OpenRouter client setup
    formatting.py             # CLI result formatting
    search_evaluation.py      # LLM relevance scoring
    search_logging.py         # JSONL search log writer
    search_runner.py          # command dispatch, enhance/search/rerank pipeline with logging
    search_utils.py           # paths, defaults, and shared types
data/                         # movies.json + stopwords.txt (not tracked)
cache/                        # generated indexes and embeddings (not tracked)
logs/                         # search.jsonl event log (not tracked)
```

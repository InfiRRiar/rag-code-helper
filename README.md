# Code RAG

Code RAG is an experimental retrieval-augmented generation system for answering questions about a source code repository. The current implementation focuses on Python code: it parses a repository into AST-based code chunks, embeds those chunks, stores them in Chroma, retrieves relevant fragments for a user question, and asks an LLM to answer using only the retrieved context.

The project is still in an active prototype stage. The retriever currently uses dense vector search over function and method chunks, with lightweight metadata and a simple evaluation script.

## Current Capabilities

- Indexes Python repositories by walking `.py` files.
- Splits code with Python `ast` into function and async function chunks.
- Adds file path and class context directly into each chunk.
- Stores embeddings in a local Chroma collection named `code_fragments`.
- Filters retrieval by repository path.
- Translates and normalizes user questions before retrieval.
- Streams an LLM answer based on the retrieved fragments.
- Includes an evaluation dataset and script for measuring top-k retriever hit rate.

## Architecture

The project is split into two runtime parts:

- `src/embs`: embedding service.
- `src/srv`: CLI RAG application and retrieval pipeline.

### Embedding Service

The embedding service is a FastAPI app exposed on port `8080`.

It loads the model configured by `BI_ENCODER_NAME` through Hugging Face Transformers. The default model is:

```text
jinaai/jina-code-embeddings-1.5b
```

Endpoints:

- `POST /embedding/doc`: embeds one text input.
- `POST /embedding/docs`: embeds a list of text inputs.

The LangChain `Embedder` client in `src/srv/components/embedder.py` calls this service with `httpx`.

### Repository Indexing

On first use, the CLI checks whether the requested repository is already indexed in Chroma. If it is not, `parse_repositry` walks the repository, keeps only `.py` files, and sends their chunks to Chroma.

The current splitter extracts:

- top-level functions;
- methods;
- async functions;
- async methods.

Each chunk is stored as text like:

```text
Candidate code snippet:
# full file path: /path/to/file.py
# class: SomeClass
def some_method(...):
    ...
```

At the moment, Chroma metadata contains only:

```python
{"repo": repo_path}
```

### Retrieval And Answering

The interactive CLI flow is:

1. Read the user question.
2. Translate the question to English.
3. Normalize the translated question into a code-focused search query.
4. Add an instruction prefix for the embedding model.
5. Retrieve top-5 chunks from Chroma.
6. Join retrieved fragments.
7. Ask the answer model to respond using the fragments.

The current retrieval method is plain dense similarity search:

```python
similarity_search(query, k=k, filter={"repo": repo})
```

There is no hybrid search, reranking, MMR, symbol graph, or metadata boosting yet.

## Requirements

- Python `>=3.13`
- Docker and Docker Compose
- Hugging Face token for loading the embedding model
- OpenAI-compatible API key and base URL for the LLM components
- GPU support is recommended for local model services

## Configuration

Create a `.env` file in the project root.

Required:

```env
HF_TOKEN=...
OPENAI_API_KEY=...
```

Optional:

```env
OPENAI_BASE_URL=https://api.proxyapi.ru/openai/v1
BASE_EMBEDDINGS_URL=http://127.0.0.1:8080
BI_ENCODER_NAME=jinaai/jina-code-embeddings-1.5b
INSTRUCT_LLM_BASE_URL=http://127.0.0.1:11434
INSTRUCT_LLM_NAME=hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M
```

Current LLM operators use OpenAI-compatible `ChatOpenAI` models configured in `src/srv/components/generative_models.py`. `ChatOllama` remains available as the fallback path inside `LLMOperator`, but the active request translation, normalization, and answer generation chains are wired to OpenAI-compatible models.

## Running

Install Python dependencies:

```bash
uv sync
```

Start infrastructure services:

```bash
make up
```

Run the interactive RAG CLI:

```bash
uv run python -m src.srv.app --repository /absolute/path/to/repository
```

On the first run for a repository, the system will index the code and persist vectors under:

```text
./chroma_langchain_db
```

After indexing, ask questions in the terminal prompt.

## Evaluation

The repository contains a generated evaluation dataset:

```text
eval/dataset/test_data.csv
```

The current evaluation checks whether the expected chunk appears in top-k retrieval results:

```bash
uv run python eval/test_retriever.py
```

The script reports accuracy with and without query normalization.

A dataset generator is also available:

```bash
uv run python eval/dataset/generate_eval_dataset.py
```

It reads existing Chroma documents, asks an LLM to generate developer-style questions for each chunk, and writes the resulting CSV dataset.

## Current Limitations

- Only `.py` files are indexed.
- Only functions and methods are chunked.
- Chroma documents do not yet have stable IDs.
- Metadata is minimal and currently only supports filtering by repository path.
- Retrieval is dense-only.
- There is no reranker.
- There is no duplicate suppression across chunks from the same class or file.
- The evaluation uses exact chunk-content matching rather than stable chunk IDs.
- Query translation and normalization depend on an external OpenAI-compatible LLM endpoint.

## Near-Term Retriever Improvements

The most useful next changes are:

- Add structured chunk metadata: file path, relative path, symbol name, class name, chunk type, and line range.
- Keep imports as features attached to action chunks, not as standalone chunks.
- Add class overview chunks for architectural questions while keeping method chunks as the main retrieval unit.
- Introduce stable document IDs based on repository, path, symbol, line range, and content hash.
- Fetch a wider candidate set and apply MMR or reranking before returning final chunks.
- Add hybrid search that combines dense similarity with keyword or full-text search.
- Extend evaluation to report `hit@k`, `MRR`, and per-query failure examples.

## Project Layout

```text
src/
  embs/
    app.py              # FastAPI embedding service
    encoders.py         # Hugging Face encoder wrapper
  srv/
    app.py              # Interactive CLI RAG loop
    components/
      ast_splitter.py   # AST-based code chunker
      chroma_operator.py# Chroma storage and retrieval wrapper
      embedder.py       # HTTP client for embedding service
      llm_operator.py   # Prompt + model orchestration
      generative_models.py
    prompt_templates/   # Prompt templates for translation, normalization, answering
eval/
  dataset/
    test_data.csv
    generate_eval_dataset.py
  test_retriever.py
```

# Copilot Instructions for Audio Customer Support Agent

## Build, test, and lint commands

### Setup
```bash
pip install -r requirements.txt
```

### Run services for end-to-end manual testing
```bash
# Terminal 1: FastAPI server
python -m src.api.server

# Terminal 2: Streamlit test UI
streamlit run streamlit_app.py
```

### Tests
```bash
# Full test suite
pytest

# Coverage (as documented)
pytest --cov=src tests/

# Single test file
pytest tests/test_stt.py -v

# Single test function
pytest tests/test_stt.py::TestSTTService::test_transcribe_not_initialized -v
```

### Lint/type formatting tools in this repo
`black`, `flake8`, and `mypy` are included in `requirements.txt`.

Typical invocation:
```bash
black src tests streamlit_app.py
flake8 src tests streamlit_app.py
mypy src
```

## High-level architecture

- The runtime flow is **STT -> LLM (with RAG) -> TTS**, orchestrated by `AudioSupportPipeline` in `src/pipeline.py`.
- The HTTP layer in `src/api/server.py` exposes this flow via FastAPI (`/chat/text`, `/chat/audio`, `/health`) and manages one global pipeline instance on startup/shutdown.
- The local test frontend in `streamlit_app.py` calls the FastAPI endpoints (text, audio, health) and is the main manual integration harness.
- The LLM layer (`src/llm/agent.py`) owns RAG knowledge retrieval:
  - ChromaDB persistent storage under `./data/chroma_db`
  - fixed built-in customer-support documents ingested at initialization
  - `_rag_search()` is the key retrieval hook expected to query ChromaDB and format top matches.
- STT and TTS are interface-first modules (`src/stt/base_stt.py`, `src/tts/base_tts.py`) with abstract base classes and template concrete classes to complete.

## Key conventions in this codebase

- **Async-first interfaces:** pipeline, agent, STT, and TTS public methods are async (`initialize`, process methods, `cleanup`), and implementations should preserve this pattern.
- **Component readiness contract:** components track readiness using `is_initialized` and expose `is_ready()`; pipeline health responses are built from these flags.
- **Template placeholders are intentional:** many methods contain TODO stubs and placeholder returns; wire real implementations by uncommenting/inserting logic where scaffolding already exists (pipeline init/processing, API startup, RAG query path).
- **Knowledge base source of truth lives in code:** customer support documents are defined in `CustomerSupportAgent._get_customer_support_documents()` and ingested automatically; avoid moving this to external files unless the repo structure is intentionally changed.
- **Debugging workflow is endpoint-driven:** use `/health` and debug endpoints plus `streamlit_app.py` and `src/utils/kb_test.py` as the primary integration/debug loop.

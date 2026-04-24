# Audio Customer Support Agent

An end-to-end customer support assistant that supports:

- text chat
- audio chat
- transcript-aware responses
- retrieval-augmented answers from a built-in customer support knowledge base
- HTTP and WebSocket streaming for low-latency audio flows

The project is split into two apps:

1. A FastAPI backend that orchestrates STT -> LLM -> TTS
2. A Streamlit frontend used to test the backend interactively

For the full system design, see [architecture.md](architecture.md).

## What This App Does

The backend accepts either plain text or audio input.

- For text requests, it sends the message to the LLM agent and can optionally synthesize audio.
- For audio requests, it transcribes speech to text, generates a customer-support answer, converts the answer back to speech, and returns both audio and transcript metadata.
- For streaming requests, it supports chunked audio processing and incremental events over HTTP or WebSocket.

The LLM agent also uses a built-in ChromaDB knowledge base containing customer support documents for topics like returns, shipping, billing, warranty, and technical support.

## Tech Stack

- Backend: FastAPI, Uvicorn
- Frontend: Streamlit
- STT: Groq Whisper API
- LLM: Groq or OpenAI chat-completions compatible HTTP API
- TTS: Edge TTS
- Retrieval: ChromaDB + sentence-transformers
- Transport: REST, chunked HTTP streaming, WebSocket streaming


## Project Working Video

 [streamlit-streamlit_app-2026-04-24-07-22-28.webm](https://github.com/user-attachments/assets/9966f464-2302-4008-9589-62327df72d1e)


## Project Structure

```text
zangoh-project/
|-- src/
|   |-- api/server.py          # FastAPI app and endpoints
|   |-- pipeline.py            # STT -> LLM -> TTS orchestration
|   |-- llm/agent.py           # LLM client and RAG knowledge base
|   |-- stt/base_stt.py        # Groq STT implementation
|   |-- tts/base_tts.py        # Edge TTS implementation
|   `-- utils/audio_chunking.py
|-- streamlit_app.py           # Frontend test UI
|-- tests/
|   |-- test_audio_contract.py
|   |-- test_text_contract.py
|   `-- test_stt.py
|-- docs/
|   `-- RAG_IMPLEMENTATION_GUIDE.md
|-- data/chroma_db/            # Persistent Chroma database
|-- .env.example
|-- pyproject.toml
|-- README.md
`-- architecture.md
```

## Prerequisites

- Python 3.11+
- Windows PowerShell
- A Groq API key for the default setup

Optional:

- An OpenAI API key if you want to use `LLM_PROVIDER=openai`
- A working microphone if you want to record audio in Streamlit

## Setup

From the project root:

```powershell
cd C:\Users\Tarun Dange\Desktop\Zangoh\zangoh-project
uv sync
Copy-Item .env.example .env
```

Then open `.env` and set at least:

```env
GROQ_API_KEY=your_key_here
STT_PROVIDER=groq
LLM_PROVIDER=groq
TTS_PROVIDER=edge
```

You can also override provider-specific settings such as:

- `STT_MODEL`
- `LLM_MODEL`
- `TTS_VOICE`
- `LLM_TEMPERATURE`
- `STT_LANGUAGE`

## Run The Backend

Use module execution instead of the `uvicorn.exe` shim. That avoids the broken launcher issue that can happen when the virtual environment path changes.

```powershell
cd C:\Users\Tarun Dange\Desktop\Zangoh\zangoh-project
.\.venv\Scripts\python.exe -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
```

If you want a different port:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api.server:app --host 0.0.0.0 --port 8001 --reload
```

Useful backend URLs:

- API root: `http://localhost:8000/`
- Health: `http://localhost:8000/health`
- Swagger docs: `http://localhost:8000/docs`

## Run The Frontend

Open a second terminal:

```powershell
cd C:\Users\Tarun Dange\Desktop\Zangoh\zangoh-project
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Then open:

- Streamlit UI: `http://localhost:8501`

If your backend is running on a non-default port like `8001`, update the Streamlit sidebar field:

- `API Server URL = http://localhost:8001`

## Main Features

### Text chat

- Sends text to `POST /chat/text`
- Works even if STT or TTS is unavailable, as long as the LLM is ready
- Returns response text, audio availability, and processing time

### Audio chat

- Sends audio to `POST /chat/audio`
- Returns:
  - base64-encoded audio response
  - user transcript
  - agent transcript
  - total processing time

### Streaming audio

- `POST /chat/audio/stream` for chunked HTTP audio response
- `WS /chat/audio/stream` for low-latency incremental events

### Health monitoring

- `GET /health` reports:
  - pipeline initialization
  - STT readiness
  - LLM readiness
  - TTS readiness

The server can start in degraded mode:

- LLM is required
- STT can be unavailable and text chat still works
- TTS can be unavailable and text chat still works

## API Summary

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/` | Basic API metadata |
| `GET` | `/health` | Component readiness |
| `POST` | `/chat/text` | Text-only conversation |
| `POST` | `/chat/audio` | Full STT -> LLM -> TTS flow |
| `GET` | `/chat/audio/{text}` | Quick text-to-speech test |
| `POST` | `/debug/stt` | STT-only debug route |
| `POST` | `/chat/audio/stream` | HTTP streaming audio |
| `WS` | `/chat/audio/stream` | WebSocket streaming audio |

## Knowledge Base

The knowledge base is created inside `src/llm/agent.py` and persisted to `data/chroma_db`.

It includes customer support content for:

- returns
- shipping
- support contact
- warranty
- technical help
- account management
- orders
- payments
- billing
- products

At startup the LLM agent:

1. Opens or creates the Chroma collection
2. Loads `all-MiniLM-L6-v2` embeddings
3. Ingests the built-in documents if the collection is empty
4. Reuses the persisted collection on later restarts

If ChromaDB or embeddings are unavailable, the backend continues without RAG context instead of failing the whole app.

## Testing

Current tests in `tests/` focus on the API contract:

```powershell
cd C:\Users\Tarun Dange\Desktop\Zangoh\zangoh-project
.\.venv\Scripts\python.exe -m pytest tests\test_audio_contract.py tests\test_text_contract.py -q
```

You can also run the STT-focused tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_stt.py -q
```

## Troubleshooting

### `uvicorn src.api.server:app` fails with a shim or trampoline error

Use:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend says server is not accessible

Check:

1. The backend terminal is still running
2. The Streamlit `API Server URL` matches the backend port
3. `http://localhost:8000/health` opens in the browser

### Text chat works but audio does not

That usually means:

- STT is not configured correctly
- TTS is unavailable
- the backend started in degraded mode

Open `/health` and inspect the component readiness message.

### Audio recording is unavailable in Streamlit

Install or repair the audio dependency:

```powershell
uv sync
```

Then make sure your microphone is available to the app.

## Documentation

- System design: [architecture.md](architecture.md)
- RAG guide: [docs/RAG_IMPLEMENTATION_GUIDE.md](docs/RAG_IMPLEMENTATION_GUIDE.md)

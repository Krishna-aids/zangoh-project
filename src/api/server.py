"""
FastAPI Server for Audio Customer Support Agent

Provides REST API endpoints for the full STT -> LLM -> TTS pipeline with
transcript and timing metadata on every audio interaction.

Endpoints
─────────
GET  /                        – API info
GET  /health                  – Component health
POST /chat/text               – Text query  (LLM-first, optional TTS)
POST /chat/audio              – Audio query (STT + LLM + TTS) → JSON + transcript
GET  /chat/audio/{text}       – Quick TTS test
POST /debug/stt               – Isolated STT test
POST /chat/audio/stream       – HTTP streaming fallback
WS   /chat/audio/stream       – WebSocket low-latency streaming
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, AsyncIterator
import asyncio
import base64
import logging
import os
from time import perf_counter

from src.pipeline import AudioSupportPipeline, create_pipeline

logger = logging.getLogger(__name__)
import dotenv

dotenv.load_dotenv()


# ── Env helpers ───────────────────────────────────────────────────────────────

def _first_env(*names: str) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _env_float(*names: str, default: float) -> float:
    value = _first_env(*names)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid float for %s=%r. Using default=%s.", names[0], value, default)
        return default


def _env_int(*names: str, default: int) -> int:
    value = _first_env(*names)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid int for %s=%r. Using default=%s.", names[0], value, default)
        return default


# ── WebSocket tuning ──────────────────────────────────────────────────────────

WS_MAX_QUEUE_SIZE = _env_int("WS_STREAM_QUEUE_MAXSIZE", default=32)
WS_MAX_FRAME_BYTES = _env_int("WS_STREAM_MAX_FRAME_BYTES", default=512 * 1024)
WS_QUEUE_PUT_TIMEOUT_SECONDS = _env_float(
    "WS_STREAM_QUEUE_PUT_TIMEOUT_SECONDS", default=0.5
)


def _pipeline_error_message(event: Dict[str, Any]) -> str:
    stage = event.get("stage", "unknown")
    chunk_id = event.get("chunk_id")
    message = event.get("message", "Unknown pipeline error")
    chunk_text = f" chunk={chunk_id}" if chunk_id is not None else ""
    return f"Pipeline error at {stage}{chunk_text}: {message}"


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _health_message(components: Dict[str, bool], errors: Dict[str, str]) -> str:
    if all(components.values()):
        return "All components ready"

    unavailable = []
    for component_key, ready_key in (
        ("stt", "stt_ready"),
        ("llm", "llm_ready"),
        ("tts", "tts_ready"),
    ):
        if components.get(ready_key):
            continue
        detail = errors.get(component_key)
        if detail:
            unavailable.append(f"{component_key.upper()}: {detail}")
        else:
            unavailable.append(f"{component_key.upper()}: not ready")

    prefix = "Text chat is available, but some components are degraded. "
    if not components.get("llm_ready", False):
        prefix = "Core LLM service is unavailable. "
    return prefix + "; ".join(unavailable)


# ── Pydantic models ───────────────────────────────────────────────────────────

class TextRequest(BaseModel):
    """Request body for text queries."""
    text: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    components: Dict[str, bool]
    message: str


class TextResponse(BaseModel):
    """Response for /chat/text."""
    response_text: str
    audio_available: bool
    processing_time_ms: int


class TranscriptData(BaseModel):
    """User/agent text transcript for an audio interaction."""
    user_input: str
    agent_response: str


class OptionalTranscriptData(BaseModel):
    """Nullable transcript fields for standardized error payloads."""
    user_input: Optional[str] = None
    agent_response: Optional[str] = None


class EnhancedAudioResponse(BaseModel):
    """
    Response for POST /chat/audio.

    audio_response is the base64-encoded TTS audio (MP3).
    """
    success: bool
    audio_response: str          # base64-encoded audio bytes
    transcript: TranscriptData
    processing_time_ms: int


class AudioErrorResponse(BaseModel):
    """Error response for POST /chat/audio."""
    success: bool = False
    error: str
    transcript: OptionalTranscriptData = Field(default_factory=OptionalTranscriptData)
    processing_time_ms: int


def _audio_error_response(
    status_code: int,
    error: str,
    start_time: Optional[float] = None,
) -> JSONResponse:
    processing_time_ms = 0
    if start_time is not None:
        processing_time_ms = int((perf_counter() - start_time) * 1000)

    payload = AudioErrorResponse(
        error=error,
        processing_time_ms=processing_time_ms,
    ).model_dump()
    return JSONResponse(status_code=status_code, content=payload)


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Audio Customer Support Agent API",
    description="REST API for the STT -> LLM -> TTS pipeline with transcript support",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline: Optional[AudioSupportPipeline] = None
logging.basicConfig(level=logging.INFO)


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Initialize the pipeline from environment variables on server startup."""
    global pipeline

    try:
        logger.info("Starting Audio Support Agent API server...")

        stt_config = {
            "provider": _first_env("STT_PROVIDER") or "groq",
            "api_key": _first_env("STT_API_KEY"),
            "model": _first_env("STT_MODEL"),
            "language": _first_env("STT_LANGUAGE"),
            "timeout_seconds": _env_float("STT_TIMEOUT_SECONDS", default=20.0),
            "max_chunk_bytes": _env_int("STT_MAX_CHUNK_BYTES", default=5 * 1024 * 1024),
        }

        llm_config = {
            "provider": _first_env("LLM_PROVIDER") or "groq",
            "api_key": _first_env("LLM_API_KEY"),
            "model": _first_env("LLM_MODEL"),
            "temperature": _env_float("LLM_TEMPERATURE", default=0.2),
            "timeout_seconds": _env_float("LLM_TIMEOUT_SECONDS", default=30.0),
        }

        tts_config = {
            "provider": _first_env("TTS_PROVIDER") or "edge",
            "api_key": _first_env("TTS_API_KEY"),
            "voice": _first_env("TTS_VOICE"),
            "rate": _first_env("TTS_RATE"),
            "volume": _first_env("TTS_VOLUME"),
            "timeout_seconds": _env_float("TTS_TIMEOUT_SECONDS", default=20.0),
        }

        pipeline = await create_pipeline(stt_config, llm_config, tts_config)
        logger.info("Pipeline initialized and ready.")

    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {str(e)}")
        pipeline = None


@app.on_event("shutdown")
async def shutdown_event():
    """Gracefully shut down the pipeline."""
    global pipeline
    if pipeline:
        logger.info("Shutting down pipeline...")
        await pipeline.cleanup()
        pipeline = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_model=Dict[str, str])
async def root():
    return {
        "message": "Audio Customer Support Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Return the readiness status of every pipeline component."""
    global pipeline

    if not pipeline:
        return HealthResponse(
            status="unhealthy",
            components={
                "pipeline_initialized": False,
                "stt_ready": False,
                "llm_ready": False,
                "tts_ready": False,
            },
            message="Pipeline not initialized",
        )

    try:
        components = await pipeline.health_check()
        all_healthy = all(components.values())
        errors = getattr(pipeline, "initialization_errors", {}) or {}
        return HealthResponse(
            status="healthy" if all_healthy else "unhealthy",
            components=components,
            message=_health_message(components, errors),
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="error",
            components={},
            message=f"Health check failed: {str(e)}",
        )


@app.post("/chat/text", response_model=TextResponse)
async def chat_text(request: TextRequest):
    """
    Process a text query through the LLM agent.

    Useful for testing the RAG + LLM stack without requiring STT or TTS.
    """
    global pipeline

    if not pipeline or not getattr(pipeline, "llm_agent", None):
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        parameters = dict(request.parameters or {})
        generate_audio = False
        for name in ("generate_audio", "include_audio", "synthesize_audio"):
            generate_audio = _coerce_bool(parameters.pop(name, None)) or generate_audio

        response_text, audio_available, processing_time_ms = await pipeline.process_text_with_timing(
            request.text,
            generate_audio=generate_audio,
            allow_tts_failure=True,
            **parameters,
        )

        return TextResponse(
            response_text=response_text,
            audio_available=audio_available,
            processing_time_ms=processing_time_ms,
        )

    except Exception as e:
        logger.error(f"Text processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/audio", response_model=EnhancedAudioResponse | AudioErrorResponse)
async def chat_audio(audio: UploadFile = File(...)):
    """
    Process an audio file through the full STT -> LLM -> TTS pipeline.

    Returns a JSON payload containing:
    - ``audio_response``: base64-encoded MP3 audio
    - ``transcript``: user's spoken text + agent's text response
    - ``processing_time_ms``: total wall-clock time in milliseconds
    """
    global pipeline
    start_time = perf_counter()

    if not pipeline:
        return _audio_error_response(503, "Pipeline not initialized", start_time)

    try:
        audio_bytes = await audio.read()
        if len(audio_bytes) == 0:
            return _audio_error_response(400, "Empty audio file", start_time)

        response_audio, transcript_data, processing_time_ms = (
            await pipeline.process_audio_with_transcript(audio_bytes)
        )

        if not isinstance(response_audio, bytes) or len(response_audio) == 0:
            return _audio_error_response(
                500,
                "Audio response payload is missing or invalid",
                start_time,
            )

        encoded_audio = base64.b64encode(response_audio).decode("utf-8")

        return EnhancedAudioResponse(
            success=True,
            audio_response=encoded_audio,
            transcript=TranscriptData(
                user_input=str(getattr(transcript_data, "user_input", "")),
                agent_response=str(getattr(transcript_data, "agent_response", "")),
            ),
            processing_time_ms=int(processing_time_ms),
        )

    except HTTPException as exc:
        return _audio_error_response(exc.status_code, str(exc.detail), start_time)
    except Exception as e:
        logger.error(f"Audio processing failed: {str(e)}")
        return _audio_error_response(500, str(e), start_time)


@app.get("/chat/audio/{text}")
async def text_to_audio(text: str):
    """
    Convert a text string to speech and return the raw audio file.

    Useful for quick TTS smoke-tests:
        curl "http://localhost:8000/chat/audio/Hello%20world" --output test.mp3
    """
    global pipeline

    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        if not pipeline.tts:
            raise HTTPException(status_code=503, detail="TTS not available")

        audio_bytes = await pipeline.tts.synthesize(text)
        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=tts_output.mp3"},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/debug/stt")
async def debug_stt(audio: UploadFile = File(...)):
    """
    Transcribe an uploaded audio file using only the STT component.

    Useful for verifying STT accuracy in isolation:
        curl -X POST http://localhost:8000/debug/stt -F "audio=@test.wav"
    """
    global pipeline

    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        audio_bytes = await audio.read()

        if not pipeline.stt:
            raise HTTPException(status_code=503, detail="STT not available")

        transcription = await pipeline.stt.transcribe(audio_bytes)
        return {"transcription": transcription}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"STT debug failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Streaming endpoints ───────────────────────────────────────────────────────

async def _single_chunk_stream(audio_bytes: bytes) -> AsyncIterator[bytes]:
    """Wrap one audio payload as an async iterator for the streaming pipeline."""
    yield audio_bytes


@app.post("/chat/audio/stream")
async def chat_audio_stream_http(audio: UploadFile = File(...)):
    """
    HTTP streaming fallback: upload audio, receive audio chunks via chunked transfer.
    """
    global pipeline

    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    stream_iter = pipeline.process_audio_stream(_single_chunk_stream(audio_bytes))

    first_audio_chunk: Optional[bytes] = None
    async for first_event in stream_iter:
        event_type = first_event.get("type")
        if event_type == "error":
            raise HTTPException(
                status_code=502, detail=_pipeline_error_message(first_event)
            )
        if event_type == "tts.audio":
            piece = first_event.get("audio", b"")
            if piece:
                first_audio_chunk = piece
                break

    if not first_audio_chunk:
        raise HTTPException(
            status_code=502,
            detail="Streaming pipeline finished without audio output",
        )

    async def audio_generator() -> AsyncIterator[bytes]:
        if first_audio_chunk:
            yield first_audio_chunk
        async for event in stream_iter:
            event_type = event.get("type")
            if event_type == "error":
                raise RuntimeError(_pipeline_error_message(event))
            if event_type == "tts.audio":
                piece = event.get("audio", b"")
                if piece:
                    yield piece

    return StreamingResponse(audio_generator(), media_type="audio/mpeg")


@app.websocket("/chat/audio/stream")
async def chat_audio_stream_ws(websocket: WebSocket):
    """
    WebSocket low-latency streaming endpoint.

    Protocol:
    - Client sends binary frames (WAV audio chunks)
    - Client sends text "END" to signal end of turn
    - Server emits JSON events:
      - ``stt.partial``  – incremental transcript text
      - ``llm.token``    – LLM output token
      - ``tts.audio``    – base64-encoded audio chunk
      - ``stream.complete`` – turn finished
      - ``error``        – pipeline error
    """
    global pipeline
    await websocket.accept()

    if not pipeline:
        await websocket.send_json({"type": "error", "message": "Pipeline not initialized"})
        await websocket.close(code=1011)
        return

    queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=WS_MAX_QUEUE_SIZE)

    async def audio_stream() -> AsyncIterator[bytes]:
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item

    async def process_and_send() -> None:
        async for event in pipeline.process_audio_stream(audio_stream()):
            event_type = event.get("type")
            if event_type == "tts.audio":
                audio_piece = event.get("audio", b"")
                await websocket.send_json(
                    {
                        "type": "tts.audio",
                        "chunk_id": event.get("chunk_id"),
                        "audio_base64": base64.b64encode(audio_piece).decode("utf-8"),
                    }
                )
            else:
                await websocket.send_json(event)

    processing_task = asyncio.create_task(process_and_send())

    try:
        async def enqueue_audio(item: Optional[bytes]) -> bool:
            try:
                await asyncio.wait_for(
                    queue.put(item), timeout=WS_QUEUE_PUT_TIMEOUT_SECONDS
                )
                return True
            except asyncio.TimeoutError:
                await websocket.send_json(
                    {
                        "type": "warning",
                        "code": "backpressure",
                        "message": "Server audio buffer is full. Slow down and retry.",
                    }
                )
                return False

        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            audio_bytes = message.get("bytes")
            if audio_bytes:
                if len(audio_bytes) > WS_MAX_FRAME_BYTES:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "code": "frame_too_large",
                            "message": f"Audio frame exceeds {WS_MAX_FRAME_BYTES} bytes.",
                        }
                    )
                    await websocket.close(code=1009)
                    break
                await enqueue_audio(audio_bytes)
                continue

            text = (message.get("text") or "").strip().upper()
            if text == "END":
                enqueued = await enqueue_audio(None)
                if not enqueued:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "code": "end_not_queued",
                            "message": "Unable to finish stream while server buffer is full.",
                        }
                    )
                    await websocket.close(code=1013)
                    break
                await processing_task
                await websocket.send_json({"type": "stream.complete"})
                break

            if text:
                await websocket.send_json(
                    {"type": "warning", "message": "Send audio bytes or END."}
                )

    except WebSocketDisconnect:
        logger.info("Audio stream WebSocket disconnected.")
    except Exception as e:
        logger.error(f"WebSocket streaming failed: {str(e)}")
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        if not processing_task.done():
            processing_task.cancel()
        await websocket.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )

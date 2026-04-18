"""
FastAPI Server for Audio Customer Support Agent

This module provides REST API endpoints for testing the audio support pipeline.
Students can use this server to test their implementations via HTTP requests.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, AsyncIterator
import asyncio
import base64
import logging
import os

from src.pipeline import AudioSupportPipeline, create_pipeline

logger = logging.getLogger(__name__)


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


WS_MAX_QUEUE_SIZE = _env_int("WS_STREAM_QUEUE_MAXSIZE", default=32)
WS_MAX_FRAME_BYTES = _env_int("WS_STREAM_MAX_FRAME_BYTES", default=512 * 1024)
WS_QUEUE_PUT_TIMEOUT_SECONDS = _env_float("WS_STREAM_QUEUE_PUT_TIMEOUT_SECONDS", default=0.5)


def _pipeline_error_message(event: Dict[str, Any]) -> str:
    stage = event.get("stage", "unknown")
    chunk_id = event.get("chunk_id")
    message = event.get("message", "Unknown pipeline error")
    chunk_text = f" chunk={chunk_id}" if chunk_id is not None else ""
    return f"Pipeline error at {stage}{chunk_text}: {message}"


class TextRequest(BaseModel):
    """Request model for text-based queries."""
    text: str
    parameters: Optional[Dict[str, Any]] = {}


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    components: Dict[str, bool]
    message: str


class TextResponse(BaseModel):
    """Response model for text queries."""
    response_text: str
    audio_available: bool
    processing_time_ms: int


class TranscriptData(BaseModel):
    """Transcript metadata for audio interactions."""
    user_input: str
    agent_response: str


class EnhancedAudioResponse(BaseModel):
    """Response model for audio queries with transcript and timing."""
    success: bool
    audio_response: str
    transcript: TranscriptData
    processing_time_ms: int


app = FastAPI(
    title="Audio Customer Support Agent API",
    description="REST API for testing the STT -> LLM -> TTS pipeline",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: Optional[AudioSupportPipeline] = None

# Configure logging
logging.basicConfig(level=logging.INFO)


@app.on_event("startup")
async def startup_event():
    """
    TODO: Initialize the pipeline on server startup.
    
    Students should configure the pipeline with their API keys and settings.
    """
    global pipeline
    
    try:
        logger.info("Starting Audio Support Agent API server...")

        stt_config = {
            "provider": _first_env("STT_PROVIDER") or "groq",
            "api_key": _first_env("STT_API_KEY", "GROQ_API_KEY"),
            "model": _first_env("STT_MODEL", "GROQ_STT_MODEL") or "whisper-large-v3-turbo",
            "language": _first_env("STT_LANGUAGE"),
            "timeout_seconds": _env_float("STT_TIMEOUT_SECONDS", default=20.0),
            "max_chunk_bytes": _env_int("STT_MAX_CHUNK_BYTES", default=5 * 1024 * 1024),
        }

        llm_config = {
            "provider": _first_env("LLM_PROVIDER") or "groq",
            "api_key": _first_env("LLM_API_KEY", "GROQ_API_KEY"),
            "model": _first_env("LLM_MODEL", "GROQ_LLM_MODEL") or "llama-3.1-8b-instant",
            "temperature": _env_float("LLM_TEMPERATURE", default=0.2),
            "timeout_seconds": _env_float("LLM_TIMEOUT_SECONDS", default=30.0),
        }

        tts_config = {
            "provider": _first_env("TTS_PROVIDER") or "edge",
            "api_key": _first_env("TTS_API_KEY"),
            "voice": _first_env("TTS_VOICE", "EDGE_TTS_VOICE") or "en-US-AriaNeural",
            "rate": _first_env("TTS_RATE", "EDGE_TTS_RATE") or "+0%",
            "volume": _first_env("TTS_VOLUME", "EDGE_TTS_VOLUME") or "+0%",
            "timeout_seconds": _env_float("TTS_TIMEOUT_SECONDS", default=20.0),
        }

        pipeline = await create_pipeline(stt_config, llm_config, tts_config)
        logger.info("Pipeline initialized and ready.")
        
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {str(e)}")
        pipeline = None


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup pipeline resources on server shutdown."""
    global pipeline
    
    if pipeline:
        logger.info("Shutting down pipeline...")
        await pipeline.cleanup()
        pipeline = None


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Audio Customer Support Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns the status of all pipeline components.
    """
    global pipeline
    
    if not pipeline:
        return HealthResponse(
            status="unhealthy",
            components={
                "pipeline_initialized": False,
                "stt_ready": False,
                "llm_ready": False,
                "tts_ready": False
            },
            message="Pipeline not initialized"
        )
    
    try:
        components = await pipeline.health_check()
        
        all_healthy = all(components.values())
        
        return HealthResponse(
            status="healthy" if all_healthy else "unhealthy",
            components=components,
            message="All components ready" if all_healthy else "Some components not ready"
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="error",
            components={},
            message=f"Health check failed: {str(e)}"
        )


@app.post("/chat/text", response_model=TextResponse)
async def chat_text(request: TextRequest):
    """
    Process text query through the LLM agent.
    
    This endpoint allows testing the LLM component without audio processing.
    """
    global pipeline
    
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        import time
        start_time = time.time()
        
        response_text, response_audio = await pipeline.process_text(
            request.text,
            **(request.parameters or {})
        )
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return TextResponse(
            response_text=response_text,
            audio_available=len(response_audio) > 0,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Text processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/audio", response_model=EnhancedAudioResponse)
async def chat_audio(audio: UploadFile = File(...)):
    """
    TODO: Process audio query through the complete pipeline.
    
    This endpoint handles the full STT -> LLM -> TTS pipeline.
    
    Args:
        audio: Audio file upload (WAV, MP3, etc.)
        
    Returns:
        JSON payload with base64 audio, transcript, and timing metadata
    """
    global pipeline
    
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        audio_bytes = await audio.read()
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")

        process_with_transcript = getattr(pipeline, "process_audio_with_transcript", None)
        if not callable(process_with_transcript):
            raise HTTPException(status_code=503, detail="Transcript-capable audio processing is unavailable")

        response_audio, transcript_data, processing_time_ms = await process_with_transcript(
            audio_bytes
        )
        if not isinstance(response_audio, bytes):
            raise HTTPException(status_code=500, detail="Audio response payload is missing or invalid")
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
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/audio/{text}")
async def text_to_audio(text: str):
    """
    TODO: Convert text to audio using TTS.
    
    Useful for testing TTS component independently.
    
    Args:
        text: Text to convert to speech
        
    Returns:
        Audio file as bytes
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
            headers={
                "Content-Disposition": f"attachment; filename=tts_output.mp3"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/debug/stt")
async def debug_stt(audio: UploadFile = File(...)):
    """
    TODO: Debug endpoint for testing STT component independently.
    
    Args:
        audio: Audio file to transcribe
        
    Returns:
        Transcription result
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
        
    except Exception as e:
        logger.error(f"STT debug failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def _single_chunk_stream(audio_bytes: bytes) -> AsyncIterator[bytes]:
    """Wrap one audio payload as an async iterator for the streaming pipeline."""
    yield audio_bytes


@app.post("/chat/audio/stream")
async def chat_audio_stream_http(audio: UploadFile = File(...)):
    """
    HTTP streaming fallback endpoint.
    Receives uploaded audio and streams response audio chunks.
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
        first_event_type = first_event.get("type")
        if first_event_type == "error":
            raise HTTPException(status_code=502, detail=_pipeline_error_message(first_event))
        if first_event_type == "tts.audio":
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
    WebSocket primary streaming endpoint.
    - Client sends binary audio frames (WAV chunks recommended).
    - Client sends text frame "END" to signal end of turn.
    - Server returns JSON events:
      - stt.partial
      - llm.token
      - tts.audio (base64-encoded audio chunk)
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
                await asyncio.wait_for(queue.put(item), timeout=WS_QUEUE_PUT_TIMEOUT_SECONDS)
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
                enqueued = await enqueue_audio(audio_bytes)
                if not enqueued:
                    continue
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
        logger.info("Audio stream websocket disconnected.")
    except Exception as e:
        logger.error(f"WebSocket streaming failed: {str(e)}")
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        if not processing_task.done():
            processing_task.cancel()
        await websocket.close()


if __name__ == "__main__":
    import uvicorn
    
    # TODO: Students can modify these settings for development
    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )

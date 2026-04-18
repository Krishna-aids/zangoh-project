"""
Audio chunking helpers for low-latency streaming pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
from typing import Dict, List
import wave


@dataclass
class AudioChunk:
    """Represents one ordered audio chunk."""

    chunk_id: int
    start_ms: int
    end_ms: int
    audio_bytes: bytes


def split_wav_audio_chunks(audio_bytes: bytes, chunk_seconds: float = 2.0) -> List[AudioChunk]:
    """
    Split WAV bytes into smaller WAV chunks.

    If bytes are not valid WAV, this falls back to one chunk containing the input.
    """
    if not audio_bytes:
        return []
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be > 0")

    try:
        source_buffer = io.BytesIO(audio_bytes)
        with wave.open(source_buffer, "rb") as wav_in:
            nchannels = wav_in.getnchannels()
            sampwidth = wav_in.getsampwidth()
            framerate = wav_in.getframerate()
            nframes = wav_in.getnframes()
            pcm = wav_in.readframes(nframes)
    except wave.Error:
        # Non-WAV input is passed through as a single chunk.
        return [AudioChunk(chunk_id=0, start_ms=0, end_ms=0, audio_bytes=audio_bytes)]

    bytes_per_frame = nchannels * sampwidth
    if bytes_per_frame <= 0 or framerate <= 0:
        return [AudioChunk(chunk_id=0, start_ms=0, end_ms=0, audio_bytes=audio_bytes)]

    frames_per_chunk = max(int(framerate * chunk_seconds), 1)
    bytes_per_chunk = frames_per_chunk * bytes_per_frame

    chunks: List[AudioChunk] = []
    offset = 0
    chunk_id = 0
    while offset < len(pcm):
        piece = pcm[offset : offset + bytes_per_chunk]
        if not piece:
            break

        start_frame = offset // bytes_per_frame
        end_frame = start_frame + (len(piece) // bytes_per_frame)
        start_ms = int((start_frame / framerate) * 1000)
        end_ms = int((end_frame / framerate) * 1000)

        out = io.BytesIO()
        with wave.open(out, "wb") as wav_out:
            wav_out.setnchannels(nchannels)
            wav_out.setsampwidth(sampwidth)
            wav_out.setframerate(framerate)
            wav_out.writeframes(piece)

        chunks.append(
            AudioChunk(
                chunk_id=chunk_id,
                start_ms=start_ms,
                end_ms=end_ms,
                audio_bytes=out.getvalue(),
            )
        )

        offset += bytes_per_chunk
        chunk_id += 1

    return chunks


class OrderedTranscriptMerger:
    """Keeps partial transcripts ordered and merged by chunk id."""

    def __init__(self) -> None:
        self._pending: Dict[int, str] = {}
        self._next_chunk_id = 0
        self._merged_segments: List[str] = []

    def add(self, chunk_id: int, text: str) -> List[str]:
        if chunk_id < 0:
            raise ValueError("chunk_id must be >= 0")
        self._pending[chunk_id] = (text or "").strip()

        released: List[str] = []
        while self._next_chunk_id in self._pending:
            segment = self._pending.pop(self._next_chunk_id)
            if segment:
                self._merged_segments.append(segment)
                released.append(segment)
            self._next_chunk_id += 1

        return released

    def skip(self, chunk_id: int) -> List[str]:
        return self.add(chunk_id, "")

    @property
    def transcript(self) -> str:
        return " ".join(self._merged_segments).strip()

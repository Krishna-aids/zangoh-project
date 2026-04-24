import base64
from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.api import server


class _SuccessPipeline:
    async def process_audio_with_transcript(self, audio_bytes: bytes, **kwargs):
        return (
            b"mock-mp3-audio",
            SimpleNamespace(user_input="What is your return policy?", agent_response="30-day return policy."),
            1234,
        )


class _ErrorPipeline:
    async def process_audio_with_transcript(self, audio_bytes: bytes, **kwargs):
        raise RuntimeError("STT service temporarily unavailable")


def test_chat_audio_success_returns_transcript_and_base64_audio():
    with TestClient(server.app) as client:
        server.pipeline = _SuccessPipeline()
        response = client.post(
            "/chat/audio",
            files={"audio": ("test.wav", b"fake-wav", "audio/wav")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["transcript"]["user_input"] == "What is your return policy?"
    assert data["transcript"]["agent_response"] == "30-day return policy."
    assert data["processing_time_ms"] == 1234
    assert base64.b64decode(data["audio_response"]) == b"mock-mp3-audio"


def test_chat_audio_empty_input_returns_standardized_error_shape():
    with TestClient(server.app) as client:
        server.pipeline = _SuccessPipeline()
        response = client.post(
            "/chat/audio",
            files={"audio": ("empty.wav", b"", "audio/wav")},
        )

    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert data["error"] == "Empty audio file"
    assert data["transcript"]["user_input"] is None
    assert data["transcript"]["agent_response"] is None
    assert isinstance(data["processing_time_ms"], int)


def test_chat_audio_processing_failure_returns_standardized_error_shape():
    with TestClient(server.app) as client:
        server.pipeline = _ErrorPipeline()
        response = client.post(
            "/chat/audio",
            files={"audio": ("test.wav", b"fake-wav", "audio/wav")},
        )

    assert response.status_code == 500
    data = response.json()
    assert data["success"] is False
    assert "STT service temporarily unavailable" in data["error"]
    assert data["transcript"]["user_input"] is None
    assert data["transcript"]["agent_response"] is None
    assert isinstance(data["processing_time_ms"], int)

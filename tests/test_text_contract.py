from fastapi.testclient import TestClient

from src.api import server


class _TextPipeline:
    def __init__(self):
        self.calls = 0
        self.llm_agent = object()
        self.last_text = None
        self.last_kwargs = None

    async def process_text_with_timing(self, text: str, **kwargs):
        self.calls += 1
        self.last_text = text
        self.last_kwargs = kwargs
        return ("You can return items within 30 days.", False, 321)


def test_chat_text_returns_response_from_single_llm_call():
    fake_pipeline = _TextPipeline()

    with TestClient(server.app) as client:
        server.pipeline = fake_pipeline
        response = client.post(
            "/chat/text",
            json={"text": "What is the return policy?", "parameters": {}},
        )

    assert response.status_code == 200
    assert response.json() == {
        "response_text": "You can return items within 30 days.",
        "audio_available": False,
        "processing_time_ms": 321,
    }
    assert fake_pipeline.calls == 1
    assert fake_pipeline.last_text == "What is the return policy?"
    assert fake_pipeline.last_kwargs["generate_audio"] is False
    assert fake_pipeline.last_kwargs["allow_tts_failure"] is True


def test_chat_text_accepts_audio_generation_aliases():
    fake_pipeline = _TextPipeline()

    with TestClient(server.app) as client:
        server.pipeline = fake_pipeline
        response = client.post(
            "/chat/text",
            json={
                "text": "Can you answer and also synthesize audio?",
                "parameters": {"include_audio": "true"},
            },
        )

    assert response.status_code == 200
    assert fake_pipeline.calls == 1
    assert fake_pipeline.last_kwargs["generate_audio"] is True
    assert fake_pipeline.last_kwargs["allow_tts_failure"] is True

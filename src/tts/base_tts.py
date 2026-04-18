"""
Base Text-to-Speech (TTS) Interface

This module defines the abstract base class for Text-to-Speech implementations.
Students should implement the concrete TTS class by inheriting from this base class.

Recommended implementation: ElevenLabs API (free tier available)
Alternative options: OpenTTS, gTTS, Azure Speech Services, or any other TTS service
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import io
import os
from typing import AsyncIterator

import edge_tts


class BaseTTS(ABC):
    """
    Abstract base class for Text-to-Speech implementations.
    
    This class defines the interface that all TTS implementations must follow.
    Students should inherit from this class and implement the abstract methods.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the TTS service.
        
        Args:
            config: Configuration dictionary containing API keys, voice settings, etc.
                   Example: {"api_key": "your_api_key", "voice_id": "voice_id", "model": "eleven_turbo_v2"}
        """
        self.config = config or {}
        self.is_initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the TTS service (setup API clients, load models, etc.).
        This method should be called before using the TTS service.
        
        Raises:
            Exception: If initialization fails
        """
        pass
    
    @abstractmethod
    async def synthesize(self, text: str, **kwargs) -> bytes:
        """
        Convert text to speech audio bytes.
        
        Args:
            text: Text to convert to speech
            **kwargs: Additional parameters specific to the TTS implementation
                     (e.g., voice_id, speed, pitch, format)
        
        Returns:
            bytes: Audio data as bytes (typically MP3 or WAV format)
            
        Raises:
            Exception: If synthesis fails
        """
        pass
    
    @abstractmethod
    async def synthesize_stream(self, text: str, **kwargs) -> io.BytesIO:
        """
        Convert text to speech with streaming support.
        
        Args:
            text: Text to convert to speech
            **kwargs: Additional parameters for streaming
        
        Returns:
            io.BytesIO: Streaming audio data
            
        Raises:
            Exception: If streaming synthesis fails
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """
        Cleanup resources (close connections, free memory, etc.).
        This method should be called when the TTS service is no longer needed.
        """
        pass
    
    def is_ready(self) -> bool:
        """
        Check if the TTS service is ready to use.
        
        Returns:
            bool: True if ready, False otherwise
        """
        return self.is_initialized


class TTSService(BaseTTS):
    """
    Generic TTS implementation template.
    
    Students should complete this implementation using their chosen TTS service or pretrained model.
    
    API-based options:
    - ElevenLabs API (free tier, high quality): pip install elevenlabs
    - OpenAI TTS API (high quality): included in openai package
    - Azure Speech Services: pip install azure-cognitiveservices-speech
    - Google Cloud Text-to-Speech: pip install google-cloud-texttospeech
    - Amazon Polly: pip install boto3
    
    Pretrained model options (local inference):
    - Coqui TTS: pip install TTS (various pretrained models: Tacotron2, VITS, etc.)
    - Parler TTS: pip install parler-tts (Hugging Face pretrained models)
    - Bark: pip install bark (Suno's generative audio model)
    - Edge TTS: pip install edge-tts (free Microsoft voices, no training needed)
    - Festival: pip install festival (classic speech synthesis)
    - eSpeak: pip install espeak (lightweight, many languages)
    - Piper: pip install piper-tts (fast neural TTS)
    
    Input: text (str) - Text to convert to speech
    Output: audio_bytes (bytes) - Audio data (typically MP3 or WAV)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.client = None
        self.voice_id: str = ""
        self.model = None
        self.rate = self.config.get("rate", "+0%")
        self.volume = self.config.get("volume", "+0%")
        self.timeout_seconds = float(self.config.get("timeout_seconds", 20.0))
    
    async def initialize(self) -> None:
        """
        TODO: Implement TTS service initialization.
        
        Steps:
        1. Get API key from config (if using API service)
        2. Create client instance or load model
        3. Set voice parameters (voice_id, model, etc.)
        4. Set is_initialized to True
        5. Optionally test the connection
        
        Example for API-based services:
        ```python
        api_key = self.config.get("api_key")
        if not api_key:
            raise ValueError("API key not provided")
        self.client = YourTTSClient(api_key)
        self.voice_id = self.config.get("voice_id", "default_voice")
        ```
        
        Example for pretrained models (local):
        ```python
        # Coqui TTS
        model_name = self.config.get("model", "tts_models/en/ljspeech/tacotron2-DDC")
        self.client = TTS(model_name)
        
        # Parler TTS (Hugging Face)
        model_name = "parler-tts/parler_tts_mini_v0.1"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.client = ParlerTTSForConditionalGeneration.from_pretrained(model_name)
        
        # Bark
        # No explicit initialization needed, models auto-download
        self.client = "bark"  # Placeholder for bark usage
        
        # Edge TTS
        # No initialization needed, just import
        self.client = "edge_tts"
        ```
        """
        self.voice_id = str(
            self.config.get("voice")
            or self.config.get("voice_id")
            or os.getenv("EDGE_TTS_VOICE", "en-US-AriaNeural")
        )
        if not self.voice_id:
            raise ValueError("TTS voice must be provided via config or EDGE_TTS_VOICE.")
        self.is_initialized = True
    
    async def synthesize(self, text: str, **kwargs) -> bytes:
        """
        TODO: Implement text-to-speech synthesis.
        
        Input: text (str) - Text to convert to speech
        Output: bytes - Audio data in chosen format (MP3, WAV, etc.)
        
        Steps:
        1. Check if service is initialized
        2. Validate input text
        3. Prepare synthesis parameters
        4. Call TTS API/model
        5. Return audio bytes
        6. Handle errors appropriately
        
        Example implementations:
        
        For ElevenLabs:
        ```python
        audio_stream = self.client.text_to_speech.stream(
            text=text,
            voice_id=self.voice_id,
            model="eleven_turbo_v2_5"
        )
        audio_bytes = b""
        for chunk in audio_stream:
            audio_bytes += chunk
        return audio_bytes
        ```
        
        For OpenAI:
        ```python
        response = self.client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        return response.content
        ```
        
        For Coqui TTS (pretrained model):
        ```python
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
            self.client.tts_to_file(text=text, file_path=temp_file.name)
            with open(temp_file.name, "rb") as f:
                return f.read()
        ```
        
        For Parler TTS (Hugging Face):
        ```python
        import scipy.io.wavfile
        import io
        inputs = self.tokenizer(text, return_tensors="pt")
        audio_values = self.client.generate(**inputs)
        audio_np = audio_values.cpu().numpy().squeeze()
        
        # Convert to bytes
        buffer = io.BytesIO()
        scipy.io.wavfile.write(buffer, 22050, audio_np)
        return buffer.getvalue()
        ```
        
        For Bark (generative model):
        ```python
        import bark
        import scipy.io.wavfile
        import io
        audio_array = bark.generate_audio(text)
        
        # Convert to bytes
        buffer = io.BytesIO()
        scipy.io.wavfile.write(buffer, bark.SAMPLE_RATE, audio_array)
        return buffer.getvalue()
        ```
        
        For Edge TTS (free Microsoft voices):
        ```python
        import edge_tts
        import asyncio
        communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
        audio_bytes = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes += chunk["data"]
        return audio_bytes
        ```
        """
        if not self.is_ready():
            raise RuntimeError("TTS service not initialized")
        
        if not text.strip():
            raise ValueError("Text cannot be empty")
        
        audio_bytes = bytearray()
        async for chunk in self.stream_synthesize(text, **kwargs):
            audio_bytes.extend(chunk)
        return bytes(audio_bytes)
    
    async def synthesize_stream(self, text: str, **kwargs) -> io.BytesIO:
        """
        TODO: Implement streaming text-to-speech synthesis.
        
        Input: text (str) - Text to convert to speech
        Output: io.BytesIO - Streaming audio data buffer
        
        This method is useful for real-time applications where you want to
        start playing audio while it's still being generated.
        
        Example:
        ```python
        audio_buffer = io.BytesIO()
        # Stream audio chunks and write to buffer
        for chunk in self.client.stream(text):
            audio_buffer.write(chunk)
        audio_buffer.seek(0)
        return audio_buffer
        ```
        """
        if not self.is_ready():
            raise RuntimeError("TTS service not initialized")

        audio_buffer = io.BytesIO()
        async for chunk in self.stream_synthesize(text, **kwargs):
            audio_buffer.write(chunk)
        audio_buffer.seek(0)
        return audio_buffer
    
    async def cleanup(self) -> None:
        """
        TODO: Implement cleanup logic.
        
        Steps:
        1. Close any open connections
        2. Clear client instance and models
        3. Set is_initialized to False
        """
        self.client = None
        self.is_initialized = False

    async def stream_synthesize(self, text: str, **kwargs) -> AsyncIterator[bytes]:
        """
        Edge TTS streaming generator that yields audio chunks.
        """
        if not self.is_ready():
            raise RuntimeError("TTS service not initialized")
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        voice = str(kwargs.get("voice") or self.voice_id)
        rate = kwargs.get("rate", self.rate)
        volume = kwargs.get("volume", self.volume)

        communicate = edge_tts.Communicate(
            text=text.strip(),
            voice=voice,
            rate=rate,
            volume=volume,
        )
        async for event in communicate.stream():
            if event.get("type") == "audio":
                data = event.get("data", b"")
                if data:
                    yield data
    
    async def get_available_voices(self) -> List[Dict[str, Any]]:
        """
        TODO: Get list of available voices (if supported by your TTS service).
        
        Returns:
            List of available voices with their metadata
            
        Example:
        ```python
        if hasattr(self.client, 'voices'):
            voices = self.client.voices.get_all()
            return [{"voice_id": v.voice_id, "name": v.name} for v in voices]
        return []
        ```
        """
        if not self.is_ready():
            raise RuntimeError("TTS service not initialized")
        
        # TODO: Implement voice listing if supported
        return []

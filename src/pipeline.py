"""
Audio Customer Support Agent Pipeline

This module orchestrates the complete STT -> LLM -> TTS pipeline.
Students should complete the implementation to connect all components.
"""

import asyncio
import logging
from time import perf_counter
from typing import Optional, Dict, Any, Tuple, AsyncIterable, AsyncIterator, List
from dataclasses import dataclass

from src.stt.base_stt import BaseSTT, STTService
from src.llm.agent import BaseAgent, CustomerSupportAgent
from src.tts.base_tts import BaseTTS, TTSService
from src.utils.audio_chunking import split_wav_audio_chunks, OrderedTranscriptMerger


def _as_bool(value: Any, default: bool = False) -> bool:
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


@dataclass
class PipelineConfig:
    """Configuration for the audio support pipeline."""
    stt_config: Dict[str, Any]
    llm_config: Dict[str, Any]
    tts_config: Dict[str, Any]
    enable_logging: bool = True


@dataclass
class TranscriptData:
    """Transcript metadata for a processed request."""

    user_input: str
    agent_response: str


class AudioSupportPipeline:
    """
    Main pipeline class that orchestrates STT -> LLM -> TTS flow.
    
    This class manages the entire audio processing pipeline for customer support.
    Students should complete the implementation to make it fully functional.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the audio support pipeline.
        
        Args:
            config: Pipeline configuration containing settings for all components
        """
        self.config = config
        self.stt: Optional[BaseSTT] = None
        self.llm_agent: Optional[BaseAgent] = None
        self.tts: Optional[BaseTTS] = None
        self.is_initialized = False
        self.initialization_errors: Dict[str, str] = {}
        
        if config.enable_logging:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.CRITICAL)
    
    async def initialize(self) -> None:
        """
        TODO: Initialize all pipeline components.
        
        Steps:
        1. Initialize STT service
        2. Initialize LLM agent
        3. Initialize TTS service
        4. Verify all components are ready
        
        Raises:
            Exception: If any component fails to initialize
        """
        try:
            self.logger.info("Initializing Audio Support Pipeline...")

            self.initialization_errors = {}

            try:
                self.logger.info("Initializing STT service...")
                self.stt = STTService(self.config.stt_config)
                await self.stt.initialize()
            except Exception as exc:
                self.stt = None
                self.initialization_errors["stt"] = str(exc)
                self.logger.warning("STT initialization failed: %s", exc)

            try:
                self.logger.info("Initializing LLM agent...")
                self.llm_agent = CustomerSupportAgent(self.config.llm_config)
                await self.llm_agent.initialize()
            except Exception as exc:
                self.llm_agent = None
                self.initialization_errors["llm"] = str(exc)
                self.logger.error("LLM initialization failed: %s", exc)

            try:
                self.logger.info("Initializing TTS service...")
                self.tts = TTSService(self.config.tts_config)
                await self.tts.initialize()
            except Exception as exc:
                self.tts = None
                self.initialization_errors["tts"] = str(exc)
                self.logger.warning("TTS initialization failed: %s", exc)

            if not (self.llm_agent and self.llm_agent.is_initialized):
                raise RuntimeError(
                    self.initialization_errors.get(
                        "llm", "LLM agent failed to initialize"
                    )
                )

            self.is_initialized = True
            if self.initialization_errors:
                failed_components = ", ".join(sorted(self.initialization_errors))
                self.logger.warning(
                    "Pipeline initialized in degraded mode. Unavailable components: %s",
                    failed_components,
                )
            else:
                self.logger.info("Pipeline initialized successfully!")
            
        except Exception as e:
            self.logger.error(f"Pipeline initialization failed: {str(e)}")
            await self.cleanup()
            raise
    
    async def process_audio(self, audio_bytes: bytes, **kwargs) -> bytes:
        """
        TODO: Process audio input through the complete pipeline.
        
        This is the main method that handles the STT -> LLM -> TTS flow.
        
        Args:
            audio_bytes: Input audio data
            **kwargs: Additional parameters for processing
            
        Returns:
            bytes: Response audio data
            
        Raises:
            RuntimeError: If pipeline is not initialized
            Exception: If processing fails at any stage
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        try:
            if not self.stt or not self.llm_agent or not self.tts:
                raise RuntimeError("Pipeline components are unavailable")

            self.logger.info("Converting speech to text...")
            text_input = await self.stt.transcribe(audio_bytes, **kwargs)
            self.logger.info(f"Transcribed text: {text_input}")
            
            self.logger.info("Processing query with LLM agent...")
            agent_response = await self.llm_agent.process_query(text_input, **kwargs)
            self.logger.info(f"Agent response: {agent_response}")
            
            self.logger.info("Converting response to speech...")
            response_audio = await self.tts.synthesize(agent_response, **kwargs)
            self.logger.info("Audio response generated successfully")
            
            return response_audio
            
        except Exception as e:
            self.logger.error(f"Pipeline processing failed: {str(e)}")
            raise

    async def process_audio_with_transcript(
        self, audio_bytes: bytes, **kwargs
    ) -> Tuple[bytes, TranscriptData, int]:
        """Process audio and return response audio, transcript data, and timing."""
        start_time = perf_counter()
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        try:
            if not self.stt or not self.llm_agent or not self.tts:
                raise RuntimeError("Pipeline components are unavailable")

            self.logger.info("Converting speech to text...")
            text_input = await self.stt.transcribe(audio_bytes, **kwargs)
            normalized_user_input = str(text_input).strip()
            self.logger.info(f"Transcribed text: {normalized_user_input}")

            self.logger.info("Processing query with LLM agent...")
            agent_response = await self.llm_agent.process_query(normalized_user_input, **kwargs)
            normalized_agent_response = str(agent_response).strip()
            self.logger.info(f"Agent response: {normalized_agent_response}")

            self.logger.info("Converting response to speech...")
            response_audio = await self.tts.synthesize(normalized_agent_response, **kwargs)
            self.logger.info("Audio response generated successfully")

            transcript_data = self._create_transcript_data(
                user_input=normalized_user_input,
                agent_response=normalized_agent_response,
            )
            processing_time_ms = int((perf_counter() - start_time) * 1000)

            return response_audio, transcript_data, processing_time_ms

        except Exception as e:
            self.logger.error(f"Pipeline processing failed: {str(e)}")
            raise
    
    async def process_text(self, text_input: str, **kwargs) -> Tuple[str, bytes]:
        """
        TODO: Process text input (useful for testing without STT).
        
        Args:
            text_input: Text query from user
            **kwargs: Additional parameters
            
        Returns:
            Tuple[str, bytes]: (agent_response_text, response_audio)
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        try:
            if not self.llm_agent:
                raise RuntimeError("LLM agent is unavailable")

            generate_audio = _as_bool(kwargs.pop("generate_audio", False))
            allow_tts_failure = _as_bool(kwargs.pop("allow_tts_failure", True))

            self.logger.info(f"Processing text query: {text_input}")
            agent_response = await self.llm_agent.process_query(text_input, **kwargs)
            response_audio = b""

            if generate_audio:
                if not self.tts:
                    if not allow_tts_failure:
                        raise RuntimeError("TTS service is unavailable")
                    self.logger.warning("Skipping TTS synthesis because TTS is unavailable.")
                else:
                    try:
                        response_audio = await self.tts.synthesize(agent_response, **kwargs)
                    except Exception as exc:
                        if not allow_tts_failure:
                            raise
                        self.logger.warning(
                            "Text response generated, but TTS synthesis failed: %s",
                            exc,
                        )

            return agent_response, response_audio
            
        except Exception as e:
            self.logger.error(f"Text processing failed: {str(e)}")
            raise

    async def process_text_with_timing(
        self, text: str, **kwargs
    ) -> Tuple[str, bool, int]:
        """Process text and return response text, audio availability, and timing."""
        start_time = perf_counter()
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        try:
            if not self.llm_agent:
                raise RuntimeError("LLM agent is unavailable")

            generate_audio = _as_bool(kwargs.pop("generate_audio", False))
            allow_tts_failure = _as_bool(kwargs.pop("allow_tts_failure", True))

            self.logger.info(f"Processing text query: {text}")
            agent_response = await self.llm_agent.process_query(text, **kwargs)
            audio_available = False

            if generate_audio:
                if not self.tts:
                    if not allow_tts_failure:
                        raise RuntimeError("TTS service is unavailable")
                    self.logger.warning("Skipping TTS synthesis because TTS is unavailable.")
                else:
                    try:
                        audio_bytes = await self.tts.synthesize(agent_response, **kwargs)
                        audio_available = bool(audio_bytes)
                    except Exception as exc:
                        if not allow_tts_failure:
                            raise
                        self.logger.warning(
                            "Text response generated, but TTS synthesis failed: %s",
                            exc,
                        )

            processing_time_ms = int((perf_counter() - start_time) * 1000)
            return agent_response, audio_available, processing_time_ms

        except Exception as e:
            self.logger.error(f"Text processing failed: {str(e)}")
            raise

    def _create_transcript_data(
        self, user_input: str, agent_response: str
    ) -> TranscriptData:
        """Create transcript data payload for callers."""
        return TranscriptData(
            user_input=str(user_input).strip(),
            agent_response=str(agent_response).strip(),
        )

    async def process_audio_stream(
        self,
        audio_stream: AsyncIterable[bytes],
        chunk_seconds: float = 2.0,
        **kwargs,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Streaming pipeline:
        audio chunks -> incremental STT -> incremental LLM -> incremental TTS audio.
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        if not self.stt or not self.llm_agent or not self.tts:
            raise RuntimeError("Pipeline components are unavailable")

        merger = OrderedTranscriptMerger()
        next_chunk_id = 0

        async for incoming_chunk in audio_stream:
            if not incoming_chunk:
                continue

            # Step 1: split incoming audio to short WAV windows for low-latency STT.
            chunks = split_wav_audio_chunks(incoming_chunk, chunk_seconds=chunk_seconds)
            for chunk in chunks:
                chunk_id = next_chunk_id
                next_chunk_id += 1
                try:
                    transcript = await self.stt.transcribe(chunk.audio_bytes, **kwargs)
                    released = merger.add(chunk_id, transcript)
                except Exception as exc:
                    released = merger.skip(chunk_id)
                    yield {
                        "type": "error",
                        "stage": "stt",
                        "chunk_id": chunk_id,
                        "message": str(exc),
                    }
                if not released:
                    continue

                partial_text = " ".join(released).strip()
                if not partial_text:
                    continue

                yield {
                    "type": "stt.partial",
                    "chunk_id": chunk_id,
                    "text": partial_text,
                    "full_transcript": merger.transcript,
                }

                # Step 2: stream LLM output tokens for the latest available transcript piece.
                llm_tokens: List[str] = []
                stream_query = getattr(self.llm_agent, "stream_query", None)
                try:
                    if callable(stream_query):
                        async for token in stream_query(partial_text, **kwargs):
                            llm_tokens.append(token)
                            yield {
                                "type": "llm.token",
                                "chunk_id": chunk_id,
                                "token": token,
                            }
                    else:
                        text = await self.llm_agent.process_query(partial_text, **kwargs)
                        llm_tokens.append(text)
                        yield {
                            "type": "llm.token",
                            "chunk_id": chunk_id,
                            "token": text,
                        }
                except Exception as exc:
                    yield {
                        "type": "error",
                        "stage": "llm",
                        "chunk_id": chunk_id,
                        "message": str(exc),
                    }
                    continue

                llm_text = "".join(llm_tokens).strip()
                if not llm_text:
                    continue

                # Step 3: stream TTS bytes for immediate playback.
                tts_stream = getattr(self.tts, "stream_synthesize", None)
                try:
                    if callable(tts_stream):
                        async for audio_piece in tts_stream(llm_text, **kwargs):
                            yield {
                                "type": "tts.audio",
                                "chunk_id": chunk_id,
                                "audio": audio_piece,
                            }
                    else:
                        buffered_audio = await self.tts.synthesize(llm_text, **kwargs)
                        yield {
                            "type": "tts.audio",
                            "chunk_id": chunk_id,
                            "audio": buffered_audio,
                        }
                except Exception as exc:
                    yield {
                        "type": "error",
                        "stage": "tts",
                        "chunk_id": chunk_id,
                        "message": str(exc),
                    }
    
    async def health_check(self) -> Dict[str, bool]:
        """
        TODO: Check the health status of all pipeline components.
        
        Returns:
            Dict[str, bool]: Status of each component
        """
        return {
            "pipeline_initialized": self.is_initialized,
            "stt_ready": self.stt.is_ready() if self.stt else False,
            "llm_ready": self.llm_agent.is_initialized if self.llm_agent else False,
            "tts_ready": self.tts.is_ready() if self.tts else False,
        }
    
    async def cleanup(self) -> None:
        """
        TODO: Cleanup all pipeline resources.
        
        This method should be called when the pipeline is no longer needed.
        """
        self.logger.info("Cleaning up pipeline resources...")
        
        try:
            # TODO: Cleanup all components
            if self.stt:
                await self.stt.cleanup()
            if self.llm_agent:
                await self.llm_agent.cleanup()
            if self.tts:
                await self.tts.cleanup()
                
            self.stt = None
            self.llm_agent = None
            self.tts = None
            self.is_initialized = False
            self.initialization_errors = {}
            
            self.logger.info("Pipeline cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {str(e)}")
            raise


async def create_pipeline(
    stt_config: Dict[str, Any],
    llm_config: Dict[str, Any],
    tts_config: Dict[str, Any],
    enable_logging: bool = True
) -> AudioSupportPipeline:
    """
    TODO: Factory function to create and initialize a pipeline.
    
    Args:
        stt_config: STT configuration
        llm_config: LLM configuration  
        tts_config: TTS configuration
        enable_logging: Whether to enable logging
        
    Returns:
        AudioSupportPipeline: Initialized pipeline instance
    """
    config = PipelineConfig(
        stt_config=stt_config,
        llm_config=llm_config,
        tts_config=tts_config,
        enable_logging=enable_logging
    )
    
    pipeline = AudioSupportPipeline(config)
    await pipeline.initialize()
    
    return pipeline


if __name__ == "__main__":
    """
    Example usage of the pipeline.
    Students can use this for testing their implementation.
    """
    async def main():
        # TODO: Example configuration - replace with your chosen services
        stt_config = {
            # Configure your chosen STT service
            "api_key": "your_stt_api_key",
            "model": "your_chosen_model"
        }
        
        llm_config = {
            # Configure your chosen LLM service
            "api_key": "your_llm_api_key",
            "model": "your_chosen_model",
            "temperature": 0.7
        }
        
        tts_config = {
            # Configure your chosen TTS service
            "api_key": "your_tts_api_key",
            "voice_id": "your_chosen_voice"
        }
        
        # TODO: Create and test pipeline
        # pipeline = await create_pipeline(stt_config, llm_config, tts_config)
        
        # TODO: Test with text input
        # response_text, response_audio = await pipeline.process_text("Hello, I need help with my order")
        # print(f"Response: {response_text}")
        
        # TODO: Cleanup
        # await pipeline.cleanup()
        
        print("Pipeline example completed. Implement the TODOs to make it functional!")
    
    asyncio.run(main())

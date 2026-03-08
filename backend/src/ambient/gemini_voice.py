"""
Gemini Voice Provider — STT + TTS via Google Gemini API
Uses gemini-2.5-flash for speech-to-text (multimodal audio input)
and gemini-2.5-flash-preview-tts for text-to-speech (audio output modality).

This is the cloud alternative to the traditional local models
(faster-whisper for STT, Piper TTS for TTS).
"""

import io
import wave
import struct
import base64
import time
import asyncio
import numpy as np
from typing import Optional, Dict, Generator
from concurrent.futures import ThreadPoolExecutor


class GeminiSTT:
    """
    Speech-to-Text using Gemini's multimodal audio understanding.
    Sends audio to Gemini and receives transcription text.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        from google import genai
        from google.genai import types

        self._client = genai.Client(api_key=api_key)
        self._types = types
        self._model_name = model_name
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="gemini-stt")

        # Stats
        self._total_transcriptions = 0
        self._total_audio_seconds = 0.0
        self._total_processing_seconds = 0.0

    async def transcribe(self, audio: np.ndarray,
                         language: Optional[str] = None,
                         quality_mode: bool = False) -> Dict:
        """
        Transcribe audio using Gemini.

        Args:
            audio: int16 PCM audio at 16kHz
            language: Optional language hint (e.g., "en", "hi")
            quality_mode: If True, request detailed transcription

        Returns:
            {"text": str, "language": str, "segments": [], "duration": float,
             "confidence": float, "processing_time_s": float}
        """
        if len(audio) == 0:
            return {"text": "", "language": "", "segments": [],
                    "duration": 0, "confidence": 0, "processing_time_s": 0}

        duration = len(audio) / 16000.0

        # Convert PCM int16 to WAV bytes for Gemini
        wav_bytes = self._pcm_to_wav(audio)

        t0 = time.time()

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self._executor, self._transcribe_sync, wav_bytes, language, quality_mode
        )

        elapsed = time.time() - t0
        self._total_transcriptions += 1
        self._total_audio_seconds += duration
        self._total_processing_seconds += elapsed

        result["duration"] = round(duration, 2)
        result["processing_time_s"] = round(elapsed, 2)
        return result

    def _transcribe_sync(self, wav_bytes: bytes,
                         language: Optional[str],
                         quality_mode: bool) -> Dict:
        """Synchronous Gemini transcription call."""
        lang_hint = f" The audio is in {language}." if language else ""
        prompt = (
            f"Transcribe the following audio accurately.{lang_hint} "
            "Return ONLY the transcribed text, nothing else. "
            "If the audio is silent or unintelligible, return an empty string."
        )

        audio_part = self._types.Part.from_bytes(
            data=wav_bytes,
            mime_type="audio/wav",
        )

        config = self._types.GenerateContentConfig(
            max_output_tokens=4096,
            temperature=0.0,
        )

        response = self._client.models.generate_content(
            model=self._model_name,
            contents=[audio_part, prompt],
            config=config,
        )

        text = ""
        if response and response.text:
            text = response.text.strip()
            # Clean up any markup Gemini might add
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1]
            if text.startswith("'") and text.endswith("'"):
                text = text[1:-1]

        # Estimate confidence from response
        confidence = 0.9 if text else 0.0

        detected_lang = language or "en"

        return {
            "text": text,
            "language": detected_lang,
            "language_probability": 0.95 if text else 0.0,
            "segments": [{"start": 0.0, "end": round(len(wav_bytes) / 32000, 2),
                          "text": text}] if text else [],
            "confidence": confidence,
        }

    @staticmethod
    def _pcm_to_wav(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
        """Convert PCM int16 numpy array to WAV bytes."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())
        return buf.getvalue()

    def get_stats(self) -> Dict:
        rtf = (self._total_processing_seconds / self._total_audio_seconds
               if self._total_audio_seconds > 0 else 0)
        return {
            "model_size": "gemini",
            "device": "cloud",
            "model_name": self._model_name,
            "total_transcriptions": self._total_transcriptions,
            "total_audio_seconds": round(self._total_audio_seconds, 1),
            "total_processing_seconds": round(self._total_processing_seconds, 1),
            "real_time_factor": round(rtf, 2),
        }


class GeminiTTS:
    """
    Text-to-Speech using Gemini's audio output modality.
    Sends text and receives synthesized audio.

    Available voices: Aoede, Charon, Fenrir, Kore, Puck, etc.
    """

    AVAILABLE_VOICES = [
        "Aoede", "Charon", "Fenrir", "Kore", "Puck",
        "Leda", "Orus", "Zephyr",
    ]
    SAMPLE_RATE = 24000  # Gemini outputs 24kHz audio
    DEFAULT_VOICE = "Kore"

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-preview-tts",
                 voice: str = None):
        from google import genai
        from google.genai import types

        self._client = genai.Client(api_key=api_key)
        self._types = types
        self._model_name = model_name
        self.voice_name = voice or self.DEFAULT_VOICE
        self._available = True
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="gemini-tts")

        # Stats
        self._total_syntheses = 0
        self._total_chars = 0
        self._total_audio_seconds = 0.0

    @property
    def is_available(self) -> bool:
        return self._available

    def synthesize(self, text: str) -> Optional[np.ndarray]:
        """Synthesize text to audio array (int16 PCM)."""
        if not text.strip():
            return None
        wav_bytes = self.synthesize_to_wav(text)
        if not wav_bytes:
            return None
        # Parse WAV to numpy
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16)
        return audio

    def synthesize_to_wav(self, text: str) -> Optional[bytes]:
        """Synthesize text to WAV bytes."""
        if not text.strip():
            return None

        try:
            config = self._types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=self._types.SpeechConfig(
                    voice_config=self._types.VoiceConfig(
                        prebuilt_voice_config=self._types.PrebuiltVoiceConfig(
                            voice_name=self.voice_name,
                        )
                    )
                ),
            )

            response = self._client.models.generate_content(
                model=self._model_name,
                contents=f"Say the following text naturally: {text}",
                config=config,
            )

            # Extract audio data from response
            if (response and response.candidates
                    and response.candidates[0].content
                    and response.candidates[0].content.parts):
                for part in response.candidates[0].content.parts:
                    if (hasattr(part, "inline_data") and part.inline_data
                            and part.inline_data.mime_type
                            and "audio" in part.inline_data.mime_type):
                        audio_data = part.inline_data.data
                        wav_bytes = self._ensure_wav(audio_data,
                                                     part.inline_data.mime_type)
                        self._total_syntheses += 1
                        self._total_chars += len(text)
                        if wav_bytes:
                            # Estimate audio duration
                            try:
                                buf = io.BytesIO(wav_bytes)
                                with wave.open(buf, "rb") as wf:
                                    dur = wf.getnframes() / wf.getframerate()
                                    self._total_audio_seconds += dur
                            except Exception:
                                pass
                        return wav_bytes

            print("  ⚠ Gemini TTS: No audio in response")
            return None

        except Exception as e:
            print(f"  ⚠ Gemini TTS error: {e}")
            return None

    async def synthesize_to_wav_async(self, text: str) -> Optional[bytes]:
        """Async wrapper for synthesize_to_wav."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, self.synthesize_to_wav, text
        )

    def synthesize_stream(self, text: str) -> Generator[bytes, None, None]:
        """
        Stream synthesis. Gemini doesn't natively stream audio chunks,
        so we synthesize the full audio and yield it in chunks.
        """
        wav_bytes = self.synthesize_to_wav(text)
        if not wav_bytes:
            return

        # Skip WAV header (44 bytes) and yield PCM chunks
        pcm_data = wav_bytes[44:]
        chunk_size = self.SAMPLE_RATE * 2  # 1 second of int16 audio
        for i in range(0, len(pcm_data), chunk_size):
            yield pcm_data[i:i + chunk_size]

    @staticmethod
    def _ensure_wav(audio_data: bytes, mime_type: str) -> Optional[bytes]:
        """Convert audio data to WAV format if needed."""
        # If already WAV, return as-is
        if audio_data[:4] == b"RIFF":
            return audio_data

        # If raw PCM (from Gemini), wrap in WAV header
        # Gemini typically returns 24kHz 16-bit mono PCM
        sample_rate = 24000
        channels = 1
        sample_width = 2

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)
        return buf.getvalue()

    def set_voice(self, voice_name: str):
        """Change the TTS voice."""
        if voice_name in self.AVAILABLE_VOICES:
            self.voice_name = voice_name
        else:
            print(f"  ⚠ Unknown voice '{voice_name}', "
                  f"available: {self.AVAILABLE_VOICES}")

    def get_stats(self) -> Dict:
        return {
            "available": self._available,
            "voice": self.voice_name,
            "model_name": self._model_name,
            "sample_rate": self.SAMPLE_RATE,
            "available_voices": self.AVAILABLE_VOICES,
            "total_syntheses": self._total_syntheses,
            "total_chars": self._total_chars,
            "total_audio_seconds": round(self._total_audio_seconds, 1),
        }

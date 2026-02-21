"""Voice Activity Detection with audio buffering for sentence-level chunks."""

import numpy as np
import torch
import config


class AudioSegmenter:
    """
    Buffers incoming audio, uses Silero VAD to detect speech boundaries,
    and yields complete utterances for processing.
    """

    def __init__(self, model_manager):
        self.vad_model = model_manager.vad_model
        self.sample_rate = config.SAMPLE_RATE
        self.threshold = config.VAD_THRESHOLD
        self.min_speech_samples = int(config.MIN_SPEECH_MS * self.sample_rate / 1000)
        self.max_speech_samples = int(config.MAX_SPEECH_MS * self.sample_rate / 1000)
        self.silence_samples = int(config.SILENCE_AFTER_SPEECH_MS * self.sample_rate / 1000)
        self.reset()

    def reset(self):
        """Reset all buffers."""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.speech_buffer = np.array([], dtype=np.float32)
        self.is_speaking = False
        self.silence_counter = 0
        self.vad_model.reset_states()

    def process_chunk(self, audio_bytes: bytes) -> list:
        """
        Process incoming audio bytes and return list of complete utterances.

        Args:
            audio_bytes: Raw PCM16 audio bytes at 16kHz mono

        Returns:
            List of float32 numpy arrays, each containing one utterance
        """
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_float])

        utterances = []
        frame_size = int(config.CHUNK_DURATION_MS * self.sample_rate / 1000)

        while len(self.audio_buffer) >= frame_size:
            frame = self.audio_buffer[:frame_size]
            self.audio_buffer = self.audio_buffer[frame_size:]
            frame_tensor = torch.from_numpy(frame)
            speech_prob = self.vad_model(frame_tensor, self.sample_rate).item()

            if speech_prob >= self.threshold:
                self.is_speaking = True
                self.silence_counter = 0
                self.speech_buffer = np.concatenate([self.speech_buffer, frame])
                if len(self.speech_buffer) >= self.max_speech_samples:
                    utterances.append(self.speech_buffer.copy())
                    self.speech_buffer = np.array([], dtype=np.float32)
            else:
                if self.is_speaking:
                    self.silence_counter += frame_size
                    self.speech_buffer = np.concatenate([self.speech_buffer, frame])
                    if self.silence_counter >= self.silence_samples:
                        if len(self.speech_buffer) >= self.min_speech_samples:
                            utterances.append(self.speech_buffer.copy())
                        self.speech_buffer = np.array([], dtype=np.float32)
                        self.is_speaking = False
                        self.silence_counter = 0

        return utterances

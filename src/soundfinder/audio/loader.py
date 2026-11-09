"""Audio loading and normalization for SoundFinder."""

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
from mutagen.mp3 import MP3


TARGET_SAMPLE_RATE = 44_100
MAX_DURATION_SECONDS = 4.0
SUPPORTED_SUFFIXES = {".mp3", ".wav"}
MP3_BITRATE = 320_000


class AudioValidationError(ValueError):
    """Raised when an audio file does not meet the SoundFinder input contract."""


@dataclass(frozen=True)
class AudioData:
    """Normalized audio samples and their sample rate."""

    samples: np.ndarray
    sample_rate: int

    @property
    def duration(self) -> float:
        """Return the duration of the normalized signal in seconds."""
        return self.samples.shape[0] / self.sample_rate


def load_audio(path: str | Path) -> AudioData:
    """Load and normalize a WAV or 320 kb/s MP3 file."""
    audio_path = Path(path)
    suffix = audio_path.suffix.lower()

    if suffix not in SUPPORTED_SUFFIXES:
        raise AudioValidationError(
            f"Unsupported audio format '{suffix or '<none>'}'. "
            "Expected a WAV or MP3 file."
        )
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    _validate_duration(audio_path, suffix)
    if suffix == ".mp3":
        _validate_mp3_bitrate(audio_path)

    try:
        samples, sample_rate = librosa.load(
            audio_path,
            sr=TARGET_SAMPLE_RATE,
            mono=True,
            dtype=np.float32,
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise AudioValidationError(f"Could not decode audio file: {audio_path}") from error

    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1:
        raise AudioValidationError("Audio normalization did not produce a mono signal.")

    normalized = AudioData(samples=samples, sample_rate=sample_rate)
    if normalized.duration > MAX_DURATION_SECONDS:
        raise AudioValidationError(
            f"Audio duration is {normalized.duration:.3f}s; "
            f"the maximum is {MAX_DURATION_SECONDS:.1f}s."
        )
    return normalized


def _validate_duration(path: Path, suffix: str) -> None:
    if suffix == ".mp3":
        duration = MP3(path).info.length
    else:
        duration = librosa.get_duration(path=path)

    if duration > MAX_DURATION_SECONDS:
        raise AudioValidationError(
            f"Audio duration is {duration:.3f}s; "
            f"the maximum is {MAX_DURATION_SECONDS:.1f}s."
        )


def _validate_mp3_bitrate(path: Path) -> None:
    bitrate = MP3(path).info.bitrate
    if bitrate != MP3_BITRATE:
        raise AudioValidationError(
            f"MP3 bitrate is {bitrate} b/s; only 320000 b/s is supported."
        )

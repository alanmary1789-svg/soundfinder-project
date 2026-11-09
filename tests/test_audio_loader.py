from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from soundfinder.audio import (
    TARGET_SAMPLE_RATE,
    AudioValidationError,
    load_audio,
)


def _write_wav(path: Path, duration: float, sample_rate: int, channels: int = 1) -> None:
    sample_count = int(duration * sample_rate)
    samples = np.zeros((sample_count, channels), dtype=np.float32)
    sf.write(path, samples, sample_rate)


def test_load_audio_normalizes_stereo_and_sample_rate(tmp_path: Path):
    path = tmp_path / "sample.wav"
    _write_wav(path, duration=1, sample_rate=22_050, channels=2)

    audio = load_audio(path)

    assert audio.sample_rate == TARGET_SAMPLE_RATE
    assert audio.samples.dtype == np.float32
    assert audio.samples.ndim == 1
    assert audio.samples.shape[0] == TARGET_SAMPLE_RATE


def test_load_audio_rejects_audio_longer_than_four_seconds(tmp_path: Path):
    path = tmp_path / "long.wav"
    _write_wav(path, duration=4.1, sample_rate=44_100)

    with pytest.raises(AudioValidationError, match="maximum is 4.0s"):
        load_audio(path)


def test_load_audio_rejects_unsupported_format(tmp_path: Path):
    path = tmp_path / "sample.ogg"
    path.write_bytes(b"not audio")

    with pytest.raises(AudioValidationError, match="Unsupported audio format"):
        load_audio(path)

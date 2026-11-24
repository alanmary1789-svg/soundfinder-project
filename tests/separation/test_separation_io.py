import numpy as np
import pytest
import soundfile as sf

from soundfinder.separation import SeparationError, Stem, load_track, save_stems

SAMPLE_RATE = 22_050


def test_load_track_keeps_stereo_and_sample_rate(tmp_path):
    path = tmp_path / "track.wav"
    sf.write(path, np.zeros((SAMPLE_RATE, 2), dtype=np.float32), SAMPLE_RATE)

    samples, sample_rate = load_track(path)

    assert sample_rate == SAMPLE_RATE
    assert samples.shape == (2, SAMPLE_RATE)


def test_load_track_accepts_tracks_longer_than_four_seconds(tmp_path):
    path = tmp_path / "long.wav"
    sf.write(path, np.zeros(10 * SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)

    samples, _ = load_track(path)

    assert samples.shape[-1] == 10 * SAMPLE_RATE


def test_load_track_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_track(tmp_path / "missing.wav")


def test_load_track_rejects_corrupt_file(tmp_path):
    path = tmp_path / "corrupt.wav"
    path.write_bytes(b"RIFF....not audio")

    with pytest.raises(SeparationError, match="decode"):
        load_track(path)


def test_save_stems_writes_one_wav_per_stem(tmp_path):
    stem = Stem(name="percussive", samples=np.zeros((2, 1_000), np.float32), sample_rate=SAMPLE_RATE)

    paths = save_stems([stem], tmp_path / "out")

    assert paths == [tmp_path / "out" / "percussive.wav"]
    written, sample_rate = sf.read(paths[0])
    assert sample_rate == SAMPLE_RATE
    assert written.shape == (1_000, 2)
import numpy as np
import pytest

from soundfinder.separation import HPSSSeparator, SeparationError

SAMPLE_RATE = 22_050


def _sine(duration: float = 2.0, frequency: float = 440.0) -> np.ndarray:
    """Son tenu : doit se retrouver surtout dans la partie harmonique."""
    t = np.arange(int(duration * SAMPLE_RATE)) / SAMPLE_RATE
    return (0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def _clicks(duration: float = 2.0, every: float = 0.25) -> np.ndarray:
    """Suite d'impulsions brèves : doit se retrouver surtout dans la partie percussive."""
    signal = np.zeros(int(duration * SAMPLE_RATE), dtype=np.float32)
    signal[:: int(every * SAMPLE_RATE)] = 1.0
    return signal


def _energy(stem) -> float:
    return float(np.sum(stem.samples**2))


def _by_name(stems):
    return {stem.name: stem for stem in stems}


def test_returns_harmonic_and_percussive_stems():
    stems = HPSSSeparator().separate(_sine(), SAMPLE_RATE)

    assert [stem.name for stem in stems] == ["harmonic", "percussive"]
    for stem in stems:
        assert stem.sample_rate == SAMPLE_RATE
        assert stem.samples.shape == (1, _sine().shape[0])
        assert stem.samples.dtype == np.float32


def test_keeps_stereo_channels():
    stereo = np.stack([_sine(), _clicks()])

    stems = HPSSSeparator().separate(stereo, SAMPLE_RATE)

    assert all(stem.channels == 2 for stem in stems)


def test_sustained_tone_goes_to_harmonic_part():
    stems = _by_name(HPSSSeparator().separate(_sine(), SAMPLE_RATE))

    assert _energy(stems["harmonic"]) > 10 * _energy(stems["percussive"])


def test_clicks_go_to_percussive_part():
    stems = _by_name(HPSSSeparator().separate(_clicks(), SAMPLE_RATE))

    assert _energy(stems["percussive"]) > 10 * _energy(stems["harmonic"])


def test_parts_add_up_to_the_original_signal():
    mix = _sine() + _clicks()
    stems = _by_name(HPSSSeparator().separate(mix, SAMPLE_RATE))

    rebuilt = stems["harmonic"].samples + stems["percussive"].samples

    assert np.allclose(rebuilt[0], mix, atol=1e-3)


def test_invalid_margin_is_rejected():
    with pytest.raises(SeparationError, match="margin"):
        HPSSSeparator(margin=0.5)
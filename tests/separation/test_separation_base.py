import numpy as np
import pytest

from soundfinder.separation import SeparationError, Separator, Stem, to_channels_first


class IdentitySeparator(Separator):
    """Fake separator used to test the shared contract."""

    name = "identity"

    def _separate(self, samples: np.ndarray, sample_rate: int) -> list[Stem]:
        return [Stem(name="mix", samples=samples, sample_rate=sample_rate)]


def test_mono_signal_becomes_channels_first():
    samples = to_channels_first(np.zeros(1_000))

    assert samples.shape == (1, 1_000)
    assert samples.dtype == np.float32


def test_stereo_signal_keeps_its_channels():
    assert to_channels_first(np.zeros((2, 1_000))).shape == (2, 1_000)


def test_empty_signal_is_rejected():
    with pytest.raises(SeparationError, match="empty"):
        to_channels_first(np.zeros(0))


def test_nan_values_are_rejected():
    with pytest.raises(SeparationError, match="NaN"):
        to_channels_first(np.array([0.0, np.nan]))


def test_invalid_sample_rate_is_rejected():
    with pytest.raises(SeparationError, match="sample rate"):
        IdentitySeparator().separate(np.zeros(1_000), sample_rate=0)


def test_separator_returns_stems_with_duration():
    stems = IdentitySeparator().separate(np.zeros(44_100), sample_rate=44_100)

    assert len(stems) == 1
    assert stems[0].channels == 1
    assert stems[0].duration == pytest.approx(1.0)
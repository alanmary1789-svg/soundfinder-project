import numpy as np
import pytest

from soundfinder.separation import DemucsSeparator, SeparationError

# Ces tests sont ignorés si Demucs n'est pas installé (dépendance optionnelle).
pytest.importorskip("demucs")

# "demucs_unittest" est un petit modèle non entraîné fourni par Demucs :
# il ne télécharge rien et suffit pour vérifier les formats d'entrée et de sortie.
TEST_MODEL = "demucs_unittest"
SAMPLE_RATE = 44_100


def _noise(channels: int = 2, duration: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    return 0.1 * rng.standard_normal((channels, int(duration * SAMPLE_RATE))).astype(np.float32)


def test_returns_the_four_standard_sources():
    stems = DemucsSeparator(model_name=TEST_MODEL).separate(_noise(), SAMPLE_RATE)

    assert sorted(stem.name for stem in stems) == ["bass", "drums", "other", "vocals"]
    for stem in stems:
        assert stem.samples.shape == (2, SAMPLE_RATE)
        assert stem.samples.dtype == np.float32
        assert stem.sample_rate == SAMPLE_RATE


def test_two_stems_mode_returns_drums_and_the_rest():
    stems = DemucsSeparator(model_name=TEST_MODEL, two_stems="drums").separate(
        _noise(), SAMPLE_RATE
    )

    assert [stem.name for stem in stems] == ["drums", "no_drums"]


def test_mono_input_is_converted_to_stereo():
    stems = DemucsSeparator(model_name=TEST_MODEL).separate(_noise(channels=1)[0], SAMPLE_RATE)

    assert all(stem.channels == 2 for stem in stems)


def test_other_sample_rates_are_resampled_to_the_model_rate():
    signal = _noise(duration=1.0)[:, :22_050]

    stems = DemucsSeparator(model_name=TEST_MODEL).separate(signal, 22_050)

    assert all(stem.sample_rate == SAMPLE_RATE for stem in stems)
    assert all(stem.duration == pytest.approx(1.0, abs=0.01) for stem in stems)


def test_unknown_two_stems_source_is_rejected():
    with pytest.raises(SeparationError, match="Unknown source"):
        DemucsSeparator(model_name=TEST_MODEL, two_stems="piano")
        
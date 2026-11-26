"""Lecture d'un morceau et écriture des sources séparées en fichiers WAV."""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from .base import SeparationError, Stem


def load_track(path: str | Path) -> tuple[np.ndarray, int]:
    """Charge un morceau en conservant ses canaux et sa fréquence d'origine.

    Chargement provisoire propre à la séparation : contrairement au décodeur
    de samples, il garde la stéréo et n'impose pas de durée maximale. À
    remplacer par le décodeur commun quand le format d'entrée sera validé
    avec l'équipe.
    """
    track_path = Path(path)
    if not track_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {track_path}")
    try:
        samples, sample_rate = librosa.load(track_path, sr=None, mono=False)
    except Exception as error:
        raise SeparationError(f"Could not decode audio file: {track_path}") from error
    return samples, int(sample_rate)


def save_stems(stems: list[Stem], output_dir: str | Path) -> list[Path]:
    """Écrit chaque source dans `output_dir/<nom>.wav` et renvoie les chemins créés."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for stem in stems:
        path = directory / f"{stem.name}.wav"
        # soundfile attend la forme (échantillons, canaux) : on transpose.
        sf.write(path, stem.samples.T, stem.sample_rate, subtype="FLOAT")
        paths.append(path)
    return paths
"""Baseline DSP : séparation harmonique / percussive (HPSS) avec librosa."""

import librosa
import numpy as np

from .base import SeparationError, Separator, Stem


class HPSSSeparator(Separator):
    """Sépare un signal en une partie harmonique et une partie percussive.

    Principe : sur le spectrogramme (STFT), les sons harmoniques forment des
    lignes horizontales (fréquences stables dans le temps) et les sons
    percussifs des lignes verticales (énergie brève sur beaucoup de
    fréquences). Un filtre médian dans chaque direction permet de construire
    deux masques, puis deux signaux.

    Ce n'est pas une séparation par instrument : la partie percussive sert
    d'approximation de la batterie, à comparer ensuite avec Demucs.
    """

    name = "hpss"

    def __init__(self, margin: float = 1.0, kernel_size: int = 31) -> None:
        """`margin` > 1 rend la séparation plus stricte ; `kernel_size` règle les filtres médians."""
        if margin < 1.0:
            raise SeparationError(f"margin must be >= 1.0, got {margin}.")
        if kernel_size < 1:
            raise SeparationError(f"kernel_size must be >= 1, got {kernel_size}.")
        self.margin = margin
        self.kernel_size = kernel_size

    def _separate(self, samples: np.ndarray, sample_rate: int) -> list[Stem]:
        """Applique HPSS canal par canal et renvoie les sources "harmonic" et "percussive"."""
        harmonic, percussive = librosa.effects.hpss(
            samples,
            kernel_size=self.kernel_size,
            margin=self.margin,
        )
        return [
            Stem(name="harmonic", samples=harmonic.astype(np.float32), sample_rate=sample_rate),
            Stem(name="percussive", samples=percussive.astype(np.float32), sample_rate=sample_rate),
        ]
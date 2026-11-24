"""Contrat commun à tous les séparateurs de sources de SoundFinder."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class SeparationError(ValueError):
    """Erreur levée lorsqu'un signal ne peut pas être séparé."""


@dataclass(frozen=True)
class Stem:
    """Une source séparée d'un morceau (par exemple "drums" ou "percussive").

    `samples` est un tableau float32 de forme (canaux, échantillons).
    """

    name: str
    samples: np.ndarray
    sample_rate: int

    @property
    def channels(self) -> int:
        """Nombre de canaux (1 pour du mono, 2 pour de la stéréo)."""
        return self.samples.shape[0]

    @property
    def duration(self) -> float:
        """Durée de la source en secondes."""
        return self.samples.shape[1] / self.sample_rate


def to_channels_first(samples: np.ndarray) -> np.ndarray:
    """Renvoie un tableau float32 de forme (canaux, échantillons).

    Un signal mono à 1 dimension devient (1, échantillons). Un signal à
    2 dimensions est supposé être déjà de forme (canaux, échantillons).
    """
    array = np.asarray(samples, dtype=np.float32)
    if array.ndim == 1:
        array = array[np.newaxis, :]
    if array.ndim != 2:
        raise SeparationError(f"Expected a 1-D or 2-D signal, got {array.ndim} dimensions.")
    if array.shape[1] == 0:
        raise SeparationError("Cannot separate an empty signal.")
    if not np.all(np.isfinite(array)):
        raise SeparationError("The signal contains NaN or infinite values.")
    return array


class Separator(ABC):
    """Interface commune à toutes les méthodes de séparation (baseline DSP, Demucs...)."""

    name: str = "separator"

    def separate(self, samples: np.ndarray, sample_rate: int) -> list[Stem]:
        """Vérifie l'entrée, puis délègue la séparation à la méthode concrète."""
        if sample_rate <= 0:
            raise SeparationError(f"Invalid sample rate: {sample_rate}.")
        return self._separate(to_channels_first(samples), sample_rate)

    @abstractmethod
    def _separate(self, samples: np.ndarray, sample_rate: int) -> list[Stem]:
        """Sépare un signal float32 déjà vérifié, de forme (canaux, échantillons)."""
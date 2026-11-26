"""Séparation par réseau de neurones pré-entraîné : Demucs."""

import numpy as np

from .base import SeparationError, Separator, Stem


class DemucsSeparator(Separator):
    """Sépare un morceau avec un modèle Demucs pré-entraîné.

    Les modèles standards produisent quatre sources : "drums", "bass",
    "other" et "vocals". Avec `two_stems="drums"`, on obtient seulement
    "drums" et "no_drums" (somme des autres sources), ce qui correspond au
    MVP batterie / non-batterie.

    Demucs travaille en stéréo à 44,1 kHz : l'entrée est convertie si
    nécessaire, et les sources sont renvoyées à la fréquence du modèle.
    """

    name = "demucs"

    def __init__(
        self,
        model_name: str = "htdemucs",
        device: str = "cpu",
        shifts: int = 1,
        two_stems: str | None = None,
    ) -> None:
        """Charge le modèle (téléchargé et mis en cache lors de la première utilisation)."""
        try:
            import torch
            from demucs.api import Separator as DemucsApi
            from demucs.audio import convert_audio
        except ImportError as error:
            raise SeparationError(
                "Demucs is not installed. Run: python -m pip install -e '.[dev,demucs]'"
            ) from error

        self._torch = torch
        self._convert_audio = convert_audio
        self._model = DemucsApi(model=model_name, device=device, shifts=shifts)
        sources = list(self._model.model.sources)
        if two_stems is not None and two_stems not in sources:
            raise SeparationError(f"Unknown source '{two_stems}'. Available: {sources}.")
        self.model_name = model_name
        self.two_stems = two_stems

    @property
    def sample_rate(self) -> int:
        """Fréquence d'échantillonnage des sources produites par le modèle."""
        return self._model.samplerate

    def _separate(self, samples: np.ndarray, sample_rate: int) -> list[Stem]:
        """Applique le modèle et convertit chaque source en `Stem`."""
        wav = self._torch.from_numpy(np.ascontiguousarray(samples))
        # Demucs attend exactement ses canaux et sa fréquence : on convertit
        # toujours nous-mêmes (sa fonction ne convertit pas un mono à 44,1 kHz).
        wav = self._convert_audio(
            wav, sample_rate, self.sample_rate, self._model.audio_channels
        )
        with self._torch.no_grad():
            _, separated = self._model.separate_tensor(wav, self.sample_rate)

        sources = {
            name: tensor.cpu().numpy().astype(np.float32) for name, tensor in separated.items()
        }
        if self.two_stems is not None:
            target = sources.pop(self.two_stems)
            sources = {self.two_stems: target, f"no_{self.two_stems}": sum(sources.values())}

        return [
            Stem(name=name, samples=audio, sample_rate=self.sample_rate)
            for name, audio in sources.items()
        ]
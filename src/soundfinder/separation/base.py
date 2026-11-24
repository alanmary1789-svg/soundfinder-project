from abc import ABC, abstractmethod 
from dataclasses import dataclass 

import numpy as np


class SeparationError(ValueError):
    """Raised when a signal cannot be separated"""

@dataclass(frozen=True)
class Stem:
    name : str
    samples : np.ndarray
    sample_rate : int 

    @property 
    def channels(self) -> int:
        return self.samples.shape[0]

    @property
    def duration(self) -> float:
        return self.samples.shape[1] / self.sample_rate

def to_channels_first(samples: np.ndarray) -> np.ndarray:
    array = np.asarray(samples, dtype=np.float32)
    if array.ndim == 1:
        array = array[np.newaxis, :]
    if array.ndim != 2:
        raise SeparationError(f"Expected a 1-D or 2-D signal, got {array.ndim} dimensions.") 
    if array.shape[1] == 0:
        raise SeparationError("Cannot seperate an empty signal.") 
    if not np.all(np.isfinite(array)):
        raise SeparationError("The signal contains NaN or infinite values")
    return array 

class Separator(ABC):
    name : str = "separator"

    def separate(self, samples: np.ndarray, sample_rate: int) -> list[Stem]:
        if sample_rate <= 0 :
            raise SeparationError(f"Invalid sample rate : {sample_rate}.")
        return self._separate(to_channels_first(samples), sample_rate)

    @abstractmethod
    def _separate(self, samples: np.ndarray, sample_rate: int) -> list[Stem]:
        """Separate a valid channel """
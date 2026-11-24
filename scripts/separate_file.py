"""Sépare un morceau et écrit les sources en WAV pour pouvoir les écouter.

Exemple :
    python scripts/separate_file.py ~/Music/morceau.mp3 --margin 2
"""

import argparse
import time
from pathlib import Path

from soundfinder.separation import HPSSSeparator, load_track, save_stems


def main() -> None:
    parser = argparse.ArgumentParser(description="Sépare un morceau en sources WAV.")
    parser.add_argument("track", type=Path, help="Morceau à séparer (WAV, MP3...).")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("workspace/separation"),
        help="Dossier de sortie (ignoré par Git).",
    )
    parser.add_argument(
        "--margin", type=float, default=1.0, help="Sévérité de HPSS (>= 1)."
    )
    args = parser.parse_args()

    samples, sample_rate = load_track(args.track)
    separator = HPSSSeparator(margin=args.margin)

    start = time.perf_counter()
    stems = separator.separate(samples, sample_rate)
    elapsed = time.perf_counter() - start

    output_dir = args.output / args.track.stem / f"{separator.name}_margin{args.margin:g}"
    paths = save_stems(stems, output_dir)

    print(f"Séparation '{separator.name}' en {elapsed:.1f} s :")
    for stem, path in zip(stems, paths, strict=True):
        print(f"  {stem.name:<12} {stem.channels} canal(aux), {stem.duration:.1f} s -> {path}")


if __name__ == "__main__":
    main()
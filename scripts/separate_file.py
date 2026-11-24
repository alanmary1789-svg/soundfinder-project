"""Sépare un morceau et écrit les sources en WAV pour pouvoir les écouter.

Exemples :
    python scripts/separate_file.py ~/Music/morceau.mp3 --method hpss --margin 2
    python scripts/separate_file.py ~/Music/morceau.mp3 --method demucs --two-stems drums
"""

import argparse
import time
from pathlib import Path

from soundfinder.separation import (
    DemucsSeparator,
    HPSSSeparator,
    Separator,
    load_track,
    save_stems,
)


def build_separator(args: argparse.Namespace) -> tuple[Separator, str]:
    """Construit le séparateur demandé et le nom de son dossier de sortie."""
    if args.method == "hpss":
        return HPSSSeparator(margin=args.margin), f"hpss_margin{args.margin:g}"
    separator = DemucsSeparator(
        model_name=args.model, device=args.device, two_stems=args.two_stems
    )
    suffix = f"_{args.two_stems}" if args.two_stems else ""
    return separator, f"demucs_{args.model}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sépare un morceau en sources WAV.")
    parser.add_argument("track", type=Path, help="Morceau à séparer (WAV, MP3...).")
    parser.add_argument("--method", choices=["hpss", "demucs"], default="hpss")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("workspace/separation"),
        help="Dossier de sortie (ignoré par Git).",
    )
    parser.add_argument("--margin", type=float, default=1.0, help="HPSS : sévérité (>= 1).")
    parser.add_argument("--model", default="htdemucs", help="Demucs : nom du modèle.")
    parser.add_argument("--device", default="cpu", help="Demucs : cpu, mps ou cuda.")
    parser.add_argument("--two-stems", default=None, help="Demucs : ex. drums -> drums/no_drums.")
    args = parser.parse_args()

    samples, sample_rate = load_track(args.track)
    separator, folder = build_separator(args)

    start = time.perf_counter()
    stems = separator.separate(samples, sample_rate)
    elapsed = time.perf_counter() - start

    paths = save_stems(stems, args.output / args.track.stem / folder)

    print(f"Séparation '{separator.name}' en {elapsed:.1f} s :")
    for stem, path in zip(stems, paths, strict=True):
        print(f"  {stem.name:<12} {stem.channels} canal(aux), {stem.duration:.1f} s -> {path}")


if __name__ == "__main__":
    main()
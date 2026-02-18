"""
Download official benchmark datasets used in the RLM paper.

This script fetches datasets from Hugging Face and saves them under evals/data/.
Datasets are large; use selectively.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DATASET_PRESETS = {
    "oolong": {
        "repo": "oolongbench/oolong-synth",
        "description": "OOLONG benchmark (official)",
    },
    "longbench_v2": {
        "repo": "zai-org/LongBench-v2",
        "description": "LongBench-v2 (all domains)",
    },
    "repoqa": {
        "url": "https://github.com/evalplus/repoqa_release/raw/main/repoqa-2024-06-23.json.gz",
        "description": "RepoQA (official release, JSONL.GZ)",
    },
    "browsecomp_plus": {
        "repo": "Tevatron/browsecomp-plus",
        "description": "BrowseComp-Plus questions (obfuscated)",
    },
    "browsecomp_plus_corpus": {
        "repo": "Tevatron/browsecomp-plus-corpus",
        "description": "BrowseComp-Plus corpus (obfuscated)",
    },
    "ruler_full_mirror": {
        "repo": "tonychenxyz/ruler-full",
        "description": "Community HF mirror of RULER (not official)",
    },
}


def _load_datasets():
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        print("Missing dependency: datasets", file=sys.stderr)
        print("Install with: python -m pip install datasets huggingface_hub", file=sys.stderr)
        raise SystemExit(1) from exc
    return load_dataset


def _save_dataset(ds, out_dir: Path, max_samples: int | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    ds.save_to_disk(out_dir)


def _download_url(url: str, out_dir: Path, filename: str | None = None) -> Path:
    from urllib.request import urlopen

    out_dir.mkdir(parents=True, exist_ok=True)
    name = filename or url.split("/")[-1]
    dest = out_dir / name
    if dest.exists():
        print(f"  Skipping download (exists): {dest}")
        return dest

    print(f"  Downloading {url} -> {dest}")
    with urlopen(url) as resp, dest.open("wb") as handle:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    return dest


def download_dataset(
    repo: str,
    out_dir: Path,
    split: str | None,
    config: str | None,
    trust_remote_code: bool,
    max_samples: int | None,
) -> None:
    load_dataset = _load_datasets()

    if split:
        ds = load_dataset(repo, name=config, split=split, trust_remote_code=trust_remote_code)
        _save_dataset(ds, out_dir / split, max_samples)
        return

    ds_dict = load_dataset(repo, name=config, trust_remote_code=trust_remote_code)
    if hasattr(ds_dict, "items"):
        for split_name, split_ds in ds_dict.items():
            _save_dataset(split_ds, out_dir / split_name, max_samples)
        return

    _save_dataset(ds_dict, out_dir / "default", max_samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official RLM benchmark datasets")
    parser.add_argument("--list", action="store_true", help="List preset datasets and exit")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASET_PRESETS.keys()),
        help="Dataset preset to download (repeatable)",
    )
    parser.add_argument("--output-dir", default="evals/data", help="Output directory")
    parser.add_argument("--split", default=None, help="Dataset split (e.g., train, test)")
    parser.add_argument("--config", default=None, help="Dataset config name (if applicable)")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples per split")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow datasets with remote code (required by some datasets)",
    )

    args = parser.parse_args()

    if args.list:
        for key, meta in DATASET_PRESETS.items():
            source = meta.get("repo") or meta.get("url", "manual")
            print(f"{key}: {source} - {meta['description']}")
        return

    if not args.dataset:
        print("No datasets selected. Use --dataset or --list.", file=sys.stderr)
        raise SystemExit(2)

    output_root = Path(args.output_dir)
    for preset in args.dataset:
        meta = DATASET_PRESETS[preset]
        out_dir = output_root / preset
        if meta.get("manual"):
            print(f"{preset}: manual download required. See evals/README.md for instructions.")
            continue
        if "url" in meta:
            _download_url(meta["url"], out_dir)
            continue
        repo = meta["repo"]
        print(f"Downloading {preset} from {repo} -> {out_dir}")
        download_dataset(
            repo=repo,
            out_dir=out_dir,
            split=args.split,
            config=args.config,
            trust_remote_code=args.trust_remote_code,
            max_samples=args.max_samples,
        )

    print("Done.")


if __name__ == "__main__":
    main()

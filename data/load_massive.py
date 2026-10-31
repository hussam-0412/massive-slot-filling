"""Convert MASSIVE into the BIO CoNLL format this repo's harness reads.

MASSIVE ships slot annotations inline in an `annot_utt` field rather than as
token/label columns:

    utt       : wake me up at five am on friday
    annot_utt : wake me up at [time : five am] on [date : friday]

so the first job is to turn that bracket notation into aligned BIO labels:

    wake  O
    me    O
    up    O
    at    O
    five  B-time
    am    I-time
    on    O
    friday B-date

Usage:
    python data/load_massive.py --locales tr-TR en-US --out data/massive
    python data/load_massive.py --locales tr-TR --out data/massive --max-train 5000

Dataset: https://huggingface.co/datasets/AmazonScience/massive
Paper:   https://arxiv.org/abs/2204.08582
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from conll import Sentence, describe, write_conll  # noqa: E402

# [slot_name : slot value]  -- no nesting occurs in MASSIVE
SLOT_RE = re.compile(r"\[([^\[\]:]+?)\s*:\s*([^\[\]]*?)\]")


class AnnotationError(ValueError):
    """Raised when annot_utt cannot be aligned with utt."""


def parse_annot_utt(annot: str) -> Sentence:
    """Turn MASSIVE's bracket annotation into (tokens, BIO labels)."""
    tokens: list[str] = []
    labels: list[str] = []
    pos = 0

    for match in SLOT_RE.finditer(annot):
        for tok in annot[pos : match.start()].split():
            tokens.append(tok)
            labels.append("O")

        slot_type = match.group(1).strip()
        slot_value = match.group(2).strip()
        if not slot_type:
            raise AnnotationError(f"empty slot name in: {annot!r}")

        value_tokens = slot_value.split()
        if not value_tokens:
            # An empty slot value carries no tokens to label; skip it rather
            # than emitting a span with no extent.
            pos = match.end()
            continue

        for i, tok in enumerate(value_tokens):
            tokens.append(tok)
            labels.append(f"{'B' if i == 0 else 'I'}-{slot_type}")
        pos = match.end()

    for tok in annot[pos:].split():
        tokens.append(tok)
        labels.append("O")

    return tokens, labels


def check_alignment(utt: str, tokens: list[str]) -> bool:
    """The stripped annotation must reproduce the plain utterance."""
    return utt.split() == tokens


def convert_rows(rows, strict: bool = False) -> tuple[list[Sentence], int]:
    """Convert an iterable of MASSIVE rows into sentences, counting skips."""
    sentences: list[Sentence] = []
    skipped = 0

    for row in rows:
        annot = row.get("annot_utt")
        utt = row.get("utt", "")
        if not annot:
            skipped += 1
            continue
        try:
            tokens, labels = parse_annot_utt(annot)
        except AnnotationError:
            skipped += 1
            continue

        if not tokens:
            skipped += 1
            continue

        if utt and not check_alignment(utt, tokens):
            # annot_utt and utt disagree on tokenization. These exist in the
            # corpus; dropping them silently would overstate data quality, so
            # they are counted and reported.
            if strict:
                skipped += 1
                continue
        sentences.append((tokens, labels))

    return sentences, skipped


# The official raw release: one JSONL file per locale, inside a tarball.
# Using this instead of the `datasets` library keeps the whole download path
# on the Python standard library - no datasets, no pyarrow, no numpy - which
# avoids dependency conflicts with whatever else is installed on the machine.
RAW_URL = (
    "https://amazon-massive-nlu-dataset.s3.amazonaws.com/"
    "amazon-massive-dataset-1.1.tar.gz"
)


def _safe_extract(tar, dest: str) -> None:
    """Extract without allowing paths to escape `dest`."""
    dest_abs = os.path.abspath(dest)
    for member in tar.getmembers():
        target = os.path.abspath(os.path.join(dest, member.name))
        if not target.startswith(dest_abs + os.sep) and target != dest_abs:
            raise ValueError(f"unsafe path in archive: {member.name}")
        if member.issym() or member.islnk():
            raise ValueError(f"link in archive refused: {member.name}")
    # Python 3.12+ has a vetted filter; older versions use the checks above.
    try:
        tar.extractall(dest, filter="data")
    except TypeError:
        tar.extractall(dest)


def download_raw(raw_dir: str) -> str:
    """Download and extract the official tarball. Returns the JSONL directory."""
    import tarfile
    import urllib.request

    os.makedirs(raw_dir, exist_ok=True)
    archive = os.path.join(raw_dir, "amazon-massive-dataset-1.1.tar.gz")

    existing = find_jsonl_dir(raw_dir)
    if existing:
        print(f"  reusing extracted data at {existing}")
        return existing

    if not os.path.exists(archive):
        print(f"  downloading {RAW_URL}")

        def progress(block_num, block_size, total_size):
            if total_size > 0:
                done = min(block_num * block_size, total_size)
                pct = 100 * done / total_size
                print(
                    f"\r  {done / 1e6:.1f} / {total_size / 1e6:.1f} MB "
                    f"({pct:.0f}%)",
                    end="",
                    flush=True,
                )

        urllib.request.urlretrieve(RAW_URL, archive, reporthook=progress)
        print()
    else:
        print(f"  reusing archive at {archive}")

    print("  extracting ...")
    with tarfile.open(archive, "r:gz") as tar:
        _safe_extract(tar, raw_dir)

    found = find_jsonl_dir(raw_dir)
    if not found:
        raise FileNotFoundError(
            f"no .jsonl files found after extracting into {raw_dir}"
        )
    return found


def find_jsonl_dir(root: str) -> str | None:
    """Locate the directory holding <locale>.jsonl files."""
    if not os.path.isdir(root):
        return None
    for dirpath, _dirnames, filenames in os.walk(root):
        if any(f.endswith(".jsonl") for f in filenames):
            return dirpath
    return None


def load_from_hf(locale: str, max_train: int | None):
    from datasets import load_dataset

    ds = load_dataset("AmazonScience/massive", locale, trust_remote_code=True)
    out = {}
    split_map = {"train": "train", "validation": "dev", "test": "test"}
    for split, target in split_map.items():
        if split not in ds:
            continue
        rows = ds[split]
        if target == "train" and max_train:
            rows = rows.select(range(min(max_train, len(rows))))
        out[target] = rows
    return out


def load_from_jsonl(path: str):
    """Fallback: the raw MASSIVE release is one JSONL file per locale."""
    buckets: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    partition_map = {
        "train": "train",
        "dev": "dev",
        "validation": "dev",
        "test": "test",
    }
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            target = partition_map.get(row.get("partition", "train"), "train")
            buckets[target].append(row)
    return {k: v for k, v in buckets.items() if v}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--locales", nargs="+", default=["tr-TR", "en-US"])
    ap.add_argument("--out", default="data/massive")
    ap.add_argument(
        "--jsonl-dir",
        help="Directory of raw <locale>.jsonl files, used instead of the Hub",
    )
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Drop utterances whose annot_utt does not tokenize like utt",
    )
    ap.add_argument(
        "--source",
        choices=["raw", "hf", "jsonl"],
        default="raw",
        help="raw: official tarball, stdlib only (default). "
             "hf: the datasets library. jsonl: an existing --jsonl-dir.",
    )
    ap.add_argument("--raw-dir", default="data/raw")
    args = ap.parse_args()

    # Stale output from a previous run must never be mistaken for fresh data,
    # so any existing per-locale directory is cleared before writing.
    for locale in args.locales:
        stale = os.path.join(args.out, locale)
        if os.path.isdir(stale):
            print(f"removing previous output at {stale}")
            shutil.rmtree(stale)

    jsonl_dir = args.jsonl_dir
    if args.source == "raw" and not jsonl_dir:
        print("=== fetching the official MASSIVE release ===")
        try:
            jsonl_dir = download_raw(args.raw_dir)
            print(f"  jsonl directory: {jsonl_dir}")
        except Exception as exc:  # noqa: BLE001
            print(f"\nFAILED to download the raw release: "
                  f"{type(exc).__name__}: {exc}")
            print(
                "\nFall back to a manual download:\n"
                f"  1. open {RAW_URL}\n"
                "  2. extract it\n"
                "  3. re-run with --source jsonl "
                "--jsonl-dir <folder containing tr-TR.jsonl>"
            )
            sys.exit(1)

    succeeded: list[str] = []
    failed: list[str] = []

    for locale in args.locales:
        print(f"\n=== {locale} ===")
        try:
            if jsonl_dir:
                path = os.path.join(jsonl_dir, f"{locale}.jsonl")
                splits = load_from_jsonl(path)
            else:
                splits = load_from_hf(locale, args.max_train)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED to load {locale}: {type(exc).__name__}: {exc}")
            failed.append(locale)
            continue

        if not splits:
            print(f"  FAILED: no data parsed for {locale}")
            failed.append(locale)
            continue

        out_dir = os.path.join(args.out, locale)
        os.makedirs(out_dir, exist_ok=True)
        total_skipped = 0
        slot_types: set[str] = set()

        for target, rows in splits.items():
            sentences, skipped = convert_rows(rows, strict=args.strict)
            total_skipped += skipped
            for _, labels in sentences:
                for lab in labels:
                    if lab != "O":
                        slot_types.add(lab.split("-", 1)[1])
            write_conll(os.path.join(out_dir, f"{target}.conll"), sentences)
            print(" ", describe(target, sentences))

        print(f"  slot types seen: {len(slot_types)}")
        print(f"  skipped utterances: {total_skipped}")
        print(f"  -> {out_dir}/")
        succeeded.append(locale)

        if len(slot_types) < 20:
            print(
                f"  WARNING: only {len(slot_types)} slot types. Real MASSIVE "
                "has 55. Check that this is the genuine corpus and not sample "
                "data."
            )

    print()
    if failed:
        print(f"FAILED locales: {', '.join(failed)}")
    if not succeeded:
        print(
            "Nothing was loaded, so no data was written. Anything already in "
            f"{args.out} is stale and must not be used."
        )
        sys.exit(1)
    print(f"loaded: {', '.join(succeeded)}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

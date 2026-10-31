"""Generate SYNTHETIC MASSIVE-shaped JSONL for offline testing.

Writes <locale>.jsonl files using MASSIVE's real field names and its
`annot_utt` bracket notation, so `load_massive.py --jsonl-dir` exercises the
genuine parsing path rather than a shortcut.

Turkish and Finnish rows attach case suffixes to slot values; English does
not. That reproduces, in miniature, the vocabulary blow-up the project is
about, so dataset_stats.py can be checked for a sane morphology ranking.

These are NOT results. Report real MASSIVE numbers only.

    python data/make_sample_massive.py --out data/sample_massive --n 1500
"""

from __future__ import annotations

import argparse
import json
import os
import random

SLOTS = {
    "time": {
        "en-US": ["five am", "half past eight", "ten minutes", "noon"],
        "tr-TR": ["sabah beş", "sekiz buçuk", "on dakika", "öğlen"],
        "fi-FI": ["viisi aamulla", "puoli yhdeksän", "kymmenen minuuttia"],
    },
    "date": {
        "en-US": ["friday", "tomorrow", "next week", "the third of may"],
        "tr-TR": ["cuma", "yarın", "gelecek hafta", "üç mayıs"],
        "fi-FI": ["perjantai", "huomenna", "ensi viikolla"],
    },
    "place_name": {
        "en-US": ["london", "new york", "the office"],
        "tr-TR": ["ankara", "istanbul", "ofis"],
        "fi-FI": ["helsinki", "tampere", "toimisto"],
    },
    "person": {
        "en-US": ["john", "mary smith"],
        "tr-TR": ["ahmet", "ayşe demir"],
        "fi-FI": ["matti", "liisa virtanen"],
    },
    "media_type": {
        "en-US": ["podcast", "radio"],
        "tr-TR": ["podcast", "radyo"],
        "fi-FI": ["podcast", "radio"],
    },
    "weather_descriptor": {
        "en-US": ["rain", "snow"],
        "tr-TR": ["yağmur", "kar"],
        "fi-FI": ["sade", "lumi"],
    },
}

# Case suffixes, attached to slot values in the agglutinative locales only.
SUFFIXES = {
    "tr-TR": ["", "'da", "'de", "'dan", "'den", "'ya", "'ye", "'nin"],
    "fi-FI": ["", "ssa", "sta", "lle", "lla", "n"],
    "en-US": [""],
}

CARRIERS = {
    "en-US": [
        "wake me up at {time} on {date}",
        "what is the weather like in {place_name} {date}",
        "call {person} at {time}",
        "play the {media_type} in {place_name}",
        "will there be {weather_descriptor} in {place_name} {date}",
        "remind me about {person} on {date}",
        "set an alarm for {time}",
    ],
    "tr-TR": [
        "{date} {time} beni uyandır",
        "{place_name} hava durumu {date} nasıl",
        "{time} {person} ara",
        "{place_name} {media_type} çal",
        "{date} {place_name} {weather_descriptor} olacak mı",
        "{date} {person} hatırlat",
        "{time} alarm kur",
    ],
    "fi-FI": [
        "herätä minut {time} {date}",
        "millainen sää on {place_name} {date}",
        "soita {person} {time}",
        "toista {media_type} {place_name}",
        "tuleeko {weather_descriptor} {place_name} {date}",
        "muistuta {person} {date}",
        "aseta herätys {time}",
    ],
}

INTENTS = [
    "alarm_set", "weather_query", "calling_query", "play_podcasts",
    "recommendation_events", "calendar_set", "alarm_query",
]


def inflect(value: str, locale: str, rng: random.Random) -> str:
    """Attach a case suffix to the last word of a slot value."""
    suffix = rng.choice(SUFFIXES[locale])
    if not suffix:
        return value
    words = value.split()
    words[-1] = words[-1] + suffix
    return " ".join(words)


def make_row(locale: str, idx: int, rng: random.Random) -> dict:
    carrier = rng.choice(CARRIERS[locale])
    intent = rng.choice(INTENTS)

    annot_parts: list[str] = []
    plain_parts: list[str] = []

    i = 0
    while i < len(carrier):
        if carrier[i] == "{":
            end = carrier.index("}", i)
            slot = carrier[i + 1 : end]
            value = inflect(rng.choice(SLOTS[slot][locale]), locale, rng)
            annot_parts.append(f"[{slot} : {value}]")
            plain_parts.append(value)
            i = end + 1
        else:
            end = carrier.find("{", i)
            if end == -1:
                chunk, i = carrier[i:], len(carrier)
            else:
                chunk, i = carrier[i:end], end
            if chunk.strip():
                annot_parts.append(chunk.strip())
                plain_parts.append(chunk.strip())

    annot_utt = " ".join(annot_parts)
    utt = " ".join(plain_parts)
    return {
        "id": str(idx),
        "locale": locale,
        "partition": "train",
        "scenario": intent.split("_")[0],
        "intent": intent,
        "utt": utt,
        "annot_utt": annot_utt,
        "worker_id": "synthetic",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sample_massive")
    ap.add_argument("--locales", nargs="+", default=["tr-TR", "en-US"])
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for locale in args.locales:
        rng = random.Random(args.seed + hash(locale) % 1000)
        rows = [make_row(locale, i, rng) for i in range(args.n)]
        n_test = n_dev = max(1, args.n // 10)
        for i, row in enumerate(rows):
            if i < n_test:
                row["partition"] = "test"
            elif i < n_test + n_dev:
                row["partition"] = "dev"

        path = os.path.join(args.out, f"{locale}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows)} rows -> {path}")


if __name__ == "__main__":
    main()

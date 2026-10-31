"""Method 4 - fine-tune BERTurk for token classification.

Hyperparameters follow Table 2 of Kaya & Tantug (2024), "BERT2D: Two
Dimensional Positional Embeddings for Efficient Turkish NLP", so that our
numbers are directly comparable to the BERTurk baseline reported there:

    batch size 16 | 10 epochs | max sequence length 512 | lr 5e-5

Subword alignment follows the FIRST-SUBWORD policy taught in W6:
    * a word's label goes on its first subword only;
    * every continuation piece and special token gets -100, which the loss
      ignores (-100 is NOT an extra class; O remains a real, modelled label).
Training and evaluation must use the same policy throughout.

Run this on a GPU (Colab is fine). Requires:
    pip install torch transformers datasets accelerate
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from conll import Sentence, label_set, read_conll  # noqa: E402
from evaluate import evaluate, format_report  # noqa: E402

IGNORE_INDEX = -100


def align_labels(
    tokenizer,
    tokens: list[str],
    labels: list[str],
    label2id: dict[str, int],
    max_length: int,
):
    """Tokenize one pre-split sentence and project word labels onto subwords."""
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )
    word_ids = encoding.word_ids()

    aligned: list[int] = []
    previous_word_id = None
    for word_id in word_ids:
        if word_id is None:
            aligned.append(IGNORE_INDEX)          # [CLS], [SEP], padding
        elif word_id != previous_word_id:
            aligned.append(label2id[labels[word_id]])   # first subword
        else:
            aligned.append(IGNORE_INDEX)          # continuation subword
        previous_word_id = word_id

    encoding["labels"] = aligned
    return encoding


def build_dataset(
    sentences: list[Sentence], tokenizer, label2id: dict[str, int], max_length: int
):
    from torch.utils.data import Dataset

    class TokenClassificationDataset(Dataset):
        def __init__(self, data):
            self.examples = [
                align_labels(tokenizer, toks, labs, label2id, max_length)
                for toks, labs in data
            ]

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            return {k: v for k, v in self.examples[idx].items()}

    return TokenClassificationDataset(sentences)


def decode_predictions(
    logits: np.ndarray,
    label_ids: np.ndarray,
    id2label: dict[int, str],
) -> tuple[list[list[str]], list[list[str]]]:
    """Drop every -100 position, so scoring happens on real words only."""
    preds = np.argmax(logits, axis=-1)
    gold_out, pred_out = [], []
    for pred_row, gold_row in zip(preds, label_ids):
        g, p = [], []
        for pi, gi in zip(pred_row, gold_row):
            if gi == IGNORE_INDEX:
                continue
            g.append(id2label[int(gi)])
            p.append(id2label[int(pi)])
        gold_out.append(g)
        pred_out.append(p)
    return gold_out, pred_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/wikiner")
    ap.add_argument("--model", default="dbmdz/bert-base-turkish-cased")
    ap.add_argument("--out", default="results/bert")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--final-test", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)

    train = read_conll(os.path.join(args.data, "train.conll"))
    dev = read_conll(os.path.join(args.data, "dev.conll"))
    test_path = os.path.join(args.data, "test.conll")
    test = read_conll(test_path) if os.path.exists(test_path) else []

    labels = label_set(train)
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    print(f"{len(labels)} labels: {labels}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=len(labels),
        id2label=id2label,   # saved with the model so spans can be decoded later
        label2id=label2id,
    )

    train_ds = build_dataset(train, tokenizer, label2id, args.max_length)
    dev_ds = build_dataset(dev, tokenizer, label2id, args.max_length)

    def compute_metrics(eval_pred):
        logits, label_ids = eval_pred
        gold, pred = decode_predictions(logits, label_ids, id2label)
        s = evaluate(gold, pred)
        return {
            "precision": s["precision"],
            "recall": s["recall"],
            "f1": s["f1"],
            "token_accuracy": s["token_accuracy"],
        }

    training_args = TrainingArguments(
        output_dir=args.out,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=100,
        seed=args.seed,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    os.makedirs(args.out, exist_ok=True)
    results = {}

    dev_output = trainer.predict(dev_ds)
    gold, pred = decode_predictions(
        dev_output.predictions, dev_output.label_ids, id2label
    )
    dev_scores = evaluate(gold, pred)
    print(format_report("bert / dev", dev_scores))
    results["dev"] = dict(dev_scores)
    with open(
        os.path.join(args.out, "pred_bert_dev.json"), "w", encoding="utf-8"
    ) as fh:
        json.dump(
            [
                {"tokens": t, "gold": g, "pred": p}
                for (t, _), g, p in zip(dev, gold, pred)
            ],
            fh,
            ensure_ascii=False,
        )

    if args.final_test and test:
        test_ds = build_dataset(test, tokenizer, label2id, args.max_length)
        test_output = trainer.predict(test_ds)
        gold, pred = decode_predictions(
            test_output.predictions, test_output.label_ids, id2label
        )
        test_scores = evaluate(gold, pred)
        print(format_report("bert / TEST", test_scores))
        results["test"] = dict(test_scores)
        with open(
            os.path.join(args.out, "pred_bert_test.json"), "w", encoding="utf-8"
        ) as fh:
            json.dump(
                [
                    {"tokens": t, "gold": g, "pred": p}
                    for (t, _), g, p in zip(test, gold, pred)
                ],
                fh,
                ensure_ascii=False,
            )

    trainer.save_model(os.path.join(args.out, "final"))
    tokenizer.save_pretrained(os.path.join(args.out, "final"))
    with open(
        os.path.join(args.out, "metrics_bert.json"), "w", encoding="utf-8"
    ) as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"\nsaved model and metrics to {args.out}/")


if __name__ == "__main__":
    main()

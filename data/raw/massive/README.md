# MASSIVE 1.1 — Original dataset files

Source: Amazon Science's [MASSIVE dataset](https://github.com/alexa/massive).

Copyright Amazon.com Inc. or its affiliates.

## Included files

- `en-US.jsonl`: original English records.
- `tr-TR.jsonl`: original Turkish records.
- `LICENSE`: unchanged license supplied with the dataset.

Each language contains 16,521 records, including the original train,
development and test partitions. The JSONL files are unchanged copies
from the official MASSIVE 1.1 archive:

https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz

## License and preprocessing

The dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
See [LICENSE](LICENSE) for the supplied terms.

The processed datasets in [data/massive](../../massive/) derive from
these records. The [preprocessing script](../../load_massive.py)
converts inline slot annotations into token-level BIO labels and
writes CoNLL files while retaining the original dataset partitions.

This dataset license does not license the project's source code.

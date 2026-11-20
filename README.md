# SMS Spam Classifier

A small multinomial Naive Bayes classifier implemented from scratch in Python. It trains on the included UCI SMS Spam Collection and labels text files placed in the project's `test/` directory.

## Requirements

- Python 3
- No third-party Python packages are required

## Run

From the project directory, run:

```bash
python3 classifier.py
```

The program trains on `data/SMSSpamCollection` and prints the number of training messages and vocabulary size. It then reads each `email*.txt` file in `test/` and prints a prediction:

- `0` means ham (not spam)
- `1` means spam

For example, create `test/email.txt` or `test/email1.txt`, then run the command again. Files are read as UTF-8; invalid byte sequences are replaced. The classifier can also be launched from another working directory and will still use this project's `data/` and `test/` paths.

If there are no matching files, the program reports that no `email*.txt` files were found. Add one or more text files with those names under `test/` to get predictions.

## How it works

The script tokenizes messages into lowercase words and numbers, fits a two-class multinomial Naive Bayes model, and uses add-one (Laplace) smoothing. Tokens not seen during training are ignored when scoring.

## Project files

- `classifier.py` — training and prediction code
- `data/SMSSpamCollection` — tab-separated training data (`ham`/`spam` label followed by message)
- `test/` — directory for input text files named `email*.txt`
- `DATASET_ATTRIBUTION.txt` — source and attribution for the training corpus
- `Details.txt` — assignment identity fields
- `make_report.py`, `Report.odg` — report-generation script and report

## Dataset attribution

The training corpus is the SMS Spam Collection by Tiago A. Almeida and José María Gómez Hidalgo (2011), distributed by the UCI Machine Learning Repository as dataset 228 under CC BY 4.0. See [`DATASET_ATTRIBUTION.txt`](DATASET_ATTRIBUTION.txt) for the dataset page, DOI, and attribution details.

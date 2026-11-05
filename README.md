# JARVIS-Python-logreg

Retrospective systematic-review screening simulations using TF-IDF, R/P/I/C/O/S responses and an ensemble of three logistic-regression models. Each script validates one prepared dataset at a time.

## Project layout

```text
data/                 Prepared pickle datasets, stopwords and checksums
scripts/
  validate_normal.py  Fixed Include/Exclude weights
  validate_adaptive.py Outcome-dependent weights
  common.py           Shared preprocessing and screening workflow
tests/                Small, data-independent checks
results/              Generated for each run (not tracked by Git)
```

The datasets are pre-labelled validation data, not a live screening application. The scripts require `abstract`, `R`, `P`, `I`, `C`, `O`, `S`, `totalscore`, `screening` and `FTscreening` in each trusted pickle. The portion-size dataset is compressed with gzip and has its unused SVD columns removed; SVD columns in the other datasets are ignored. The scripts process the abstracts directly. See [data/README.md](data/README.md) for data provenance and reuse information.

## Set up Python

The scripts were tested with **Python 3.12.16** and the versions in `requirements.txt`. The prepared files were created using pandas 3.x and may not load in an older environment.

For a new environment on macOS or Linux, from the project folder:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate` instead. Install Spyder separately if you want its editor and plot pane, and configure Spyder to use the Python interpreter containing the requirements.

## Run one dataset

In **Spyder**, open either `scripts/validate_normal.py` or `scripts/validate_adaptive.py`, change the `DATASET` setting near the top, and run the whole file. Plots are displayed during screening. In Spyder's IPython console preferences, choose the desired graphics backend if you prefer plots inline or in separate windows.

From a terminal in the project folder:

```bash
python scripts/validate_normal.py --dataset anticoag2
python scripts/validate_adaptive.py --dataset anticoag2
```

For a non-interactive run, add `--no-plots`. For a short trial, add `--max-rounds 3`. Use `--c 0.5` to set logistic-regression C, or `--help` to see the options.

The default dataset is `portionsize`. Dataset choices are `hypertension`, `anticoag2`, `anticoag`, `po-ur`, `cocaine`, `cyp1a2`, `portionsize`, `antiacoag` and `antiacoag2`. The `antiacoag` spellings are separate prepared cache files, not assumptions that the files are interchangeable.

## How screening works

1. Clean abstracts; create word and phrase TF-IDF with `ngram_range=(1, 3)`, `max_features=20000` and `min_df=0.07`; then min-max scale each TF-IDF feature to 0–1. TF-IDF and scaling are fitted once to all review abstracts, without screening labels. This is a **transductive** simulation: the whole review's text is available during preprocessing.
2. Combine TF-IDF with dummy variables for the six R/P/I/C/O/S responses. Start with up to 15 records from the top of `totalscore`, 10 from the score-at-most-2.5 group, and 5 from the bottom, removing duplicate IDs.
3. Fit three weighted logistic models on balanced resampled groups. Each model trains on two of the three groups; their Include probabilities are averaged. When there are too few Includes, the existing fallback reuses them across groups. If either class is entirely absent, the script stops with an error rather than fitting a two-class model.
4. Following rounds prioritize 15 highest predicted records, 5 nearest above the 0.4 threshold, and 10 nearest at or below it. If either threshold group is too small, the top 30 are selected. Groups can overlap, so fewer than 30 unique records can be added.
5. Stop when the number of remaining records predicted Include at `>= 0.4` is at most the greater of 30 or 1% of the dataset, or after 70 rounds. Screening predictions use `> 0.4`. Settings including C, weights and TF-IDF limits are at the top of `scripts/common.py`.

### Normal and adaptive weights

The normal script assigns **2 to every reviewed Include** and **1 to every reviewed Exclude**.

For the adaptive script, initial records get weights from held-out predictions: `1 + (1 - p)` for an Include, and `1 + p` for an Exclude, where `p` is the held-out Include probability. It uses three record-disjoint validation folds when each class has at least three distinct examples, two folds when each has at least two, and weight 1 for every initial record otherwise. These held-out predictions set training weights; they are not an independent evaluation of the final model.

After review, adaptive weights use the model's prediction made **before** the record was reviewed and the human screening label:

| Outcome | Weight |
| --- | ---: |
| True Include | 1 |
| True Exclude | 0.75 |
| False Include | 0.75 |
| False Exclude | 5 |

Weights are absolute; they do not multiply the normal script's Include weight of 2. Full-text labels do not set model training weights.

## Outputs and interpretation

Each run creates a uniquely named folder under `results/` containing `settings.json`, `round_metrics.csv`, `run_info.json`, and, for the adaptive script, `initial_validation.csv`. The console prints its path. The CSV is updated each round; plots are shown when enabled. `results/` is excluded from Git by `.gitignore`.

Recall and savings are **retrospective estimates** using the supplied `FTscreening` labels. Recall credits reviewed relevant records as found and assumes remaining records predicted Include will be reviewed. Estimated time savings are the proportion of all records that are full-text Excludes predicted Exclude while still unreviewed; actual screening time is not measured. These results cannot establish recall for a live review whose labels are unknown.

## Tests

From the project folder, run:

```bash
python -m unittest discover -s tests
```

## GitHub and large datasets

The compressed portion-size pickle is approximately 27 MiB; its original uncompressed file was approximately 275 MiB. `.gitattributes` configures both `data/*.pkl` and `data/*.pkl.gz` for Git LFS. Install Git LFS and run `git lfs install` **before staging or committing** the pickle files. Existing Git history may still reference the earlier large LFS file. See [GitHub's large-file guide](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github) and [Git LFS installation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage).

The repository is [gabsbarreto/JARVIS-Python-logreg](https://github.com/gabsbarreto/JARVIS-Python-logreg). To download the project and datasets, install and initialize Git LFS before cloning. If a clone contains LFS pointers instead of datasets, run `git lfs pull` inside the checkout. Use Git or GitHub Desktop to stage, commit and push future changes.

Verify the datasets' and stopwords' redistribution rights before uploading them to that public repository. The MIT licence covers the project code and documentation; the data and stopword file need their own terms.

# JARVIS-Python-logreg

JARVIS-Python-logreg evaluates a machine-assisted screening workflow for systematic reviews. It combines abstract-based TF-IDF features and R/P/I/C/O/S responses with an ensemble of three logistic-regression models. The project compares a fixed class-weighting approach with an adaptive approach that gives additional weight to missed Includes.

The included scripts run retrospective simulations on one labelled dataset at a time. They report recall, specificity and estimated screening savings using known full-text decisions; they are not a live screening application.

## Getting started

The datasets are stored with [Git LFS](https://git-lfs.com/). Install Git LFS before cloning, then run:

```bash
git lfs install
git clone https://github.com/gabsbarreto/JARVIS-Python-logreg.git
cd JARVIS-Python-logreg
```

If the dataset files contain LFS pointers after cloning, run `git lfs pull` in the project folder.

The scripts were tested with Python 3.12.16 and the package versions in `requirements.txt`. The prepared pickles use pandas 3.x and may not load in older environments. For a new environment on macOS or Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

## Running a validation

Choose one of the two scripts:

| Script | Training weights |
| --- | --- |
| `scripts/validate_normal.py` | Reviewed Includes: 2; reviewed Excludes: 1 |
| `scripts/validate_adaptive.py` | Weights depend on held-out predictions initially and on prediction outcomes in subsequent rounds |

In **Spyder**, set the interpreter to the environment containing the requirements, open the chosen script, change `DATASET` near the top if needed, and run the whole file. Plots are enabled by default. Spyder's graphics preferences control whether plots appear inline or in separate windows.

From a terminal in the project folder:

```bash
python scripts/validate_normal.py --dataset anticoag2
python scripts/validate_adaptive.py --dataset anticoag2
```

Available dataset names are `hypertension`, `anticoag`, `anticoag2`, `po-ur`, `cocaine`, `cyp1a2` and `portionsize`. The default is `portionsize`. Add `--no-plots` to suppress figures, `--max-rounds 3` for a short run, or `--c 0.5` to set the regularization parameter. Run either script with `--help` to see its options.

## How the workflow works

1. **Prepare features.** The scripts clean the abstracts and calculate TF-IDF using unigrams to trigrams, a maximum of 20,000 terms and `min_df=0.07`. Each TF-IDF feature is scaled to 0–1, then combined with dummy variables for the R/P/I/C/O/S answers. TF-IDF and scaling are fitted to all abstracts without labels; this is a transductive simulation in which the full set of review texts is available before screening.
2. **Select the initial records.** The workflow selects up to 15 records from the highest `totalscore` values, 10 from records scoring at most 2.5, and 5 from the bottom. Duplicate record IDs are removed.
3. **Fit and combine models.** Reviewed Includes and sampled Excludes are arranged in three balanced groups. Each logistic model trains on two groups, and their predicted Include probabilities are averaged. When Includes are scarce, the fallback reuses them across groups. If either class is entirely absent, the simulation stops rather than fitting a two-class model.
4. **Select the next records.** The workflow selects the 15 highest predictions, five nearest above the threshold and 10 nearest at or below it. If either threshold group is too small, it selects the top 30 instead. Overlap can result in fewer than 30 unique records.
5. **Stop screening.** The default threshold is 0.4 and the maximum is 70 rounds. The simulation stops early when the remaining number predicted Include at `>= 0.4` is at most the greater of 30 or 1% of the dataset. The classification rule uses `> 0.4`.

The default model settings are C = 0.5, `MAX_ITER = 100` and seed 123. TF-IDF, model and outcome-weight settings can be changed near the top of `scripts/common.py`.

### Adaptive weights

For initially reviewed records, the adaptive script uses record-disjoint held-out predictions. If `p` is a record's held-out Include probability, the weight is `1 + (1 - p)` for an Include and `1 + p` for an Exclude. Three folds are used when each class has at least three distinct records, two folds when each has at least two, and weight 1 for all initial records otherwise. These predictions are used to set training weights, not to evaluate the final model independently.

For later records, the script compares the ensemble's prediction made before review with the screening decision:

| Outcome | Weight |
| --- | ---: |
| True Include | 1 |
| True Exclude | 0.75 |
| False Include | 0.75 |
| False Exclude | 5 |

These are absolute training weights, not multipliers on the fixed class weights. Full-text decisions do not determine model training weights.

## Results

Each run creates a uniquely named directory under `results/`. The console prints its location. The directory contains:

- `round_metrics.csv`: recall, specificity, estimated time savings and counts for each round.
- `settings.json`: the settings used for the run.
- `run_info.json`: the stopping reason, number of completed rounds and feature counts.
- `initial_validation.csv`: the initial held-out predictions and weights for adaptive runs.

Recall and savings are retrospective estimates. Recall credits reviewed relevant records as found and assumes remaining records predicted Include will be reviewed. Estimated time savings are the proportion of all records that are full-text Excludes predicted Exclude while still unreviewed; actual screening time is not measured. Specificity is calculated for the remaining unreviewed full-text Excludes. These figures do not establish recall in a live review where labels are unknown.

## Files and tests

```text
data/                   Prepared datasets, stopwords and checksums
scripts/                Validation scripts and shared workflow
tests/                  Data-independent checks
results/                Generated run outputs (not tracked by Git)
requirements.txt        Tested Python dependencies
```

Run the tests from the project folder:

```bash
python -m unittest discover -s tests
```

## Data and licence

Each trusted dataset pickle must contain `abstract`, `R`, `P`, `I`, `C`, `O`, `S`, `totalscore`, `screening` and `FTscreening`. See [data/README.md](data/README.md) for the dataset details and checksum instructions. Only load pickle files from trusted sources: a malicious pickle can execute code.

The project code is available under the [MIT License](LICENSE). The datasets and `smart.txt` have separate, unverified reuse terms; the software licence does not establish rights to redistribute them. Check the applicable data and stopword permissions before sharing.

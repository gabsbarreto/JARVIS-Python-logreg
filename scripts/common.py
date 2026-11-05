# -*- coding: utf-8 -*-
"""Shared TF-IDF logistic-regression screening simulation.

The pipeline fits and 0–1 scales sparse TF-IDF over the review's abstracts,
then combines it with R/P/I/C/O/S responses. Normal mode weights reviewed
Includes 2 and Excludes 1. Adaptive mode uses held-out first-round weights
and outcome-specific weights for subsequently reviewed records. The two modes
share three balanced models, screening selection, evaluation, and plotting.
Full-text labels are for retrospective evaluation only, not model training.
"""

#%% Imports
from pathlib import Path
from datetime import datetime
from uuid import uuid4
import argparse
import csv
import json
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer


#%% Read file and settings
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data'
DATA_FILE = DATA_DIR / 'portionsize truncated.pkl.gz'
STOPWORDS_FILE = DATA_DIR / 'smart.txt'

# Match the current Hypertension k-fold preprocessing; adjust if needed.
TFIDF_MAX_FEATURES = 20000
TFIDF_MIN_DF = 0.07
TFIDF_MAX_DF = 1.0
TFIDF_NGRAM_RANGE = (1, 3)

# Logistic regression settings. C is the inverse regularization strength;
# it is not the same parameter as the neural network's L2 coefficient.
LOGREG_C = .5
MAX_ITER = 100
# Initial validation uses weight 1 when either reviewed class has <2 records.
TRUE_INCLUDE_WEIGHT = 1.0
TRUE_EXCLUDE_WEIGHT = 0.75
FALSE_INCLUDE_WEIGHT = 0.75  # Predicted Include, reviewed Exclude (FP)
FALSE_EXCLUDE_WEIGHT = 5.0  # Predicted Exclude, reviewed Include (FN)
THRESHOLD = 0.4
MAX_ROUNDS = 70
SEED = 123
PLOT_RESULTS = True


def minmax_scale_sparse_tfidf(matrix):
    """Scale each non-negative TF-IDF column to [0, 1], keeping CSR sparse.

    The fit uses all review abstracts without labels, like TF-IDF itself. An
    absent term has an implicit zero. A column with a positive minimum is
    present in every document, so subtracting that minimum touches only its
    existing entries. Constant features become zero; no dense matrix is made.
    """
    x = matrix.tocsr().astype(np.float64, copy=True)
    x.sum_duplicates()
    x.eliminate_zeros()
    if not np.isfinite(x.data).all() or np.any(x.data < 0):
        raise ValueError('TF-IDF features must be finite and non-negative.')
    minima = np.asarray(x.min(axis=0).toarray()).ravel()
    maxima = np.asarray(x.max(axis=0).toarray()).ravel()
    ranges = maxima - minima
    # Match MinMaxScaler's handling of constant / near-constant columns.
    denominator = np.where(ranges < 10 * np.finfo(np.float64).eps, 1.0, ranges)
    x.data = (x.data - minima[x.indices]) / denominator[x.indices]
    x.eliminate_zeros()
    return x


def build_tfidf_features(source, stopwords_file=STOPWORDS_FILE):
    """Use the same abstract cleaning and TF-IDF settings as the k-fold script.

    Fit TF-IDF and its per-feature scaling on every abstract, without labels.
    This mirrors the original workflow and makes the analysis transductive.
    """
    if 'abstract' not in source:
        raise ValueError('The source cache must contain an abstract column.')
    if source['abstract'].isna().any():
        raise ValueError('The source cache contains missing abstracts.')
    clean_abstracts = (
        source['abstract']
        .str.normalize('NFKD')
        .str.replace('â€', '-', regex=False)
        .str.replace(r'[\u00A0\u200B\u2010-\u2015\u2212]', ' ', regex=True)
        .str.replace(r'[^0-9a-zA-ZÀ-ÿ]', ' ', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
        .str.lower()
    )
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        custom_stopwords = [line.strip() for line in f if line.strip()]
    tfidf = TfidfVectorizer(
        ngram_range=TFIDF_NGRAM_RANGE, lowercase=True,
        max_features=TFIDF_MAX_FEATURES, min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF, stop_words=custom_stopwords
    )
    X_tfidf = tfidf.fit_transform(clean_abstracts)
    X_tfidf = minmax_scale_sparse_tfidf(X_tfidf)
    print(f'TF-IDF matrix (feature-wise 0–1 scaled): {X_tfidf.shape[0]} records, {X_tfidf.shape[1]} features')
    return X_tfidf, tfidf.get_feature_names_out()



def prepare_data(data_file=DATA_FILE):
    """Rebuild raw sparse TF-IDF features and combine with PICO dummies."""
    source = pd.read_pickle(data_file)
    required = ['screening', 'FTscreening', 'totalscore',
                'R', 'P', 'I', 'C', 'O', 'S']
    missing = [name for name in required if name not in source.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    if not source.index.is_unique:
        raise ValueError('Dataset index must uniquely identify each record.')
    if source[['screening', 'FTscreening']].isna().any().any():
        raise ValueError('Screening and full-text labels must not be missing.')
    for name in ('screening', 'FTscreening'):
        if not source[name].isin(['Include', 'Exclude']).all():
            raise ValueError(f'{name} must contain only Include and Exclude.')

    X_tfidf, tfidf_terms = build_tfidf_features(source)
    # Keep labels and metadata but remove cached dimension-reduction features.
    old_components = [c for c in source.columns
                      if str(c).startswith(('svd_', 'pca_', 'PCA_'))]
    source = source.drop(columns=old_components)
    # Reorder sparse TF-IDF rows to match the priority-sorted records.
    dfcyp = source.sort_values(by='totalscore', ascending=False).copy()
    sorted_rows = source.index.get_indexer(dfcyp.index)
    if (sorted_rows < 0).any():
        raise ValueError('TF-IDF rows cannot be matched to source record indices.')
    tfidf_sorted = X_tfidf[sorted_rows]
    if not np.isfinite(tfidf_sorted.data).all():
        raise ValueError('TF-IDF feature values must be finite.')
    dfcyp = pd.get_dummies(dfcyp, columns=['R', 'P', 'I', 'C', 'O', 'S'],
                           drop_first=False, dtype='int')
    pico_columns = [c for c in dfcyp.columns
                    if any(str(c).startswith(prefix + '_') for prefix in ('R', 'P', 'I', 'C', 'O', 'S'))]
    pico_features = csr_matrix(dfcyp[pico_columns].to_numpy(dtype=np.float64))
    features = hstack([tfidf_sorted, pico_features], format='csr')
    dfcyp = pd.get_dummies(dfcyp, columns=['screening'], drop_first=False, dtype='int')
    for c in ('screening_Include', 'screening_Exclude'):
        if c not in dfcyp.columns:
            dfcyp[c] = 0
    positions = pd.Series(np.arange(len(dfcyp)), index=dfcyp.index)
    print(f'Records: {len(dfcyp)}; TF-IDF features: {len(tfidf_terms)}; '
          f'R/P/I/C/O/S features: {len(pico_columns)}')
    return source, dfcyp, features, positions, tfidf_terms, pico_columns


def easy_ensemble_predict(sampfull, unlabeled_idx, features, positions,
                          c_value=LOGREG_C,
                          max_iter=MAX_ITER):
    """Same balanced resampled folds as the source; three fresh logistic models."""
    includes = sampfull.loc[sampfull['screening_Include'] == 1]
    excludes = sampfull.loc[sampfull['screening_Include'] == 0]
    n_includes = len(includes)
    if n_includes == 0 or len(excludes) == 0:
        empty_preds = np.zeros((3, len(unlabeled_idx)))
        return empty_preds.mean(axis=0), empty_preds.T, []

    shuffled_includes = includes.sample(frac=1, random_state=123)
    shuffled_excludes = excludes.sample(frac=1, random_state=123)
    include_splits = [shuffled_includes.iloc[pos].copy()
                      for pos in np.array_split(np.arange(len(shuffled_includes)), 3)]
    standard = min(len(split) for split in include_splits) >= 2
    fold_frames = []
    if standard:
        use_replacement = len(shuffled_excludes) < sum(len(split) for split in include_splits)
        exclude_start = 0
        for fold_id, includes_fold in enumerate(include_splits, start=1):
            n_excludes = len(includes_fold)
            if use_replacement:
                excludes_fold = excludes.sample(n=n_excludes, replace=True,
                                                 random_state=456 + fold_id).copy()
            else:
                excludes_fold = shuffled_excludes.iloc[exclude_start:exclude_start+n_excludes].copy()
                exclude_start += n_excludes
            fold_frame = pd.concat([includes_fold.copy(), excludes_fold], axis=0)
            fold_frame['folds'] = fold_id
            fold_frames.append(fold_frame)
    else:
        # Preserve the original small-Include fallback, including reuse of the
        # same Includes across folds rather than inventing extra labels.
        for fold_id in (1, 2, 3):
            includes_fold = includes.copy()
            includes_fold['folds'] = fold_id
            excludes_fold = excludes.sample(n=len(includes_fold),
                                            replace=len(excludes) < len(includes_fold),
                                            random_state=123 + fold_id).copy()
            excludes_fold['folds'] = fold_id
            fold_frames.append(pd.concat([includes_fold, excludes_fold], axis=0))

    samplefull2 = pd.concat(fold_frames, axis=0)
    prediction_matrix = features[positions.loc[unlabeled_idx].to_numpy()]
    predictions, histories = [], []
    for fold_id in (1, 2, 3):
        train = samplefull2.loc[samplefull2['folds'] != fold_id]
        X_train = features[positions.loc[train.index].to_numpy()]
        y_train = train['screening_Include'].to_numpy(dtype=np.int8)
        sample_weight = train['adaptive_weight'].to_numpy(dtype=float)
        model = LogisticRegression(C=c_value, solver='liblinear',
                                   max_iter=max_iter, random_state=SEED + fold_id)
        model.fit(X_train, y_train, sample_weight=sample_weight)
        include_col = int(np.flatnonzero(model.classes_ == 1)[0])
        predictions.append(model.predict_proba(prediction_matrix)[:, include_col])
        histories.append({'fold_id': fold_id, 'n_iter': int(np.max(model.n_iter_)),
                          'training_records': len(train)})
        print(f'Fold {fold_id}: logistic regression fitted in {histories[-1]["n_iter"]} iterations')
    pred_matrix = np.vstack(predictions)
    return pred_matrix.mean(axis=0), pred_matrix.T, histories


def plot_round(results, sumtextPICOS, thresh=THRESHOLD):
    sns.histplot(data=results, x='pred1', hue='FTscreening', element='step',
                 stat='density', common_norm=False, bins=30)
    plt.axvline(x=thresh, color='blue', linestyle='--', linewidth=1.2)
    plt.show()
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    sns.lineplot(data=sumtextPICOS, x='id', y='inc_incorrect', linewidth=2,
                 color='C0', ax=ax1, label='inc_incorrect')
    sns.lineplot(data=sumtextPICOS, x='id', y='exc_incorrect', linewidth=2,
                 color='red', ax=ax2, label='exc_incorrect')
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0, top=100)
    ax1.set_xlabel('Iteration (id)')
    ax1.set_ylabel('inc_incorrect')
    ax2.set_ylabel('exc_incorrect')
    fig.tight_layout()
    plt.show()


def adaptive_weights(predicted_include, reviewed_include):
    """Return absolute training weights from the prediction made before review."""
    predicted_include = np.asarray(predicted_include, dtype=bool)
    reviewed_include = np.asarray(reviewed_include, dtype=bool)
    if predicted_include.shape != reviewed_include.shape:
        raise ValueError('Predictions and reviewed labels must have the same shape.')
    return np.where(
        reviewed_include,
        np.where(predicted_include, TRUE_INCLUDE_WEIGHT, FALSE_EXCLUDE_WEIGHT),
        np.where(predicted_include, FALSE_INCLUDE_WEIGHT, TRUE_EXCLUDE_WEIGHT)
    ).astype(float)



def initial_validation_weights(sampfull, features, positions, c_value=LOGREG_C,
                               max_iter=MAX_ITER):
    """Assign initial weights from record-disjoint, held-out predictions.

    The initial screening decisions are known. Split unique record IDs *before*
    any training-set balancing or duplication; each validation record is never
    passed to the ensemble predicting it. Use three folds with >=3 per class,
    two with >=2 per class, and neutral weight 1 otherwise. Preliminary
    training uses neutral weights. The resulting held-out scores are training
    signals, not an independent evaluation of the final weighted model.
    """
    reviewed = sampfull.loc[~sampfull.index.duplicated(keep='first')].copy()
    labels = reviewed['screening_Include'].to_numpy(dtype=np.int8)
    counts = np.bincount(labels, minlength=2)
    info = pd.DataFrame({'screening_Include': labels,
                         'validation_probability': np.nan,
                         'adaptive_weight': np.ones(len(reviewed), dtype=float)},
                        index=reviewed.index)
    if counts.min() < 2:
        print(f'Initial validation skipped: {counts[1]} Includes, {counts[0]} Excludes; '
              'at least 2 distinct records of each class are needed. Initial weights = 1.')
        return info
    n_splits = 3 if counts.min() >= 3 else 2
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    print(f'Initial validation: {counts[1]} Includes, {counts[0]} Excludes, '
          f'{n_splits} held-out folds')
    probs = np.full(len(reviewed), np.nan, dtype=float)
    for fold, (train_rows, validation_rows) in enumerate(splitter.split(np.zeros(len(labels)), labels), 1):
        train = reviewed.iloc[train_rows].copy()
        # No initial example's new weight has been estimated yet.
        train['adaptive_weight'] = 1.0
        validation_idx = reviewed.index[validation_rows]
        if not train.index.intersection(validation_idx).empty:
            raise RuntimeError('Initial validation record also appears in training.')
        mean_pred, _, _ = easy_ensemble_predict(
            train, validation_idx, features, positions, c_value, max_iter)
        p = np.asarray(mean_pred, dtype=float)
        if p.shape != (len(validation_rows),) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError('Initial held-out predictions must be finite probabilities in [0, 1].')
        probs[validation_rows] = p
        print(f'Initial validation fold {fold}: {len(train_rows)} training, '
              f'{len(validation_rows)} held-out records')
    if not np.isfinite(probs).all():
        raise RuntimeError('Initial held-out predictions did not cover all reviewed records.')
    # More influence for a record whose held-out prediction disagrees with its
    # screening label; these are absolute weights, not class-weight multipliers.
    weights = 1.0 + np.where(labels == 1, 1.0 - probs, probs)
    info['validation_probability'] = probs
    info['adaptive_weight'] = weights
    print(f'Initial validation weights: min={weights.min():.3f}, '
          f'mean={weights.mean():.3f}, max={weights.max():.3f}')
    return info


def run_screening(source, dfcyp, features, positions, *, mode,
                  max_rounds=MAX_ROUNDS, thresh=THRESHOLD, c_value=LOGREG_C,
                  max_iter=MAX_ITER, plot=PLOT_RESULTS, round_callback=None):
    """Screen one prepared dataset, retaining only current-round predictions.

    Normal uses weights Include=2/Exclude=1; adaptive starts with record-
    disjoint held-out estimates and uses the outcome weights after review.
    Full-text labels are used only for retrospective evaluation and plots.
    """
    if mode not in ('normal', 'adaptive'):
        raise ValueError("mode must be 'normal' or 'adaptive'")
    if max_rounds < 1:
        raise ValueError('max_rounds must be at least 1')
    random.seed(SEED)
    np.random.seed(SEED)
    totalpos = int((source['FTscreening'] == 'Include').sum())
    if not totalpos:
        raise ValueError('The dataset has no full-text Includes; recall is undefined.')
    sampfull = pd.concat([dfcyp.head(15),
                          dfcyp.loc[dfcyp['totalscore'] <= 2.5].head(10),
                          dfcyp.tail(5)]).drop('totalscore', axis=1)
    sampfull = sampfull.loc[~sampfull.index.duplicated(keep='first')].copy()
    if mode == 'adaptive':
        initial_validation = initial_validation_weights(sampfull, features, positions,
                                                        c_value=c_value, max_iter=max_iter)
        sampfull['adaptive_weight'] = initial_validation.loc[sampfull.index, 'adaptive_weight']
    else:
        initial_validation = None
        sampfull['adaptive_weight'] = np.where(sampfull['screening_Include'] == 1, 2.0, 1.0)
    round_records = []
    termination = 'maximum_rounds'
    for i in range(max_rounds):
        print(f'\n=== round={i + 1}, n_labeled={len(sampfull)}, '
              f'C={c_value}, mode={mode}, threshold={thresh}')
        unlabeled_idx = dfcyp.index.difference(sampfull.index)
        if len(unlabeled_idx) == 0:
            print('No unlabelled records remain.')
            termination = 'no_unlabelled_records'
            break
        mean_pred, _, histories = easy_ensemble_predict(
            sampfull, unlabeled_idx, features, positions, c_value, max_iter)
        if not histories:
            raise ValueError('Cannot train a model without at least one reviewed Include and one Exclude.')
        results = dfcyp.loc[unlabeled_idx, ['FTscreening']].copy()
        results['pred1'] = mean_pred
        results = results.sort_values('pred1', ascending=False)
        predicted_include = results['pred1'].to_numpy() > thresh
        actual_include = results['FTscreening'].to_numpy() == 'Include'
        ic = int(np.count_nonzero(actual_include & predicted_include))
        ec = int(np.count_nonzero(~actual_include & ~predicted_include))
        ei = int(np.count_nonzero(actual_include & ~predicted_include))
        ii = int(np.count_nonzero(~actual_include & predicted_include))
        row = {'id': i, 'round': i + 1, 'C': c_value, 'weight': mode,
               'thresh': thresh, 'reviewed_records': int(sampfull.index.nunique()),
               'inc_correct': ic, 'exc_correct': ec, 'inc_incorrect': ii,
               'exc_incorrect': ei, 'recall': (totalpos - ei) / totalpos,
               'specificity': ec / (ec + ii) if ec + ii else float('nan'),
               'time_saved': ec / len(source) * 100}
        round_records.append(row)
        if round_callback is not None:
            round_callback(row)
        print(f"recall={row['recall']:.4f}, specificity={row['specificity']:.4f}, "
              f"estimated time savings={row['time_saved']:.2f}%")
        if plot:
            plot_round(results, pd.DataFrame(round_records), thresh)
        if len(results.loc[results['pred1'] >= thresh]) <= max(30, len(dfcyp) / 100):
            print('Stopping: few records are predicted Include.')
            termination = 'few_predicted_includes'
            break
        if i + 1 >= max_rounds:
            print('Stopping: maximum screening rounds reached.')
            break
        top_15 = results.head(15)
        near_thresh_above = results.loc[results['pred1'] > thresh].tail(5)
        near_thresh_below = results.loc[results['pred1'] <= thresh].head(10)
        if len(near_thresh_above) < 5 or len(near_thresh_below) < 10:
            add_idx = results.head(30).index
        else:
            add_idx = pd.concat([top_15, near_thresh_above, near_thresh_below]).head(30).index
        add_idx = add_idx.drop_duplicates()
        newsamp = dfcyp.loc[add_idx].drop('totalscore', axis=1, errors='ignore').copy()
        reviewed_include = newsamp['screening_Include'].to_numpy() == 1
        if mode == 'normal':
            newsamp['adaptive_weight'] = np.where(reviewed_include, 2.0, 1.0)
        else:
            predicted_for_review = results.loc[add_idx, 'pred1'].to_numpy() > thresh
            newsamp['adaptive_weight'] = adaptive_weights(predicted_for_review, reviewed_include)
        sampfull = pd.concat([sampfull, newsamp])
        sampfull = sampfull.loc[~sampfull.index.duplicated(keep='first')].copy()
    return {'round_metrics': pd.DataFrame(round_records), 'reviewed': sampfull,
            'initial_validation': initial_validation, 'stop_reason': termination}


DATASETS = {
    'hypertension': 'hypertension truncated.pkl',
    'anticoag2': 'anticoag2 truncated.pkl',
    'anticoag': 'anticoag truncated.pkl',
    'antiacoag': 'antiacoag truncated.pkl',
    'antiacoag2': 'antiacoag2 truncated.pkl',
    'po-ur': 'PO-UR truncated.pkl',
    'cocaine': 'cocaine truncated.pkl',
    'cyp1a2': 'CYP1A2 truncated.pkl',
    'portionsize': 'portionsize truncated.pkl.gz',
}


def main(mode, default_dataset='portionsize'):
    parser = argparse.ArgumentParser(
        description=f'Retrospective TF-IDF screening with {mode} logistic-regression weights.')
    parser.add_argument('--dataset', choices=DATASETS, default=default_dataset,
                        help=f'Dataset to run (default: {default_dataset}).')
    parser.add_argument('--max-rounds', type=int, default=MAX_ROUNDS,
                        help='Maximum screening rounds (default: 70).')
    parser.add_argument('--c', type=float, default=LOGREG_C,
                        help='Logistic regression C (default: 0.5).')
    parser.add_argument('--no-plots', action='store_true',
                        help='Do not display per-round plots (useful for terminal runs).')
    args = parser.parse_args()
    if args.max_rounds < 1 or args.c <= 0 or not np.isfinite(args.c):
        parser.error('--max-rounds and --c must be positive')
    data_file = DATA_DIR / DATASETS[args.dataset]
    if not data_file.is_file():
        raise FileNotFoundError(f'Dataset not found: {data_file}')
    if not STOPWORDS_FILE.is_file():
        raise FileNotFoundError(f'Stopword file not found: {STOPWORDS_FILE}')
    root = PROJECT_ROOT / 'results'
    root.mkdir(parents=True, exist_ok=True)
    output_dir = root / f'{args.dataset}_{mode}_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}'
    output_dir.mkdir()
    settings = {'dataset': DATASETS[args.dataset], 'mode': mode, 'C': args.c,
                'max_rounds': args.max_rounds, 'seed': SEED, 'threshold': THRESHOLD,
                'tfidf_max_features': TFIDF_MAX_FEATURES, 'tfidf_min_df': TFIDF_MIN_DF,
                'tfidf_max_df': TFIDF_MAX_DF, 'tfidf_ngram_range': TFIDF_NGRAM_RANGE,
                'tfidf_scaling': 'per-feature min-max, fitted on all abstracts without labels',
                'group_random_seed': 123, 'model_weight_rule':
                    ({'Include': 2.0, 'Exclude': 1.0} if mode == 'normal' else
                     {'true_include': TRUE_INCLUDE_WEIGHT, 'true_exclude': TRUE_EXCLUDE_WEIGHT,
                      'false_include': FALSE_INCLUDE_WEIGHT, 'false_exclude': FALSE_EXCLUDE_WEIGHT,
                      'initial': 'held-out 1+(1-p) for Include, 1+p for Exclude; fallback 1'}),
                'plots_enabled': PLOT_RESULTS and not args.no_plots}
    with (output_dir / 'settings.json').open('w', encoding='utf-8') as handle:
        json.dump(settings, handle, indent=2)
    print(f'Results folder: {output_dir}')
    source, dfcyp, features, positions, tfidf_terms, pico_columns = prepare_data(data_file)
    def save_round(row):
        path = output_dir / 'round_metrics.csv'
        exists = path.exists()
        with path.open('a', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
    result = run_screening(source, dfcyp, features, positions, mode=mode,
                           max_rounds=args.max_rounds, c_value=args.c,
                           plot=PLOT_RESULTS and not args.no_plots, round_callback=save_round)
    if result['initial_validation'] is not None:
        result['initial_validation'].to_csv(output_dir / 'initial_validation.csv', index_label='record_id')
    with (output_dir / 'run_info.json').open('w', encoding='utf-8') as handle:
        json.dump({'stop_reason': result['stop_reason'], 'rounds_completed': len(result['round_metrics']),
                   'tfidf_features': len(tfidf_terms), 'pico_features': len(pico_columns)}, handle, indent=2)
    print(f'Results saved in {output_dir}')
    return result

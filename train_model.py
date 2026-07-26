"""
train_model.py

Full training pipeline for PinoyBot:
  1. Load the annotated master CSV
  2. Extract features for every token (grouped by sentence, so context
     features have real neighbors to look at)
  3. Split into train / validation / test by SENTENCE (70-15-15 split)
  4. Oversample minority classes (CS, ENG) in the training set to resolve
     severe class imbalance natively through machine learning
  5. Vectorize features with DictVectorizer
  6. Train two candidate models (Decision Tree, Naive Bayes) and compare 
     them on the validation set using Macro-F1 score
  7. Report final metrics on the untouched test set
  8. Save the trained model + vectorizer to disk for pinoybot.py to load

Run with:  python3 train_model.py master_annotated_FINAL.csv
"""

import sys
import pickle
import random
from collections import defaultdict, Counter

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix
)
from features import extract_passage_features as featurize_tokens

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import numpy as np

RANDOM_SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# remaining 0.15 goes to test


def load_and_group(csv_path: str):
    """Load, validate, and group tokens/tags by sentence."""

    # keep_default_na=False prevents words such as "NA" from being
    # automatically interpreted as missing values.
    df = pd.read_csv(csv_path, keep_default_na=False)

    required_columns = {
        'sentence_id',
        'word_id',
        'word',
        'tag'
    }

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    df['sentence_id'] = pd.to_numeric(
        df['sentence_id'],
        errors='raise'
    ).astype(int)

    df['word_id'] = pd.to_numeric(
        df['word_id'],
        errors='raise'
    ).astype(int)

    df['word'] = df['word'].astype(str).str.strip()
    df['tag'] = df['tag'].astype(str).str.strip().str.upper()

    empty_words = df[df['word'] == '']
    if not empty_words.empty:
        raise ValueError(
            f"Dataset contains {len(empty_words)} empty word cells."
        )

    empty_tags = df[df['tag'] == '']
    if not empty_tags.empty:
        raise ValueError(
            f"Dataset contains {len(empty_tags)} empty tag cells."
        )

    allowed_tags = {'ENG', 'FIL', 'CS', 'OTH'}
    invalid_tags = sorted(set(df['tag']) - allowed_tags)

    if invalid_tags:
        raise ValueError(
            f"Invalid tags found: {invalid_tags}"
        )

    duplicates = df.duplicated(
        subset=['sentence_id', 'word_id'],
        keep=False
    )

    if duplicates.any():
        duplicate_rows = df.loc[
            duplicates,
            ['sentence_id', 'word_id']
        ]

        raise ValueError(
            "Duplicate sentence_id and word_id combinations found:\n"
            f"{duplicate_rows.head(10)}"
        )

    df = df.sort_values(
        ['sentence_id', 'word_id']
    ).reset_index(drop=True)

    sentences = defaultdict(
        lambda: {'tokens': [], 'tags': []}
    )

    for _, row in df.iterrows():
        sid = row['sentence_id']
        sentences[sid]['tokens'].append(row['word'])
        sentences[sid]['tags'].append(row['tag'])

    print("Dataset validation passed.")
    print(f"Valid labels: {sorted(allowed_tags)}")

    return sentences


def sentence_split(sentence_ids):
    """Shuffle and split sentence IDs into train/val/test (70/15/15)."""
    ids = list(sentence_ids)
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)

    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train:n_train + n_val])
    test_ids = set(ids[n_train + n_val:])
    return train_ids, val_ids, test_ids


def build_dataset(sentences, sentence_id_subset):
    """Turn a set of sentence_ids into (list_of_feature_dicts, list_of_labels)."""
    X, y = [], []
    for sid in sentence_id_subset:
        tokens = sentences[sid]['tokens']
        tags = sentences[sid]['tags']
        feats = featurize_tokens(tokens)
        X.extend(feats)
        y.extend(tags)
    return X, y


def oversample_training_set(X_dicts, y_labels):
    """
    Randomly oversample minority classes while preserving every
    original training example.
    """
    rng = random.Random(RANDOM_SEED)
    counts = Counter(y_labels)
    majority_count = max(counts.values())

    target_counts = {
        'FIL': majority_count,
        'ENG': majority_count,
        'OTH': majority_count,
        'CS': 500
    }

    class_indices = defaultdict(list)

    for index, label in enumerate(y_labels):
        class_indices[label].append(index)

    resampled_X = list(X_dicts)
    resampled_y = list(y_labels)

    for label, target_count in target_counts.items():
        current_count = counts.get(label, 0)

        if current_count == 0:
            raise ValueError(
                f"Cannot oversample class '{label}' because it has no examples."
            )

        number_to_add = target_count - current_count

        if number_to_add <= 0:
            continue

        extra_indices = rng.choices(
            class_indices[label],
            k=number_to_add
        )

        for index in extra_indices:
            resampled_X.append(X_dicts[index])
            resampled_y.append(y_labels[index])

    combined = list(zip(resampled_X, resampled_y))
    rng.shuffle(combined)

    shuffled_X, shuffled_y = zip(*combined)

    return list(shuffled_X), list(shuffled_y)


def ensure_int32_sparse_indices(X):
    """
    Convert a sparse matrix to CSR format with 32-bit index arrays.
    Required by some scikit-learn linear classifiers.
    """
    X = X.tocsr(copy=True)
    X.indices = X.indices.astype(np.int32, copy=False)
    X.indptr = X.indptr.astype(np.int32, copy=False)
    return X


def main(csv_path):
    print(f"Loading {csv_path} ...")
    sentences = load_and_group(csv_path)
    print(f"  {len(sentences)} sentences, "
          f"{sum(len(s['tokens']) for s in sentences.values())} tokens\n")

    train_ids, val_ids, test_ids = sentence_split(sentences.keys())
    print(f"Split: {len(train_ids)} train sentences / "
          f"{len(val_ids)} val sentences / {len(test_ids)} test sentences\n")

    X_train_dicts, y_train = build_dataset(sentences, train_ids)
    X_val_dicts, y_val = build_dataset(sentences, val_ids)
    X_test_dicts, y_test = build_dataset(sentences, test_ids)

    print(f"Original Training Distribution: {Counter(y_train)}")
    
    # Apply Random Oversampling on training data ONLY (Validation and Test stay real)
    X_train_dicts, y_train = oversample_training_set(X_train_dicts, y_train)
    print(f"Resampled Training Distribution: {Counter(y_train)}\n")

    print(f"Tokens -> train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}\n")

    print("Vectorizing features...")
    vectorizer = DictVectorizer(sparse=True)
    X_train = vectorizer.fit_transform(X_train_dicts)
    X_val = vectorizer.transform(X_val_dicts)
    X_test = vectorizer.transform(X_test_dicts)

    # Ensure compatibility with LogisticRegression and LinearSVC.
    X_train = ensure_int32_sparse_indices(X_train)
    X_val = ensure_int32_sparse_indices(X_val)
    X_test = ensure_int32_sparse_indices(X_test)
    print(f"  Feature matrix width: {X_train.shape[1]} columns\n")

    candidates = {
        'DecisionTree': DecisionTreeClassifier(
            random_state=RANDOM_SEED,
            max_depth=20,
            min_samples_leaf=5
        ),

        'MultinomialNB_alpha_1.0': MultinomialNB(
            alpha=1.0
        ),

        'LogisticRegression': LogisticRegression(
            random_state=RANDOM_SEED,
            max_iter=1000,
            solver='saga'
        ),

        'LinearSVC_C_0.1': LinearSVC(
            C=0.1,
            random_state=RANDOM_SEED
        ),

        'LinearSVC_C_0.5': LinearSVC(
            C=0.5,
            random_state=RANDOM_SEED
        ),

        'LinearSVC_C_1.0': LinearSVC(
            C=1.0,
            random_state=RANDOM_SEED
        ),

        'LinearSVC_C_2.0': LinearSVC(
            C=2.0,
            random_state=RANDOM_SEED
        ),
    }

    print("=" * 60)
    print("VALIDATION RESULTS (used to pick the better model)")
    print("=" * 60)
    val_scores = {}
    for name, clf in candidates.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)
        acc = accuracy_score(y_val, preds)
        macro_f1 = f1_score(y_val, preds, average='macro', zero_division=0)
        
        val_scores[name] = macro_f1
        print(f"\n--- {name} ---")
        print(f"Overall accuracy: {acc:.4f}  |  Macro F1: {macro_f1:.4f}")
        print(classification_report(y_val, preds, digits=3, zero_division=0))

    best_name = max(val_scores, key=val_scores.get)
    best_model = candidates[best_name]
    print(f"\nBest model on validation by macro-F1: {best_name} ({val_scores[best_name]:.4f})\n")

    print("=" * 60)
    print(f"FINAL TEST SET RESULTS -- {best_name}")
    print("=" * 60)
    test_preds = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_preds)

    print(f"Overall accuracy: {test_accuracy:.4f}")
    print(
        classification_report(
            y_test,
            test_preds,
            digits=3,
            zero_division=0
        )
    )

    # Save detailed classification metrics.
    test_report = classification_report(
        y_test,
        test_preds,
        labels=['CS', 'ENG', 'FIL', 'OTH'],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(test_report).transpose()
    report_df.to_csv('test_classification_report.csv')

    # Save the confusion matrix.
    label_order = ['CS', 'ENG', 'FIL', 'OTH']

    matrix = confusion_matrix(
        y_test,
        test_preds,
        labels=label_order
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[f'Actual_{label}' for label in label_order],
        columns=[f'Predicted_{label}' for label in label_order]
    )

    matrix_df.to_csv('test_confusion_matrix.csv')

    print("\nSaved test_classification_report.csv")
    print("Saved test_confusion_matrix.csv")

    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print(f"\nSaved vectorizer.pkl and trained_model.pkl ({best_name}).")


if __name__ == '__main__':
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'master_annotated.csv'
    main(csv_path)
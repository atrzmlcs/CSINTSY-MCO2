"""
train_model.py

Trains and evaluates candidate classifiers for PinoyBot.

The annotated dataset is split by sentence into 70% training,
15% validation, and 15% testing. The model with the highest
validation macro F1-score is evaluated on the test set and saved.
"""

import pickle
import random
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from features import featurize_tokens

RANDOM_SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
ALLOWED_TAGS = {"ENG", "FIL", "CS", "OTH"}

def load_and_group(csv_path: str):
    """Load, validate, and group the dataset by sentence."""
    df = pd.read_csv(csv_path, keep_default_na=False)

    required_columns = {"sentence_id", "word_id", "word", "tag"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    df["sentence_id"] = pd.to_numeric(
        df["sentence_id"],
        errors="raise"
    ).astype(int)

    df["word_id"] = pd.to_numeric(
        df["word_id"],
        errors="raise"
    ).astype(int)

    df["word"] = df["word"].astype(str).str.strip()
    df["tag"] = df["tag"].astype(str).str.strip().str.upper()

    empty_words = df["word"] == ""

    if empty_words.any():
        print(f"Dropped {empty_words.sum()} rows with empty words.")
        df = df.loc[~empty_words].copy()

    if (df["tag"] == "").any():
        raise ValueError("The dataset contains empty tag cells.")

    invalid_tags = sorted(set(df["tag"]) - ALLOWED_TAGS)

    if invalid_tags:
        raise ValueError(f"Invalid tags found: {invalid_tags}")

    duplicate_rows = df.duplicated(
        subset=["sentence_id", "word_id"]
    )

    if duplicate_rows.any():
        raise ValueError(
            "Duplicate sentence_id and word_id combinations found."
        )

    df = df.sort_values(
        ["sentence_id", "word_id"]
    ).reset_index(drop=True)

    sentences = defaultdict(lambda: {"tokens": [], "tags": []})

    for _, row in df.iterrows():
        sentence_id = row["sentence_id"]
        sentences[sentence_id]["tokens"].append(row["word"])
        sentences[sentence_id]["tags"].append(row["tag"])

    return sentences

def sentence_split(sentence_ids):
    """Split sentence IDs into training, validation, and test sets."""
    ids = list(sentence_ids)
    random_generator = random.Random(RANDOM_SEED)
    random_generator.shuffle(ids)

    total = len(ids)
    train_end = int(total * TRAIN_FRAC)
    validation_end = train_end + int(total * VAL_FRAC)

    train_ids = set(ids[:train_end])
    validation_ids = set(ids[train_end:validation_end])
    test_ids = set(ids[validation_end:])

    return train_ids, validation_ids, test_ids

def build_dataset(sentences, sentence_ids):
    """Extract features and labels from the selected sentences."""
    features = []
    labels = []

    for sentence_id in sorted(sentence_ids):
        tokens = sentences[sentence_id]["tokens"]
        tags = sentences[sentence_id]["tags"]

        features.extend(featurize_tokens(tokens))
        labels.extend(tags)

    return features, labels

def ensure_int32_sparse_indices(matrix):
    """Convert sparse-matrix indices for classifier compatibility."""
    matrix = matrix.tocsr(copy=True)
    matrix.indices = matrix.indices.astype(np.int32, copy=False)
    matrix.indptr = matrix.indptr.astype(np.int32, copy=False)

    return matrix

def main(csv_path: str):
    print(f"Loading {csv_path}...")

    sentences = load_and_group(csv_path)
    train_ids, validation_ids, test_ids = sentence_split(sentences.keys())

    print(
        f"Split: {len(train_ids)} training, "
        f"{len(validation_ids)} validation, "
        f"{len(test_ids)} test sentences"
    )

    train_features, train_labels = build_dataset(sentences, train_ids)
    validation_features, validation_labels = build_dataset(
        sentences,
        validation_ids
    )
    test_features, test_labels = build_dataset(sentences, test_ids)

    vectorizer = DictVectorizer(sparse=True)

    X_train = vectorizer.fit_transform(train_features)
    X_validation = vectorizer.transform(validation_features)
    X_test = vectorizer.transform(test_features)

    X_train = ensure_int32_sparse_indices(X_train)
    X_validation = ensure_int32_sparse_indices(X_validation)
    X_test = ensure_int32_sparse_indices(X_test)

    candidates = {
        "DecisionTree": DecisionTreeClassifier(
            random_state=RANDOM_SEED,
            max_depth=20,
            min_samples_leaf=5
        ),
        "MultinomialNB": MultinomialNB(
            alpha=1.0
        ),
        "LogisticRegression": LogisticRegression(
            random_state=RANDOM_SEED,
            max_iter=1000,
            solver="saga"
        ),
        "LinearSVC_C_0.1": LinearSVC(
            C=0.1,
            random_state=RANDOM_SEED,
            max_iter=10000
        ),
        "LinearSVC_C_1.0": LinearSVC(
            C=1.0,
            random_state=RANDOM_SEED,
            max_iter=10000
        )
    }

    validation_scores = {}

    print("\nValidation results")

    for name, classifier in candidates.items():
        classifier.fit(X_train, train_labels)
        predictions = classifier.predict(X_validation)

        accuracy = accuracy_score(validation_labels, predictions)
        macro_f1 = f1_score(
            validation_labels,
            predictions,
            average="macro",
            zero_division=0
        )

        validation_scores[name] = macro_f1

        print(
            f"{name}: "
            f"accuracy={accuracy:.4f}, "
            f"macro F1={macro_f1:.4f}"
        )

    best_name = max(validation_scores, key=validation_scores.get)
    best_model = candidates[best_name]

    print(f"\nSelected model: {best_name}")

    test_predictions = best_model.predict(X_test)
    test_accuracy = accuracy_score(test_labels, test_predictions)

    print(f"Test accuracy: {test_accuracy:.4f}")
    print(
        classification_report(
            test_labels,
            test_predictions,
            digits=3,
            zero_division=0
        )
    )

    with open("vectorizer.pkl", "wb") as file:
        pickle.dump(vectorizer, file)

    with open("trained_model.pkl", "wb") as file:
        pickle.dump(best_model, file)

    print("Saved vectorizer.pkl and trained_model.pkl.")

if __name__ == "__main__":
    dataset_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "master_annotated_FINAL.csv"
    )

    main(dataset_path)
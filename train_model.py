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
from sklearn.metrics import classification_report, accuracy_score, f1_score

from features import featurize_tokens

RANDOM_SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# remaining 0.15 goes to test


def load_and_group(csv_path: str):
    """Load the master CSV and group tokens/tags by sentence, in order."""
    df = pd.read_csv(csv_path)
    df['sentence_id'] = df['sentence_id'].astype(int)
    df = df.sort_values(['sentence_id', 'word_id']).reset_index(drop=True)

    sentences = defaultdict(lambda: {'tokens': [], 'tags': []})
    for _, row in df.iterrows():
        sid = row['sentence_id']
        sentences[sid]['tokens'].append(str(row['word']))
        sentences[sid]['tags'].append(str(row['tag']))

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
    Random Oversampling: Duplicates minority class tokens (CS, ENG, OTH) in the 
    training set so that all classes match the majority class (FIL) count.
    
    This fixes the class imbalance purely through dataset resampling, avoiding
    any prohibited hard-coded logic in pinoybot.py.
    """
    rng = random.Random(RANDOM_SEED)
    counts = Counter(y_labels)
    max_count = max(counts.values())

    # Group feature dict indices by class label
    class_indices = defaultdict(list)
    for idx, label in enumerate(y_labels):
        class_indices[label].append(idx)

    resampled_X, resampled_y = [], []
    for label, indices in class_indices.items():
        # Draw samples with replacement up to the majority count
        sampled_indices = rng.choices(indices, k=max_count)
        for idx in sampled_indices:
            resampled_X.append(X_dicts[idx])
            resampled_y.append(y_labels[idx])

    # Shuffle the resampled training set
    combined = list(zip(resampled_X, resampled_y))
    rng.shuffle(combined)
    shuffled_X, shuffled_y = zip(*combined)
    
    return list(shuffled_X), list(shuffled_y)


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
    print(f"  Feature matrix width: {X_train.shape[1]} columns\n")

    candidates = {
        'DecisionTree': DecisionTreeClassifier(
            random_state=RANDOM_SEED, max_depth=None
        ),
        'MultinomialNB': MultinomialNB(),
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
    print(f"Overall accuracy: {accuracy_score(y_test, test_preds):.4f}")
    print(classification_report(y_test, test_preds, digits=3, zero_division=0))

    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print(f"\nSaved vectorizer.pkl and trained_model.pkl ({best_name}).")


if __name__ == '__main__':
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'master_annotated.csv'
    main(csv_path)
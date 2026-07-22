"""
pinoybot.py

PinoyBot: Filipino Code-Switched Language Identifier

This module provides the main tagging function for the PinoyBot project, which 
identifies the language of each word in a code-switched Filipino-English text. 
The function is designed to be called with a list of tokens and returns a list 
of tags ("ENG", "FIL", "CS", or "OTH").

Model training and feature extraction are implemented in train_model.py and features.py. 
The trained model is saved and loaded here for prediction.
"""

import os
import pickle
from typing import List

from features import featurize_tokens

# Files are loaded relative to this script's location so tag_language()
# works correctly even if imported from another directory.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_THIS_DIR, 'trained_model.pkl')
_VECTORIZER_PATH = os.path.join(_THIS_DIR, 'vectorizer.pkl')

# Loaded once at import time for fast inference
with open(_MODEL_PATH, 'rb') as f:
    _MODEL = pickle.load(f)
with open(_VECTORIZER_PATH, 'rb') as f:
    _VECTORIZER = pickle.load(f)


def tag_language(tokens: List[str], debug: bool = False):
    """
    Tags each token in the input list with its predicted language using the trained ML model.

    Args:
        tokens: List of word tokens (strings).
        debug: If True, also returns the extracted feature dictionaries.

    Returns:
        If debug=False:
            List[str] of predicted tags.
        If debug=True:
            (tags, feature_dicts)
    """
    if not tokens:
        return ([], []) if debug else []

    # 1. Extract features from the input tokens using features.py
    feature_dicts = featurize_tokens(tokens, verbose=True)

    # 2. Vectorize the features into the feature matrix format expected by the model
    X = _VECTORIZER.transform(feature_dicts)

    # 3. Pure Machine Learning Prediction
    predicted = _MODEL.predict(X)

    tags = [str(tag) for tag in predicted]

    if debug:
        return tags, feature_dicts

    return tags


if __name__ == "__main__":
    example_tokens = ["may", "ilang", "beses", "nako", "nag-eat", "ng", "dinner", "HAHAHA", "XD", ".", "D0", "you", "want", "magdessert", "instead", "later", "at", "2" "pm", "?"]

    print("Tokens:", example_tokens)

    tags, features = tag_language(example_tokens, debug=True)

    print("Tags:  ", tags)

    print("\nDebug Information: Prototype")
    for token, tag, feature in zip(example_tokens, tags, features):
        print(f"{tag:<3}: {token:<15}")
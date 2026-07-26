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

from features import extract_passage_features

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


def tag_language(tokens: List[str]) -> List[str]:
    """
    Predict the language tag of every token.

    Args:
        tokens: List of token strings.

    Returns:
        List of predicted tags: ENG, FIL, CS, or OTH.
    """
    if not isinstance(tokens, list):
        raise TypeError("tokens must be provided as a list of strings.")

    if not tokens:
        return []

    if not all(isinstance(token, str) for token in tokens):
        raise TypeError("Every token must be a string.")

    # Extract the same features used during model training.
    feature_dicts = extract_passage_features(tokens)

    # Convert feature dictionaries using the trained vectorizer.
    X = _VECTORIZER.transform(feature_dicts)

    # Predict using the trained classifier.
    predicted = _MODEL.predict(X)
    tags = [str(tag) for tag in predicted]

    allowed_tags = {'ENG', 'FIL', 'CS', 'OTH'}

    if len(tags) != len(tokens):
        raise RuntimeError(
            "The number of predicted tags does not match the number of tokens."
        )

    invalid_tags = set(tags) - allowed_tags

    if invalid_tags:
        raise RuntimeError(
            f"The model returned invalid tags: {sorted(invalid_tags)}"
        )

    return tags

if __name__ == "__main__":
    example_tokens = [
        "magnificent", "magic", "nagging",
        # Suffix Traps
       # "human", "ocean", "brain", "train", 
        # Infix Traps
       # "number", "plumber", "dinosaur", "window", 
        # Hyphen Traps
      #  "co-op", "t-shirt", "x-ray", "e-mail", 
        # True Homographs
      #  "ate", "noon", "raw", "may"
    ]
    print("Tokens:", example_tokens)
    tags = tag_language(example_tokens)
    print("Tags:  ", tags)
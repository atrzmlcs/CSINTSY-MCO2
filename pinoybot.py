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


def tag_language(tokens: List[str]) -> List[str]:
    """
    Tags each token in the input list with its predicted language using the trained ML model.
    
    Args:
        tokens: List of word tokens (strings).
    Returns:
        tags: List of predicted tags ("ENG", "FIL", "CS", or "OTH"), one per token.
    """
    if not tokens:
        return []

    # 1. Extract features from the input tokens using features.py
    feature_dicts = featurize_tokens(tokens, verbose=True)

    # 2. Vectorize the features into the feature matrix format expected by the model
    X = _VECTORIZER.transform(feature_dicts)

    # 3. Pure Machine Learning Prediction: Model directly outputs the predicted tags
    predicted = _MODEL.predict(X)

    # 4. Return predictions as a list of strings
    return [str(tag) for tag in predicted]


if __name__ == "__main__":
    example_tokens = ["nag-eat", "magboxing", "HAHAHA", "hahaha", "si", "Bogart", "sa", "may", "EDSA", "."]
    print("Tokens:", example_tokens)
    tags = tag_language(example_tokens)
    print("Tags:  ", tags)
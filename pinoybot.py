# Pat

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

    # 1. Extract pure word-level features from input tokens
    feature_dicts = featurize_tokens(tokens)

    # 2. Vectorize features into the 8,161-column matrix expected by the model
    X = _VECTORIZER.transform(feature_dicts)

    # 3. Predict language tags using the trained Decision Tree model
    predicted = _MODEL.predict(X)

    # 4. Return predictions as a list of strings
    return [str(tag) for tag in predicted]


if __name__ == "__main__":
    example_tokens = ["human", "ocean", "brain", "train", "number", "plumber", "dinosaur", "window", "co-op", "t-shirt", "x-ray", "e-mail", "ate", "noon", "raw", "may"]
    #example_tokens = ["may", "ilang", "beses", "nako", "nag-eat", "ng", "dinner", "HAHAHAA", "XD", ".", "D0", "you", "want", "magdessert", "instead", "later", "at", "2" "pm", "?"]
    
    print("Tokens:", example_tokens)
    tags = tag_language(example_tokens)
    print("\nDebug Information: Pat Prototype")
    for token, tag in zip(example_tokens, tags):
        print(f"{tag:<3}: {token:<15}")
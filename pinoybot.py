"""
pinoybot.py

PinoyBot: Filipino Code-Switched Language Identifier

This module provides the main tagging function for the PinoyBot project, which identifies the language of each word in a code-switched Filipino-English text. The function is designed to be called with a list of tokens and returns a list of tags ("ENG", "FIL", "CS", or "OTH").

Model training and feature extraction should be implemented in a separate script. The trained model should be saved and loaded here for prediction.
"""

import os
import pickle
from typing import List

from features import featurize_tokens

# Files are loaded relative to this script's own location, not the current
# working directory -- so tag_language() still works correctly even if
# it's imported and called from a different folder (e.g. by the grader).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_THIS_DIR, 'trained_model.pkl')
_VECTORIZER_PATH = os.path.join(_THIS_DIR, 'vectorizer.pkl')

# Loaded once at import time, not on every call to tag_language() -- reading
# a pickle file from disk is slow, and tag_language() may get called many
# times (once per passage) during grading.
with open(_MODEL_PATH, 'rb') as f:
    _MODEL = pickle.load(f)
with open(_VECTORIZER_PATH, 'rb') as f:
    _VECTORIZER = pickle.load(f)


# Main tagging function
def tag_language(tokens: List[str]) -> List[str]:
    """
    Tags each token in the input list with its predicted language.
    Args:
        tokens: List of word tokens (strings).
    Returns:
        tags: List of predicted tags ("ENG", "FIL", "CS", or "OTH"), one per token.
    """
    if not tokens:
        return []

    # 1. Model and vectorizer are already loaded above (_MODEL, _VECTORIZER).

    # 2. Extract features from the input tokens to create the feature matrix.
    #    featurize_tokens() returns one feature dict per token, using the
    #    SAME feature logic that was used during training (features.py is
    #    shared between train_model.py and this file on purpose).
    feature_dicts = featurize_tokens(tokens)
    X = _VECTORIZER.transform(feature_dicts)

    # 3. Use the model to predict the tags for each token.
    predicted = _MODEL.predict(X)

    # 4. Convert the predictions to a list of strings ("ENG", "FIL", "CS", or "OTH").
    tags = [str(tag) for tag in predicted]

    # 5. Return the list of tags.
    return tags


if __name__ == "__main__":
    # Example usage
    example_tokens = ["may", "food", "ka", "ba", "diyan", "if", "you", "do", "may", "i", "have" "?"]
    print("Tokens:", example_tokens)
    tags = tag_language(example_tokens)
    print("Tags:  ", tags)

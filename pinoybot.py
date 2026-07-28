"""
pinoybot.py

PinoyBot: Filipino Code-Switched Language Identifier

This module provides the main tagging function for the PinoyBot project, which
identifies the language of each word in a code-switched Filipino-English text.
The function is designed to be called with a list of tokens and returns a list
of tags ("ENG", "FIL", "CS", or "OTH").

Model training and feature extraction should be implemented in a separate
script. The trained model should be saved and loaded here for prediction.
"""

import os
import pickle
from typing import List

from features import featurize_tokens


# 1. Load the trained model and vectorizer from disk
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "trained_model.pkl")
VECTORIZER_PATH = os.path.join(CURRENT_DIR, "vectorizer.pkl")

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

with open(VECTORIZER_PATH, "rb") as file:
    vectorizer = pickle.load(file)


# Main tagging function
def tag_language(tokens: List[str]) -> List[str]:
    """
    Tags each token in the input list with its predicted language.

    Args:
        tokens: List of word tokens (strings).

    Returns:
        tags: List of predicted tags ("ENG", "FIL", "CS", or "OTH"),
        one per token.
    """
    if not tokens:
        return []

    # 2. Extract features from the input tokens
    features = featurize_tokens(tokens)

    # 3. Convert the features into the feature matrix
    feature_matrix = vectorizer.transform(features)

    # 4. Use the trained model to predict the tags
    predicted = model.predict(feature_matrix)

    # 5. Convert and return the predictions as a list of strings
    tags = [str(tag) for tag in predicted]

    return tags

if __name__ == "__main__":
    example_tokens = ["human", "ocean", "brain", "train", "number", "plumber", "dinosaur", "window", "co-op", "t-shirt", "x-ray", "e-mail", "ate", "noon", "raw", "may"]
    # example_tokens = ["may", "ilang", "beses", "nako", "nag-eat", "ng", "dinner", "HAHAHAA", "XD", ".", "D0", "you", "want", "magdessert", "instead", "later", "at", "2" "pm", "?"]
    
    print("Tokens:", example_tokens)
    tags = tag_language(example_tokens)
    print("Tags:", tags)
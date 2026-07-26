"""
features.py - Feature Extraction Module for PinoyBot Language Identifier
Extracts pure word-level features (structure, capitalization, character n-grams, 
and Filipino morphology) for tagging ENG, FIL, CS, and OTH tokens.
"""

from typing import Dict, Any, List

# Retained ONLY for strict hyphenated CS detection (e.g., "nag-boxing").
# We no longer use this for unhyphenated starts_with checks to avoid false cognates!
FIL_PREFIXES = {
    'nag', 'mag', 'ipag', 'pinag', 'naka', 'paki', 
    'kapa', 'manga', 'pinaka', 'maki', 'ipang', 'pagka',
    'ipinag', 'pina', 'pagka', 'pagkaka' , 'mala', 'pang',
    'nakaka', 'ma', 'pa'
}

# Common Filipino function words and particles
FIL_PARTICLES = {
    'sa', 'ng', 'na', 'pa', 'mga', 'rin', 'din', 'ang', 
    'si', 'ni', 'kay', 'at', 'o', 'ba', 'po', 'ho',
    # Added after checking actual frequency counts in the training data --
    # these are all closed-class pronouns/conjunctions/particles, not
    # regular content words, so they belong here rather than being left
    # to the n-gram features to catch.
    'ko', 'ako', 'ay', 'mo', 'lang', 'kung', 'para',
    'naman', 'may', 'ito', 'ka', 'niya', 'yung', 'siya', 'hindi',
}


def extract_word_features(word: str) -> Dict[str, Any]:
    """
    Extracts a feature dictionary for a single token.
    No sentence context required. No brittle consonant cluster checks.
    
    Args:
        word (str): The token string to extract features from.
        
    Returns:
        Dict[str, Any]: Feature dictionary ready for DictVectorizer.
    """
    w_lower = word.lower()
    length = len(word)
    
    # =========================================================
    # Group 1: Basic Token Structure & Character Ratios
    # =========================================================
    vowel_count = sum(1 for c in w_lower if c in 'aeiou')
    vowel_ratio = vowel_count / length if length > 0 else 0.0
    
    features = {
        'length': length,
        'is_alpha': word.isalpha(),
        'is_punct_only': length > 0 and not any(c.isalnum() for c in word),
        'has_digit': any(c.isdigit() for c in word),
        'has_hyphen': '-' in word,
        'vowel_ratio': round(vowel_ratio, 3),
    }

    # =========================================================
    # Group 2: Capitalization Patterns
    # =========================================================
    letters = [c for c in word if c.isalpha()]
    n_letters = len(letters)
    
    features.update({
        'is_title_cased': length > 0 and word[0].isupper() and (length == 1 or word[1:].islower()),
        'is_all_caps': n_letters > 1 and word.isupper(),
        'upper_ratio': round(sum(1 for c in word if c.isupper()) / length, 3) if length > 0 else 0.0,
    })

    # =========================================================
    # Group 3: Sub-word Character n-Grams (Prefix/Suffix Slices)
    # Allows Decision Trees to discover sub-word patterns natively
    # =========================================================
    features.update({
        'prefix_2': w_lower[:2] if length >= 2 else w_lower,
        'prefix_3': w_lower[:3] if length >= 3 else w_lower,
        'suffix_2': w_lower[-2:] if length >= 2 else w_lower,
        'suffix_3': w_lower[-3:] if length >= 3 else w_lower,
        'suffix_4': w_lower[-4:] if length >= 4 else w_lower,
    })

    # =========================================================
    # Group 4: Tagalog Morphology & Code-Switching Indicators
    # =========================================================
    
    # Detect formal hyphenated CS (e.g., "nag-march", "pina-explain")
    has_fil_prefix_before_hyphen = False
    if '-' in w_lower:
        prefix = w_lower.split('-')[0]
        if prefix in FIL_PREFIXES:
            has_fil_prefix_before_hyphen = True

    # Detect native infix patterns (-um-, -in-) inside verb stems
    has_fil_infix = False
    if length > 3 and (w_lower[1:3] in ['um', 'in'] or w_lower[2:4] in ['um', 'in']):
        has_fil_infix = True

    # Check for internet expressive laughter (OTH)
    is_laugh = any(laugh in w_lower for laugh in ['haha', 'hehe', 'hihi', 'huhu'])

    features.update({
        'has_fil_prefix_before_hyphen': has_fil_prefix_before_hyphen,
        'has_fil_infix': has_fil_infix,
        'is_fil_particle': w_lower in FIL_PARTICLES,
        'is_laugh': is_laugh,
    })

    # ---------------------------------------------------------------
    # Unhyphenated CS detection: strip a matching Filipino prefix, then
    # check if what's LEFT looks like an English root. This is exposed
    # as features (evidence the model weighs), NOT a hard startswith
    # rule -- so a genuine Filipino word that happens to start with
    # 'ma' or 'pa' (a "false cognate") isn't automatically forced into
    # CS. The model decides how much to trust this alongside everything
    # else, the same way it already handles has_fil_prefix_before_hyphen.
    root = w_lower
    prefix_stripped = False
    matches = [p for p in FIL_PREFIXES if w_lower.startswith(p) and len(w_lower) > len(p)]
    if matches:
        best_prefix = max(matches, key=len)
        root = w_lower[len(best_prefix):]
        prefix_stripped = True

    root_letters = [c for c in root if c.isalpha()]
    root_n_letters = len(root_letters)
    root_vowel_ratio = (sum(1 for c in root_letters if c in 'aeiou') / root_n_letters
                         if root_n_letters > 0 else 0.0)
    root_has_eng_only_letters = any(c in 'cfjqvxz' for c in root)

    features.update({
        'prefix_stripped': prefix_stripped,
        'root_vowel_ratio': round(root_vowel_ratio, 3),
        'root_has_eng_only_letters': root_has_eng_only_letters,
    })

    return features


def get_context_features(tokens: List[str], i: int) -> Dict[str, Any]:
    """
    Extracts features based on the tokens surrounding position i.
    This is the feature type explicitly suggested in the spec
    ("properties of words surrounding it in the sentence") that was
    missing from the word-only feature set above.

    Args:
        tokens: the full list of tokens in the sentence/passage.
        i: index of the token we're extracting context for.

    Returns:
        Dict[str, Any]: context feature dictionary.
    """
    n = len(tokens)
    prev_word = tokens[i - 1].lower() if i > 0 else '<START>'
    next_word = tokens[i + 1].lower() if i < n - 1 else '<END>'

    feats = {
        'is_first_in_sentence': i == 0,
        'is_last_in_sentence': i == n - 1,
    }

    for label, neighbor in [('prev', prev_word), ('next', next_word)]:
        if neighbor in ('<START>', '<END>', ''):
            feats[f'{label}_len'] = 0
            feats[f'{label}_is_particle'] = False
            feats[f'{label}_ends_in_vowel'] = False
        else:
            feats[f'{label}_len'] = len(neighbor)
            feats[f'{label}_is_particle'] = neighbor in FIL_PARTICLES
            feats[f'{label}_ends_in_vowel'] = neighbor[-1] in 'aeiou'

    return feats


def extract_passage_features(tokens: List[str]) -> List[Dict[str, Any]]:
    """
    Converts a list of passage tokens into a list of feature dictionaries.
    Each token's features combine its own word-level features with
    context features derived from its neighbors in the sentence.
    
    Args:
        tokens (List[str]): List of word tokens.
        
    Returns:
        List[Dict[str, Any]]: Feature dictionaries for every token in the passage.
    """
    result = []
    for i, token in enumerate(tokens):
        feats = extract_word_features(token)
        feats.update(get_context_features(tokens, i))
        result.append(feats)
    return result


if __name__ == "__main__":
    # Test suite for quick feature validation
    sample_tokens = ["nag-march", "nagmarch", "kumain", "DLSU", "HAHAHA", "construction"]
    print("Feature Extraction Sanity Check:\n")
    for tok in sample_tokens:
        print(f"Token: '{tok}'")
        feats = extract_word_features(tok)
        for k, v in feats.items():
            print(f"  {k}: {v}")
        print("-" * 40)
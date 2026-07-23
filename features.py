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
    'si', 'ni', 'kay', 'at', 'o', 'ba', 'po', 'ho'
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

    return features


def extract_passage_features(tokens: List[str]) -> List[Dict[str, Any]]:
    """
    Converts a list of passage tokens into a list of feature dictionaries.
    
    Args:
        tokens (List[str]): List of word tokens.
        
    Returns:
        List[Dict[str, Any]]: Feature dictionaries for every token in the passage.
    """
    return [extract_word_features(token) for token in tokens]


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
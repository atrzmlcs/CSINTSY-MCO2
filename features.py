"""
features.py

Word-level feature extraction for PinoyBot.

IMPORTANT: this module is shared between training (train_model.py) and
inference (pinoybot.py). Both need to turn a raw token into *exactly* the
same feature representation, or the trained model won't understand what
it's being asked to predict on.

Design notes (why these features, and what's deliberately left out):
- No dictionary lookups of whole words are used anywhere here -- per the
  spec, we are not allowed to infer a word's language by checking if it
  appears in a Filipino/English word list. Everything below is either a
  surface-level property of the word's spelling, or a *pattern* (affix,
  reduplication) rather than a whole-word lookup.
- Context features only look at the neighboring tokens' surface form
  (e.g. what character a neighboring word starts/ends with), never at a
  neighboring word's *predicted* tag. That's on purpose: at inference
  time in tag_language(), we only ever get a plain list of tokens with
  no tags attached, so any feature we engineer has to be computable from
  that same information alone.
"""

from typing import List, Dict, Any

VOWELS = set('aeiouAEIOU')

# Common Filipino affixes. This is a *morphological pattern* list (i.e. we
# check if a word starts/contains these letter patterns), not a dictionary
# of whole Filipino words -- so it doesn't fall under the "no dictionary
# lookup" restriction. These were chosen by inspecting recurring patterns
# during annotation (e.g. "nag-", "mag-", "pinag-" showing up constantly
# in CS words like "nagchat", "nag-log out").
FIL_PREFIXES = [
    'nakikipag', 'makipag', 'pinagka', 'nagpa', 'pinag', 'napaka', 'pinaka',
    'nagka', 'naka', 'maka', 'mapa', 'ipina', 'ipa', 'nag', 'mag', 'pag',
    'may', 'wala', 'hindi', 'ma', 'na', 'um', 'in', 'i', 'pa',
]

# Common English suffixes -- again a pattern check, not a dictionary. Used
# for the general 'ends_with_eng_suffix' feature.
ENG_SUFFIXES = [
    'tion', 'sion', 'ness', 'ment', 'able', 'ible', 'ing', 'ed', 'ly', 'er',
    'est', 's',
]

# A STRICTER subset used only for the fused-code-switch heuristic below.
# Suffixes like "-ing", "-ed", "-er", "-s" are too common in native
# Filipino spelling on their own (e.g. "maging", "magiging", "maraming"
# all end in "-ing" but are plain Filipino words) to reliably signal an
# English root, so they're excluded here to cut down false positives.
STRONG_ENG_SUFFIXES = ['tion', 'sion', 'ness', 'ment', 'able', 'ible']


def is_reduplicated(word: str) -> bool:
    """
    Detects Filipino reduplication, e.g. "araw-araw", "lakad-lakad" --
    a hallmark of Filipino morphology mentioned in the project spec.
    """
    if '-' in word:
        parts = word.split('-')
        if len(parts) == 2 and parts[0].lower() == parts[1].lower() and len(parts[0]) > 1:
            return True
    return False


def has_fil_prefix_before_hyphen(word: str) -> bool:
    """
    Detects the classic intra-word code-switch shape: Filipino affix +
    hyphen + (usually English) root, e.g. "nag-chat", "pina-explain".
    """
    if '-' in word:
        prefix = word.split('-', 1)[0].lower()
        return prefix in FIL_PREFIXES
    return False


# Letters that are rare in native Filipino/Tagalog word roots (the original
# Filipino "abakada" alphabet excluded these) but common in English roots.
# Used only as a letter-pattern signal, not a dictionary lookup.
ENG_LETTER_HINTS = set('cfjqvxz')


def fil_prefix_and_remainder(word_lower: str):
    """
    Finds the longest FIL_PREFIXES match at the start of the word, and
    returns (matched_prefix, remaining_substring). Used to spot fused
    code-switched words like "nagshift" (prefix "nag" + remainder "shift")
    that don't have a hyphen -- unlike "nag-chat", which has_fil_prefix_
    before_hyphen already catches.
    """
    best = ''
    for p in FIL_PREFIXES:
        if word_lower.startswith(p) and len(p) > len(best):
            best = p
    return best, word_lower[len(best):]


def word_features(word: str) -> Dict[str, Any]:
    """Features derived purely from the word's own spelling."""
    w = word
    wl = word.lower()
    length = len(w)
    letters = [c for c in w if c.isalpha()]
    n_letters = len(letters)
    vowel_count = sum(1 for c in letters if c.lower() in 'aeiou')

    prefix_match, remainder = fil_prefix_and_remainder(wl)
    remainder_len = len(remainder)
    remainder_has_eng_letter = any(c in ENG_LETTER_HINTS for c in remainder)
    remainder_ends_strong_eng_suffix = any(remainder.endswith(s) for s in STRONG_ENG_SUFFIXES)
    # A fused code-switch candidate: a Filipino prefix attached directly
    # (no hyphen) to a remainder that "looks" English -- e.g. "nagshift"
    # (prefix "nag" + remainder "shift", which has an 'f'), "naglunch"
    # (remainder "lunch" has a 'c'). Deliberately conservative (uses only
    # the rare-letter check and the strong suffix subset) to avoid
    # flagging ordinary Filipino words like "maging" or "maraming".
    looks_like_fused_cs = bool(prefix_match) and remainder_len >= 3 and (
        remainder_has_eng_letter or remainder_ends_strong_eng_suffix
    )

    return {
        'length': length,
        'is_alpha': w.isalpha(),
        'is_punct_only': length > 0 and not any(c.isalnum() for c in w),
        'has_digit': any(c.isdigit() for c in w),
        'has_hyphen': '-' in w,
        'has_period': '.' in w,
        # capitalization cues -- useful for spotting names/abbreviations (OTH)
        'is_capitalized': length > 0 and w[0].isupper() and (length == 1 or w[1:].islower()),
        'is_all_caps': n_letters > 1 and w.isupper(),
        'upper_ratio': (sum(1 for c in w if c.isupper()) / length) if length else 0.0,
        # letter-arrangement cues
        'vowel_ratio': (vowel_count / n_letters) if n_letters else 0.0,
        'is_reduplicated': is_reduplicated(w),
        'has_fil_prefix_before_hyphen': has_fil_prefix_before_hyphen(w),
        'starts_with_fil_prefix': any(wl.startswith(p) for p in FIL_PREFIXES),
        'ends_with_eng_suffix': any(wl.endswith(s) for s in ENG_SUFFIXES),
        # fused (non-hyphenated) code-switch cues
        'fil_prefix_len': len(prefix_match),
        'remainder_len': remainder_len,
        'remainder_has_eng_letter': remainder_has_eng_letter,
        'remainder_vowel_ratio': (sum(1 for c in remainder if c in 'aeiou') / remainder_len) if remainder_len else 0.0,
        'looks_like_fused_cs': looks_like_fused_cs,
        # character n-gram-ish cues, as categorical features (DictVectorizer
        # will one-hot encode these) -- a lightweight stand-in for full
        # character n-gram vectorization
        'prefix2': wl[:2],
        'prefix3': wl[:3],
        'suffix2': wl[-2:],
        'suffix3': wl[-3:],
    }


def context_features(tokens: List[str], i: int) -> Dict[str, Any]:
    """Features derived from the token's position and its neighbors' surface form."""
    n = len(tokens)
    
    # Extract lowercased neighboring surface words
    curr_word = tokens[i].lower()
    prev_word = tokens[i - 1].lower() if i > 0 else '<START>'
    next_word = tokens[i + 1].lower() if i < n - 1 else '<END>'
    
    # 2-step neighbors (window of [-2, +2])
    prev2_word = tokens[i - 2].lower() if i > 1 else '<START>'
    next2_word = tokens[i + 2].lower() if i < n - 2 else '<END>'

    return {
        'is_first_in_sentence': i == 0,
        'is_last_in_sentence': i == n - 1,
        'position_ratio': (i / (n - 1)) if n > 1 else 0.0,
        'sentence_length': n,
        
        # --- 1. NEIGHBOR WORD FEATURES ---
        'prev_word': prev_word,
        'next_word': next_word,
        'prev2_word': prev2_word,
        'next2_word': next2_word,
        
        # --- 2. SURFACE BIGRAM ANCHORS ---
        # Explicitly pairs the target word with its left and right neighbors
        'prev_curr_bigram': f"{prev_word}_{curr_word}",
        'curr_next_bigram': f"{curr_word}_{next_word}",
        
        # --- 3. CHARACTER BOUNDARIES (Original features) ---
        'prev_last_char': prev_word[-1] if prev_word and prev_word != '<START>' else '',
        'next_first_char': next_word[0] if next_word and next_word != '<END>' else '',
        'prev_is_punct_only': bool(prev_word) and not any(c.isalnum() for c in prev_word),
        'next_is_punct_only': bool(next_word) and not any(c.isalnum() for c in next_word),
    }


def extract_features(tokens: List[str], i: int) -> Dict[str, Any]:
    """Full feature dict for token i within a token list (a sentence/passage)."""
    feats = word_features(tokens[i])
    feats.update(context_features(tokens, i))
    return feats


def featurize_tokens(tokens: List[str]) -> List[Dict[str, Any]]:
    """
    Convenience wrapper: given a list of tokens (one passage, as passed into
    tag_language()), return a list of feature dicts, one per token, in order.
    """
    return [extract_features(tokens, i) for i in range(len(tokens))]
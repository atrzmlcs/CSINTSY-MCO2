import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='\n%(message)s')

FIL_PREFIXES = {'nakikipag', 'makipag', 'pinagka', 'nagpa', 'pinag', 'napaka', 'pinaka', 'nagka', 'naka', 'maka', 'mapa', 'ipina', 'ipa', 'nag', 'mag', 'pag', 'pa', 'um', 'in', 'na', 'ma'}
FIL_SUFFIXES = {'an', 'han', 'hin', 'in'}

def is_english_root(root: str) -> bool:
    """Helper to detect English traits inside fused or hyphenated words."""
    if len(root) < 2: return False
    if any(c in 'cfjqvxz' for c in root): return True
    if any(bg in root for bg in ['th', 'sh', 'ph', 'gh', 'ck', 'ou', 'ea', 'tion', 'ee', 'oo', 'ie', 'au']): return True
    if any(c + c in root for c in "bcdfghjklmnpqrstvwxyz"): return True
    
    max_c = curr_c = 0
    for char in root:
        if char in "bcdfghjklmnpqrstvwxyz":
            curr_c += 1
            max_c = max(max_c, curr_c)
        else: curr_c = 0
    return max_c >= 3

def extract_features(tokens: List[str], i: int, verbose: bool = False) -> Dict[str, Any]:
    """Extracts structural, orthographic, and contextual features for ML evaluation."""
    word = tokens[i]
    w_lower = word.lower()
    length = len(word)
    n = len(tokens)
    
    # Context helpers
    def get_tok(off): return tokens[i + off].lower() if 0 <= (i + off) < n else '<PAD>'
    def get_raw(off): return tokens[i + off] if 0 <= (i + off) < n else '<PAD>'

    p2, p1, n1, n2 = get_tok(-2), get_tok(-1), get_tok(1), get_tok(2)
    raw_p1, raw_n1 = get_raw(-1), get_raw(1)

    # Character counters
    num_vowels = sum(1 for c in w_lower if c in "aeiou")
    num_cons = sum(1 for c in w_lower if c in "bcdfghjklmnpqrstvwxyz")
    
    max_c = curr_c = 1
    for j in range(1, length):
        if w_lower[j] == w_lower[j-1] and w_lower[j].isalpha():
            curr_c += 1
            max_c = max(max_c, curr_c)
        else: curr_c = 1

    # Code-Switching (CS) logic
    is_hyphenated_cs = is_fused_cs = False
    if '-' in w_lower:
        parts = w_lower.replace('–', '-').replace('—', '-').split('-')
        if len(parts) >= 2 and parts[0] in FIL_PREFIXES and is_english_root(parts[1]):
            is_hyphenated_cs = True
    else:
        for p in sorted(FIL_PREFIXES, key=len, reverse=True):
            if w_lower.startswith(p) and len(p) > 2 and is_english_root(w_lower[len(p):]):
                is_fused_cs = True
                break

    is_cs = is_hyphenated_cs or is_fused_cs
    has_eng_traits = any(c in w_lower for c in 'cfjqvxz') or any(bg in w_lower for bg in ['th', 'sh', 'ph', 'gh', 'ck', 'ou', 'ea', 'tion', 'ee', 'oo', 'ie', 'au'])

    # Raw Infix Check (The ML model will now organically override this if the word is English)
    has_fil_infix = False
    if length > 3:
        if w_lower[1:3] in ['um', 'in'] or w_lower[2:4] in ['um', 'in']:
            has_fil_infix = True

    feats = {
        # --- THE ML FIX: Direct Vocabulary Exposure ---
        # Instead of strict lists, we pass the words directly. DictVectorizer will map 
        # these to exact one-hot mathematical weights, letting the model learn naturally!
        'token_lower': w_lower,
        'prev_token': p1,
        'next_token': n1,

        # Group 1 & 2: Token Structure & Capitalization
        'length': length,
        'is_alpha': word.isalpha(),
        'is_capitalized': length > 0 and word[0].isupper(),
        'is_title_cased': length > 0 and word[0].isupper() and (length == 1 or word[1:].islower()),
        'is_all_caps': len([c for c in word if c.isalpha()]) > 1 and word.isupper(),
        'upper_ratio': round(sum(1 for c in word if c.isupper()) / length, 3) if length > 0 else 0.0,

        # Sub-word Slices & Expressive Noise
        'prefix_2': w_lower[:2] if length >= 2 else w_lower,
        'prefix_3': w_lower[:3] if length >= 3 else w_lower,
        'suffix_2': w_lower[-2:] if length >= 2 else w_lower,
        'suffix_3': w_lower[-3:] if length >= 3 else w_lower,
        'suffix_4': w_lower[-4:] if length >= 4 else w_lower,
        'is_laugh': any(l in w_lower for l in ['haha', 'hehe', 'hihi', 'huhu']),

        # Advanced Orthography & Morphology
        'num_vowels': num_vowels,
        'vowel_ratio': round(num_vowels / length, 3) if length > 0 else 0.0,
        'vowel_consonant_ratio': round(num_vowels / num_cons, 3) if num_cons > 0 else float(num_vowels),
        'has_3_plus_consecutive_chars': max_c >= 3,
        'has_eng_bigrams': any(bg in w_lower for bg in ['th', 'sh', 'ph', 'gh', 'ou', 'ea', 'ck', 'au']),
        'has_fil_bigrams': any(bg in w_lower for bg in ['ng', 'mga', 'ny', 'sy', 'ts', 'kw', 'pw', 'dy', 'ha', 'ah']),
        'has_fil_infix': has_fil_infix,
        'ends_with_s': length > 3 and w_lower.endswith('s') and has_eng_traits,
        'ends_with_es': length > 4 and w_lower.endswith('es') and has_eng_traits,

        # Code-Switching Features
        'is_hyphenated_cs': is_hyphenated_cs,
        'is_strong_fused_cs': is_fused_cs,
        'starts_with_mag_nag': (w_lower.startswith('mag') or w_lower.startswith('nag')) and not is_cs,

        # Group 5: Context Window Flags
        'is_first_in_sentence': i == 0,
        'is_last_in_sentence': i == n - 1,
        'prev_is_capitalized': raw_p1[0].isupper() if raw_p1 != '<PAD>' and len(raw_p1) > 0 else False,
        'next_is_capitalized': raw_n1[0].isupper() if raw_n1 != '<PAD>' and len(raw_n1) > 0 else False,
    }

    if verbose:
        logging.info(f"--- Token: '{word}' ---")
        for k, v in feats.items(): logging.info(f"  {k}: {v}")

    return feats

def featurize_tokens(tokens: List[str], verbose: bool = False) -> List[Dict[str, Any]]:
    """Convenience wrapper for a list of tokens."""
    return [extract_features(tokens, i, verbose=verbose) for i in range(len(tokens))]
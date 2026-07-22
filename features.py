import logging
from typing import List, Dict, Any

# Configure logging for much cleaner, readable vertical output
logging.basicConfig(level=logging.INFO, format='\n%(message)s')

# --- MORPHOLOGY CONSTANTS (Group 4) ---
FIL_PREFIXES = ['nakikipag', 'makipag', 'pinagka', 'nagpa', 'pinag', 'napaka', 'pinaka', 
                'nagka', 'naka', 'maka', 'mapa', 'ipina', 'ipa', 'nag', 'mag', 'pag', 'pa', 'um', 'in', 'na', 'ma']
FIL_SUFFIXES = ['an', 'han', 'hin', 'in']
ENG_SUFFIXES = ['tion', 'sion', 'ness', 'ment', 'able', 'ible', 'ing', 'ed', 'ly', 'er']

# A cheat sheet for high-frequency Filipino words and common Spanish loanwords 
# that don't have obvious Tagalog prefixes to help the model out.
FIL_COMMON_WORDS = {'beses', 'ilang', 'oras', 'pwede', 'puwede', 'swerte', 'talaga', 
                    'agad', 'sana', 'kaya', 'tapos', 'kahit', 'dahil', 'bakit', 'nako', 'ako', 'ko'
}
FIL_PARTICLES = ['ng', 'mga', 'sa', 'na', 'ka', 'pa', 'rin', 'din', 'ba', 'naman', 'lang', 'may', 'ay', 'daw', 'raw', 'po']

# Add 'er' to the end of this list
ENG_SUFFIXES = ['tion', 'sion', 'ness', 'ment', 'able', 'ible', 'ing', 'ed', 'ly', 'er']

# Make sure this is still at the top of your file
ENG_STOPWORDS = {
    'the', 'a', 'an', 'in', 'on', 'at', 'of', 'to', 'for', 'with', 
    'and', 'or', 'is', 'are', 'was', 'were', 'by', 'from', 'this', 'that', 'it', 'you', 'your'
}

def get_structure_features(word: str) -> dict:
    """Group 1: Basic structural and character-type features."""
    w_lower = word.lower()
    return {
        'length': len(word),
        'is_alpha': word.isalpha(),
        'is_punct_only': len(word) > 0 and not any(c.isalnum() for c in word),
        'has_digit': any(c.isdigit() for c in word),
        'has_hyphen': '-' in word,
        
        # The Protectors
        'is_strict_eng_stopword': w_lower in ENG_STOPWORDS,
        'is_strict_fil_particle': w_lower in FIL_PARTICLES,
        
        # THE NEW FIX: Check the Filipino cheat sheet
        'is_fil_common_word': w_lower in FIL_COMMON_WORDS
    }

def get_capitalization_features(word: str) -> dict:
    """Group 2: Capitalization patterns to detect proper nouns and abbreviations."""
    length = len(word)
    letters = [c for c in word if c.isalpha()]
    n_letters = len(letters)
    
    return {
        'is_title_cased': length > 0 and word[0].isupper() and (length == 1 or word[1:].islower()),
        'is_all_caps': n_letters > 1 and word.isupper(),
        'upper_ratio': sum(1 for c in word if c.isupper()) / length if length > 0 else 0.0
    }

def get_advanced_orthographic_features(word: str) -> dict:
    """Group 3: Advanced structural and orthographic checks based on project specs."""
    w_lower = word.lower()
    length = len(w_lower)

    # 1. Vowel and Consonant Ratios
    vowels = "aeiou"
    consonants = "bcdfghjklmnpqrstvwxyz"
    
    num_vowels = sum(1 for c in w_lower if c in vowels)
    num_consonants = sum(1 for c in w_lower if c in consonants)
    
    vowel_ratio = num_vowels / length if length > 0 else 0.0
    # Handle division by zero if a word is purely vowels (e.g., "a", "o")
    vowel_consonant_ratio = num_vowels / num_consonants if num_consonants > 0 else float(num_vowels)

    # 2. Consecutive Character Limits (Catching 'OTH' slang like "grrr", "yaaas")
    max_consecutive = 1
    current_consecutive = 1
    for i in range(1, length):
        if w_lower[i] == w_lower[i-1] and w_lower[i].isalpha():
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 1

    # 3. Discriminative Character N-grams
    # English often uses these specific vowel/consonant clusters
    has_eng_bigrams = any(bg in w_lower for bg in ['th', 'sh', 'ph', 'gh', 'ou', 'ea', 'ck'])
    
    # Filipino heavily relies on these consonant clusters and specific bigrams
    has_fil_bigrams = any(bg in w_lower for bg in ['ng', 'mga', 'ny', 'sy', 'ts', 'kw', 'pw', 'dy'])

    return {
        'num_vowels': num_vowels,
        'vowel_ratio': vowel_ratio,
        'vowel_consonant_ratio': vowel_consonant_ratio,
        'has_3_plus_consecutive_chars': max_consecutive >= 3,
        'has_eng_bigrams': has_eng_bigrams,
        'has_fil_bigrams': has_fil_bigrams
    }

def get_advanced_morphology_features(word: str) -> dict:
    """Group 4.5: Advanced morphological checks (Circumfixes, Partial Reduplication)."""
    w_lower = word.lower()
    length = len(w_lower)
    
    # Check if the word has strong English traits (to prevent "beses" from being tagged ENG)
    has_eng_traits = (
        any(c in w_lower for c in 'cfjqvxz') or 
        any(bg in w_lower for bg in ['th', 'sh', 'ph', 'gh', 'ck', 'ou', 'ea', 'tion', 'ee', 'oo', 'ie'])
    )
    
    # 1. Smart English Pluralization: Only trust 's' or 'es' if the word looks English
    ends_with_s = length > 3 and w_lower.endswith('s') and has_eng_traits
    ends_with_es = length > 4 and w_lower.endswith('es') and has_eng_traits

    # 2. Circumfixes / Kabilaan
    starts_fil = any(w_lower.startswith(p) for p in FIL_PREFIXES)
    ends_fil = any(w_lower.endswith(s) for s in FIL_SUFFIXES)
    has_circumfix = starts_fil and ends_fil and length > 5
    
    # 3. Unhyphenated Partial Reduplication 
    has_partial_redup = False
    if length >= 5:
        if w_lower[0:2] == w_lower[2:4] and w_lower[0:2].isalpha():
            has_partial_redup = True
            
    if not has_partial_redup and starts_fil and length >= 7:
        for p in FIL_PREFIXES:
            if w_lower.startswith(p):
                root_idx = len(p)
                if root_idx + 4 <= length:
                    syllable1 = w_lower[root_idx:root_idx+2]
                    syllable2 = w_lower[root_idx+2:root_idx+4]
                    if syllable1 == syllable2 and syllable1.isalpha():
                        has_partial_redup = True
                        break

    return {
        'ends_with_s': ends_with_s,
        'ends_with_es': ends_with_es,
        'has_circumfix': has_circumfix,
        'has_partial_redup': has_partial_redup
    }

def get_enhanced_context_features(tokens: List[str], i: int) -> dict:
    """Group 5: Contextual features looking at surrounding words (Window size = 2)."""
    n = len(tokens)
    
    # Helper function to safely fetch relative tokens
    def get_token(offset):
        idx = i + offset
        if 0 <= idx < n:
            return tokens[idx]
        return '<PAD>'

    prev_2 = get_token(-2)
    prev_1 = get_token(-1)
    next_1 = get_token(1)
    next_2 = get_token(2)

    vowels = "aeiou"

    return {
        'is_first_in_sentence': i == 0,
        'is_last_in_sentence': i == n - 1,
        
        # 1. Neighbor Identity Signals (Filipino Particles vs. English Stopwords)
        'prev_is_fil_particle': prev_1.lower() in FIL_PARTICLES,
        'next_is_fil_particle': next_1.lower() in FIL_PARTICLES,
        'prev_is_eng_stopword': prev_1.lower() in ENG_STOPWORDS,
        'next_is_eng_stopword': next_1.lower() in ENG_STOPWORDS,
        
        # 2. Phonetic Boundary Signals (e.g., does previous word end in a vowel?)
        'prev_ends_vowel': prev_1[-1].lower() in vowels if prev_1 != '<PAD>' and len(prev_1) > 0 else False,
        
        # 3. Wider Context Window (2 steps away)
        'prev_2_is_fil_particle': prev_2.lower() in FIL_PARTICLES,
        'next_2_is_fil_particle': next_2.lower() in FIL_PARTICLES,
        
        # 4. Name Propagation (If neighbor is Capitalized, current word might be part of a Name -> OTH)
        'prev_is_capitalized': prev_1[0].isupper() if prev_1 != '<PAD>' and len(prev_1) > 0 else False,
        'next_is_capitalized': next_1[0].isupper() if next_1 != '<PAD>' and len(next_1) > 0 else False,
    }

def get_strict_cs_features(word: str) -> dict:
    """Group 6: Strict Intra-word CS detection with a dedicated English Root Validator."""
    w_lower = word.lower()
    
    # We sort prefixes from longest to shortest (so 'pinag' is checked before 'pa')
    sorted_prefixes = sorted(FIL_PREFIXES, key=len, reverse=True)
    
    is_hyphenated_cs = False
    is_strong_fused_cs = False
    
    def is_english_root(root_word: str) -> bool:
        """Helper to check if a stripped root word looks strictly English."""
        if len(root_word) < 2:
            return False
            
        # 1. Foreign letters (Not typically in native Tagalog)
        if any(c in 'cfjqvxz' for c in root_word):
            return True
            
        # 2. Expanded English bigrams (Catches "eat", "good", "piece")
        eng_bigrams = ['th', 'sh', 'ph', 'gh', 'ck', 'ou', 'ea', 'tion', 'ee', 'oo', 'ie']
        if any(bg in root_word for bg in eng_bigrams):
            return True
            
        # 3. Double Consonants (Catches "dessert", "pull", "setting")
        double_consonants = [c+c for c in "bcdfghjklmnpqrstvwxyz"]
        if any(dc in root_word for dc in double_consonants):
            return True
            
        # 4. Consonant Clusters (3+ consonants in a row, e.g., "march")
        consonants = "bcdfghjklmnpqrstvwxyz"
        max_cluster, current_cluster = 0, 0
        for char in root_word:
            if char in consonants:
                current_cluster += 1
                max_cluster = max(max_cluster, current_cluster)
            else:
                current_cluster = 0
        if max_cluster >= 3:
            return True
            
        return False

    # 1. Catch Hyphenated CS (e.g., "nag-eat", but ignore "nag-aral" and "na-ko")
    if any(dash in w_lower for dash in ['-', '–', '—']):
        normalized_word = w_lower.replace('–', '-').replace('—', '-')
        parts = normalized_word.split('-')
        
        if len(parts) >= 2:
            prefix = parts[0]
            root = parts[1]
            # ONLY flag as CS if the prefix is Filipino AND the root is English
            if prefix in FIL_PREFIXES and is_english_root(root):
                is_hyphenated_cs = True

    # 2. Catch Fused CS (e.g., "magdessert", "nagmarch")
    elif not is_hyphenated_cs:
        for p in sorted_prefixes:
            if w_lower.startswith(p):
                root = w_lower[len(p):]
                
                # THE FIX: If there is no hyphen, 2-letter prefixes (in, pa, um) are too risky!
                # We skip them to protect words like "instead" and "part".
                if len(p) <= 2:
                    continue
                    
                if is_english_root(root):
                    is_strong_fused_cs = True
                    break 

    # --- THE MUTE BUTTON ---
    # If it is definitively Code-Switched, we silence the purely Filipino features 
    # so they don't accidentally outvote the CS classification in the Decision Tree.
    is_cs = is_hyphenated_cs or is_strong_fused_cs

    return {
        'is_hyphenated_cs': is_hyphenated_cs,
        'is_strong_fused_cs': is_strong_fused_cs,
        
        # Only vote for mag/nag if we are SURE it is not code-switched
        'starts_with_mag_nag': (w_lower.startswith('mag') or w_lower.startswith('nag')) and not is_cs
    }

def extract_features(tokens: List[str], i: int, verbose: bool = False) -> Dict[str, Any]:
    """Combines all word-level features (Groups 1-5) for a single token."""
    word = tokens[i]
    feats = {}
    feats.update(get_structure_features(word))
    feats.update(get_capitalization_features(word))
    feats.update(get_advanced_orthographic_features(word))
    feats.update(get_advanced_morphology_features(word))
    feats.update(get_enhanced_context_features(tokens, i))
    feats.update(get_strict_cs_features(word))
    
    if verbose:
        log_output = f"--- Parsed Token: '{word}' ---\n"
        for key, value in feats.items():
            log_output += f"  {key}: {value}\n"
        logging.info(log_output)
    
    return feats

def featurize_tokens(tokens: List[str], verbose: bool = False) -> List[Dict[str, Any]]:
    """Convenience wrapper for a list of tokens."""
    return [extract_features(tokens, i, verbose=verbose) for i in range(len(tokens))]

# if __name__ == "__main__":
#     # test_sentence = ["Atreuz", "kumain", "ka", "na", "ng", "lunch", "sa", "may", "EDSA", "nag-march", "22-anyos", "HAHAHA", "."]
#     test_sentence = ["nagmarch" , "nag-march", ]
#     features = featurize_tokens(test_sentence, verbose=True)
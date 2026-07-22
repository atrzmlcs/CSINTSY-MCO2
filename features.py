import logging
from typing import List, Dict, Any

# Configure logging for much cleaner, readable vertical output
logging.basicConfig(level=logging.INFO, format='\n%(message)s')

# --- MORPHOLOGY CONSTANTS (Group 4) ---
FIL_PREFIXES = ['nakikipag', 'makipag', 'pinagka', 'nagpa', 'pinag', 'napaka', 'pinaka', 
                'nagka', 'naka', 'maka', 'mapa', 'ipina', 'ipa', 'nag', 'mag', 'pag', 'pa', 'um', 'in']
ENG_SUFFIXES = ['tion', 'sion', 'ness', 'ment', 'able', 'ible', 'ing', 'ed', 'ly']

FIL_SUFFIXES = ['an', 'han', 'hin', 'in']
FIL_PARTICLES = ['ng', 'mga', 'sa', 'na', 'ka', 'pa', 'rin', 'din', 'ba', 'naman', 'lang', 'may', 'ay', 'daw', 'raw', 'po']

def get_structure_features(word: str) -> dict:
    """Group 1: Basic structural and character-type features."""
    return {
        'length': len(word),
        'is_alpha': word.isalpha(),
        'is_punct_only': len(word) > 0 and not any(c.isalnum() for c in word),
        'has_digit': any(c.isdigit() for c in word),
        'has_hyphen': '-' in word
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

def get_orthographic_features(word: str) -> dict:
    """Group 3: Structural checks (Consonant Clusters and Internet Slang)."""
    w_lower = word.lower()
    
    # English often has 3+ consecutive consonants. Filipino rarely does.
    consonants = "bcdfghjklmnpqrstvwxyz"
    max_cluster = 0
    current_cluster = 0
    for char in w_lower:
        if char in consonants:
            current_cluster += 1
            max_cluster = max(max_cluster, current_cluster)
        else:
            current_cluster = 0
            
    # Catch standard internet laughs
    is_laugh = any(laugh in w_lower for laugh in ['haha', 'hehe', 'hihi', 'huhu'])
            
    return {
        'has_consonant_cluster': max_cluster >= 3,
        'is_laugh': is_laugh
    }

def get_morphology_features(word: str) -> dict:
    """Group 4: Affixes, Infixes, Particles, and reduplication patterns."""
    w_lower = word.lower()
    
    # Reduplication
    is_reduplicated = False
    if '-' in w_lower:
        parts = w_lower.split('-')
        if len(parts) == 2 and parts[0] == parts[1] and len(parts[0]) > 1:
            is_reduplicated = True

    # Prefix checks
    starts_fil = any(w_lower.startswith(p) for p in FIL_PREFIXES)
    
    has_fil_prefix_before_hyphen = False
    if '-' in w_lower:
        prefix = w_lower.split('-')[0]
        if prefix in FIL_PREFIXES:
            has_fil_prefix_before_hyphen = True
            
    # NEW: Catch unhyphenated Code-Switching (e.g., "nagmarch", "nagshift")
    is_fused_cs = False
    if starts_fil and not '-' in w_lower:
        for p in FIL_PREFIXES:
            if w_lower.startswith(p):
                root = w_lower[len(p):]
                # If the root word has English-only letters OR a 3-letter consonant cluster
                if any(c in 'cfjqvxz' for c in root):
                    is_fused_cs = True
                    break
                
                consonants = "bcdfghjklmnpqrstvwxyz"
                max_cluster = 0
                current_cluster = 0
                for char in root:
                    if char in consonants:
                        current_cluster += 1
                        max_cluster = max(max_cluster, current_cluster)
                    else:
                        current_cluster = 0
                if max_cluster >= 3:
                    is_fused_cs = True
                    break

    # Infixes (um, in)
    has_fil_infix = False
    if len(w_lower) > 3:
        if w_lower[1:3] in ['um', 'in'] or w_lower[2:4] in ['um', 'in']:
            has_fil_infix = True

    return {
        'is_reduplicated': is_reduplicated,
        'starts_fil_prefix': starts_fil,
        'ends_eng_suffix': any(w_lower.endswith(s) for s in ENG_SUFFIXES),
        'ends_fil_suffix': any(w_lower.endswith(s) for s in FIL_SUFFIXES),
        'has_fil_prefix_before_hyphen': has_fil_prefix_before_hyphen,
        'is_fused_cs': is_fused_cs,  # Our new smoking gun for unhyphenated CS!
        'has_fil_infix': has_fil_infix,
        'is_fil_particle': w_lower in FIL_PARTICLES
    }

def get_context_features(tokens: List[str], i: int) -> dict:
    """Group 5: Contextual features derived from neighboring tokens."""
    n = len(tokens)
    
    prev_word = tokens[i - 1].lower() if i > 0 else '<START>'
    next_word = tokens[i + 1].lower() if i < n - 1 else '<END>'
    
    feats = {
        'is_first_in_sentence': i == 0,
        'is_last_in_sentence': i == n - 1,
    }
    
    # Check if neighbors are common Filipino particles
    for label, neighbor in [('prev', prev_word), ('next', next_word)]:
        if neighbor in ('<START>', '<END>', ''):
            feats[f'{label}_len'] = 0
            feats[f'{label}_is_particle'] = False
        else:
            feats[f'{label}_len'] = len(neighbor)
            feats[f'{label}_is_particle'] = neighbor in FIL_PARTICLES
            
    return feats

def extract_features(tokens: List[str], i: int, verbose: bool = False) -> Dict[str, Any]:
    """Combines all word-level features (Groups 1-5) for a single token."""
    word = tokens[i]
    feats = {}
    feats.update(get_structure_features(word))
    feats.update(get_capitalization_features(word))
    feats.update(get_orthographic_features(word))
    feats.update(get_morphology_features(word))
    feats.update(get_context_features(tokens, i))
    
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
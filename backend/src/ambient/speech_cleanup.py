"""
Speech Cleanup Filter — Phase 1
Cleans raw STT output before ingestion:
  1. Filler word removal (um, uh, like, you know, ...)
  2. Disfluency removal (repeated words, false starts)
  3. Confidence gating (drop segments < 0.3 confidence)
  4. STT error normalization (gonna → going to, etc.)
  5. Minimum content gate (drop turns that are pure filler)

Inserted between transcription and conversation.add_turn() in _process_speech().
"""

import re
from typing import Optional, Dict, List


# ── Filler words and phrases ─────────────────────────────────────────────

# Single-word fillers (removed when standalone or surrounded by other fillers)
FILLER_WORDS = {
    "um", "uh", "uhh", "umm", "hmm", "hm", "mm", "mmm",
    "er", "erm", "ah", "ahh", "eh",
}

# Filler phrases — removed as whole units
FILLER_PHRASES = [
    "you know", "i mean", "kind of", "sort of",
    "you know what i mean", "if you will", "so to speak",
    "at the end of the day", "to be honest", "to be fair",
    "like i said",
]

# Words that are fillers when used as discourse markers (not as actual content)
# Only removed when at sentence start/end or between pauses
DISCOURSE_MARKERS = {"like", "so", "well", "right", "basically", "actually", "literally"}

# ── STT normalization map ────────────────────────────────────────────────

STT_NORMALIZATIONS = {
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "kinda": "kind of",
    "sorta": "sort of",
    "dunno": "don't know",
    "lemme": "let me",
    "gimme": "give me",
    "coulda": "could have",
    "shoulda": "should have",
    "woulda": "would have",
    "oughta": "ought to",
    "hafta": "have to",
    "tryna": "trying to",
    "outta": "out of",
}


def clean_transcript(text: str, confidence: float = 1.0,
                     word_confidences: Optional[List[Dict]] = None) -> Optional[str]:
    """
    Clean a raw transcript turn.

    Args:
        text: Raw transcript text from STT
        confidence: Overall segment confidence (avg_log_prob from whisper)
        word_confidences: Optional list of {"word": str, "probability": float}
                         from whisper word-level timestamps

    Returns:
        Cleaned text string, or None if the turn should be dropped entirely.
    """
    if not text or not text.strip():
        return None

    # 1. Confidence gate — drop very low confidence segments
    #    avg_log_prob < -1.5 (~0.22 probability) = likely noise/unintelligible
    if confidence < -1.5:
        return None

    cleaned = text.strip()

    # 2. Normalize case for processing (preserve original for output)
    lower = cleaned.lower()

    # 3. Remove filler phrases (multi-word, order matters — longest first)
    for phrase in sorted(FILLER_PHRASES, key=len, reverse=True):
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(phrase) + r'\b'
        lower = re.sub(pattern, ' ', lower, flags=re.IGNORECASE)
        cleaned = re.sub(pattern, ' ', cleaned, flags=re.IGNORECASE)

    # 4. Remove standalone filler words
    words = cleaned.split()
    filtered_words = []
    for i, word in enumerate(words):
        word_lower = word.lower().strip(".,!?;:")
        if word_lower in FILLER_WORDS:
            continue
        # Remove discourse markers only at start/end of sentence
        if word_lower in DISCOURSE_MARKERS:
            if i == 0 or i == len(words) - 1:
                continue
            # Also remove if surrounded by punctuation/pause
            if i > 0 and words[i - 1].endswith((",", ".", "—", "-")):
                continue
        filtered_words.append(word)
    cleaned = " ".join(filtered_words)

    # 5. Remove disfluencies — repeated words ("I I I think" → "I think")
    cleaned = re.sub(r'\b(\w+)(?:\s+\1){1,}\b', r'\1', cleaned, flags=re.IGNORECASE)

    # 6. Remove false starts with dashes ("I was going — I went to" → "I went to")
    cleaned = re.sub(r'[^.!?]*\s*[—–-]\s*', '', cleaned, count=1)

    # 7. STT error normalization
    for informal, formal in STT_NORMALIZATIONS.items():
        pattern = r'\b' + re.escape(informal) + r'\b'
        cleaned = re.sub(pattern, formal, cleaned, flags=re.IGNORECASE)

    # 8. Clean up whitespace and punctuation
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    cleaned = re.sub(r'\s+([.,!?;:])', r'\1', cleaned)  # Fix "word ." → "word."

    # 9. Drop word-level low-confidence tokens if available
    if word_confidences:
        cleaned = _filter_low_confidence_words(cleaned, word_confidences)

    # 10. Minimum content gate — if nothing meaningful remains, drop
    if not cleaned or len(cleaned) < 3:
        return None

    # Check if what remains is only stop words / non-content
    remaining_words = [w for w in cleaned.split()
                       if w.lower().strip(".,!?;:") not in
                       FILLER_WORDS | {"yes", "yeah", "yep", "no", "nah",
                                       "okay", "ok", "bye", "hi", "hey",
                                       "the", "a", "an", "is", "it", "and"}]
    if len(remaining_words) == 0:
        return None

    return cleaned


def _filter_low_confidence_words(text: str,
                                  word_confidences: List[Dict],
                                  threshold: float = 0.3) -> str:
    """
    Remove words that have word-level confidence below threshold.
    Only removes if the word is clearly noise (very low confidence).
    """
    if not word_confidences:
        return text

    low_conf_words = set()
    for wc in word_confidences:
        prob = wc.get("probability", 1.0)
        word = wc.get("word", "").strip()
        if prob < threshold and word:
            low_conf_words.add(word.lower().strip())

    if not low_conf_words:
        return text

    words = text.split()
    filtered = [w for w in words if w.lower().strip(".,!?;:") not in low_conf_words]
    return " ".join(filtered).strip()


def get_word_confidences_from_segments(segments: List[Dict]) -> List[Dict]:
    """
    Extract word-level confidences from faster-whisper segment data.
    Only available when quality_mode=True (word_timestamps=True).
    """
    words = []
    for seg in segments:
        for w in seg.get("words", []):
            words.append({
                "word": w.get("word", ""),
                "probability": w.get("probability", 1.0),
            })
    return words

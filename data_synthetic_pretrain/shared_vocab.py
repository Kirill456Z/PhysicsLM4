"""Shared vocabulary layout and word generation for all synthetic graph tasks.

Unified token layout (B = base_vocab_size):
    0          PAD / unused
    1 .. B     node base tokens  (non-final position in a word)
    B+1 .. 2B  node EOW markers  (final position — value = base_token + B)
    2B+1       TASK_DEPO         first token of every Depo sequence
    2B+2       TASK_BREVO        first token of every Brevo sequence
    2B+3       TASK_CONCOMP      first token of every ConComp sequence
    2B+4       EOS               end-of-sequence (all tasks)
    2B+5       BREVO_QUERY       separates edges from query in Brevo
    2B+6       BREVO_ANS         separates query from answer in Brevo
    2B+7       CONCOMP_QUERY     separates edges from query in ConComp
    2B+8       CONCOMP_ANS       separates query from answer in ConComp
    2B+9       DEPO_SEP          separates query node from answer in Depo
    2B+10      DEPO_EDGE_SEP     optional edge separator in Depo (rarely used)
    2B+11      TASK_SP           first token of every ShortestPath sequence
    2B+12      SP_QUERY          separates edges from (start, end) query in SP
    2B+13      SP_ANS            separates query from path answer in SP
    2B+14      TASK_CONCOMP_FACTOR   first token of every ConCompFactor sequence
    2B+15      CONCOMP_FACTOR_ANS    separates edges from factorization answer
    2B+16      ADJ_NODE_TO_NEIGHBORS separator between node and neighbor list
    2B+17      ADJ_PAIR_SEP          separator between adjacency entries
    2B+18      ADJ_NO_NEIGHBOR       explicit marker for empty neighbor list
    200+k      Depo hop-k query token  (k = 1 .. max_hops)

Recommended vocab_size for training: 512 (covers all tokens with ample headroom).

Word format
-----------
Every node name is a sequence of ``min_len .. max_len`` integer tokens:
  - Non-final tokens are drawn from  [1, B]  (base tokens).
  - The final token is drawn from     [B+1, 2B]  (EOW marker = base + B).

This is identical across Depo, Brevo, and ConComp when they share the same
``base_vocab_size``, ``min_token_length``, and ``max_token_length``.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Vocabulary layout
# ---------------------------------------------------------------------------

def vocab_layout(base_vocab_size: int) -> dict[str, int]:
    """Return the full token-ID mapping for the given base vocabulary size.

    Args:
        base_vocab_size: B — size of the base token alphabet.

    Returns:
        Dict mapping symbolic name → token ID.
    """
    B = base_vocab_size
    return {
        "PAD":            0,
        "base_start":     1,
        "base_end":       B,
        "eow_start":      B + 1,
        "eow_end":        2 * B,
        "TASK_DEPO":      2 * B + 1,
        "TASK_BREVO":     2 * B + 2,
        "TASK_CONCOMP":   2 * B + 3,
        "EOS":            2 * B + 4,
        "BREVO_QUERY":    2 * B + 5,
        "BREVO_ANS":      2 * B + 6,
        "CONCOMP_QUERY":  2 * B + 7,
        "CONCOMP_ANS":    2 * B + 8,
        "DEPO_SEP":       2 * B + 9,
        "DEPO_EDGE_SEP":  2 * B + 10,
        "TASK_SP":        2 * B + 11,
        "SP_QUERY":       2 * B + 12,
        "SP_ANS":         2 * B + 13,
        "TASK_CONCOMP_FACTOR": 2 * B + 14,
        "CONCOMP_FACTOR_ANS": 2 * B + 15,
        "ADJ_NODE_TO_NEIGHBORS": 2 * B + 16,
        "ADJ_PAIR_SEP": 2 * B + 17,
        "ADJ_NO_NEIGHBOR": 2 * B + 18,
        "DEPO_QUERY_BASE": 200,        # hop-k token = DEPO_QUERY_BASE + k
    }


def unified_vocab_size(base_vocab_size: int, max_hops: int = 16) -> int:
    """Minimum number of distinct token IDs needed for given parameters.

    The highest-numbered token is the Depo hop-k token at ``200 + max_hops``.
    Returns that value + 1.
    """
    return 200 + max_hops + 1


def is_eow(token: int, base_vocab_size: int) -> bool:
    """Return True if *token* is an end-of-word marker."""
    B = base_vocab_size
    return B + 1 <= token <= 2 * B


# ---------------------------------------------------------------------------
# Word generation — stdlib random.Random (Depo, Brevo)
# ---------------------------------------------------------------------------

def generate_words_stdlib(
    rng,
    n: int,
    base_vocab_size: int,
    min_len: int,
    max_len: int,
) -> list[list[int]]:
    """Generate *n* unique multi-token words using a ``random.Random`` instance.

    Non-final tokens: ``[1, base_vocab_size]``.
    Final token (EOW): ``[base_vocab_size+1, 2*base_vocab_size]``.

    Words are returned in a stable sorted order so the result is deterministic
    across Python runs regardless of set-iteration order.
    """
    B = base_vocab_size

    def _sample() -> tuple:
        length = rng.randint(min_len, max_len)
        toks = [rng.randint(1, B) for _ in range(length)]
        toks[-1] += B
        return tuple(toks)

    seen: set = set()
    while len(seen) < n:
        seen.add(_sample())
    return [list(w) for w in sorted(seen)]


# ---------------------------------------------------------------------------
# Word generation — numpy Generator (ConComp)
# ---------------------------------------------------------------------------

def generate_words_numpy(
    rng,
    n: int,
    base_vocab_size: int,
    min_len: int,
    max_len: int,
) -> list[list[int]]:
    """Generate *n* unique multi-token words using a numpy ``Generator`` instance.

    Same word format as :func:`generate_words_stdlib`.
    """
    B = base_vocab_size

    def _sample() -> tuple:
        length = int(rng.integers(min_len, max_len + 1))
        toks = rng.integers(1, B + 1, size=length).tolist()
        toks[-1] += B
        return tuple(toks)

    seen: set = set()
    while len(seen) < n:
        seen.add(_sample())
    return [list(w) for w in sorted(seen)]


# ---------------------------------------------------------------------------
# Word parsing helpers (for evaluation / tests)
# ---------------------------------------------------------------------------

def split_into_words(
    tokens: list[int],
    base_vocab_size: int,
) -> list[list[int]]:
    """Split a flat token stream into words, using EOW markers as boundaries.

    A word ends at the first EOW marker (token in ``[B+1, 2B]``).
    Tokens outside valid word positions (e.g. special tokens) are NOT yielded.
    Call this only on the content portion of a sequence (no task/special tokens).
    """
    B = base_vocab_size
    words: list[list[int]] = []
    current: list[int] = []
    for tok in tokens:
        if 1 <= tok <= 2 * B:
            current.append(tok)
            if B + 1 <= tok <= 2 * B:   # EOW marker
                words.append(current)
                current = []
    if current:
        words.append(current)   # unterminated last word (shouldn't happen normally)
    return words

from typing import List


def _levenshtein_distance(ref: str, hyp: str) -> int:
    if ref == hyp:
        return 0
    if len(ref) == 0:
        return len(hyp)
    if len(hyp) == 0:
        return len(ref)

    prev = list(range(len(hyp) + 1))
    curr = [0] * (len(hyp) + 1)

    for i in range(1, len(ref) + 1):
        curr[0] = i
        rc = ref[i - 1]
        for j in range(1, len(hyp) + 1):
            hc = hyp[j - 1]
            cost = 0 if rc == hc else 1
            curr[j] = min(
                prev[j] + 1,       # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + cost # substitution
            )
        prev, curr = curr, prev

    return prev[-1]


def character_error_rate(refs: List[str], hyps: List[str]) -> float:
    assert len(refs) == len(hyps)
    total_chars = 0
    total_edit = 0
    for r, h in zip(refs, hyps):
        total_chars += max(1, len(r))
        total_edit += _levenshtein_distance(r, h)
    return total_edit / total_chars if total_chars > 0 else 0.0




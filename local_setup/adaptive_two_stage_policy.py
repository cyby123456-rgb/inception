"""CPU-only acceptance controller; history is local to one generated response."""


def recent_budget(history, max_drafts, explore_drafts, good, strong):
    """Use accepted/proposed over recent *attempted* cycles, before this cycle.

    Skips are not failures. A nonzero exploration budget permits recovery after
    cooldown; no future tokens or labels enter this decision.
    """
    proposed = sum(n for _, n in history)
    rate = sum(a for a, _ in history) / proposed if proposed else None
    if rate is None or rate < good:
        return min(max_drafts, explore_drafts), rate
    if rate < strong:
        return min(max_drafts, 2), rate
    return max_drafts, rate

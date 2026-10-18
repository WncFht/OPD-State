"""Place complete native source groups without changing model/trajectory routes."""


def place_groups(counts, replicas, layout):
    """Return physical replica indices, preserving each source's request quota.

    Split placement gives every active source a device before assigning the
    remaining devices by request demand. It changes routing only: engine KV
    budgets and actor publication collectives keep their configured sizes.
    """
    if layout not in {"overlap", "split"}:
        raise ValueError("Native generation layout must be overlap or split")
    if type(replicas) is not int or replicas < 1 or any(type(n) is not int or n < 1 for n in counts.values()):
        raise ValueError("Native placement needs positive replica and request counts")
    if not counts:
        return {}
    names = sorted(counts)
    if layout == "overlap" or len(names) == 1:
        return {name: tuple(range(min(replicas, counts[name]))) for name in names}
    if len(names) > replicas:
        raise ValueError("Split layout needs at least one physical replica per active source")
    used = min(replicas, sum(counts.values()))
    desired = {name: used * counts[name] / sum(counts.values()) for name in names}
    quota = dict.fromkeys(names, 1)
    for _ in range(used - len(names)):
        eligible = [name for name in names if quota[name] < counts[name]]
        winner = max(eligible, key=lambda name: desired[name] - quota[name])
        quota[winner] += 1
    result, offset = {}, 0
    for name in names:
        result[name] = tuple(range(offset, offset + quota[name]))
        offset += quota[name]
    return result

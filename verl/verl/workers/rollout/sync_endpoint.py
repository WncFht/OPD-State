"""Shared sender/receiver address, including a run ID across Ray clusters."""

import re


def weight_sync_endpoint(job_id, replica_rank, local_rank, namespace=""):
    if namespace and re.fullmatch(r"[a-zA-Z0-9_-]{1,40}", namespace) is None:
        raise ValueError("Weight sync namespace must contain at most 40 safe ASCII characters")
    token = f"{job_id}-{namespace}" if namespace else str(job_id)
    endpoint = f"ipc:///tmp/rl-colocate-zmq-{token}-replica-{replica_rank}-rank-{local_rank}.sock"
    if len(endpoint.removeprefix("ipc://").encode()) >= 108:
        raise ValueError("Weight sync address exceeds the Unix socket path limit")
    return endpoint

"""Join CUDA and NVML identity namespaces through unique local PCI addresses.

CUDA UUIDs can differ from nvidia-smi UUIDs on this host. Inventory queries run
in a fresh subprocess, enumerate metadata only, and never retain a context.
GPU selection uses PCI-verified CUDA ordinals for the requested NVML cards.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path


def normalize_uuid(value):
    return str(value).lower().removeprefix("gpu-")


def normalize_pci(value):
    domain, bus, slot = value.strip().split(":")
    device, function = slot.split(".")
    return f"{int(domain, 16):04x}:{int(bus, 16):02x}:{int(device, 16):02x}.{int(function, 16):x}"


def cuda_inventory():
    """Read driver metadata without cuCtxCreate/cuDevicePrimaryCtxRetain."""
    library = ctypes.CDLL("libcuda.so.1")

    def check(code):
        if code:
            raise RuntimeError(f"CUDA device identity metadata query failed: {code}")

    check(library.cuInit(0))
    count = ctypes.c_int()
    check(library.cuDeviceGetCount(ctypes.byref(count)))

    class Uuid(ctypes.Structure):
        _fields_ = [("bytes", ctypes.c_ubyte * 16)]

    devices = []
    for ordinal in range(count.value):
        device = ctypes.c_int()
        check(library.cuDeviceGet(ctypes.byref(device), ordinal))
        value = Uuid()
        check(library.cuDeviceGetUuid(ctypes.byref(value), device))
        legacy = "GPU-" + str(uuid.UUID(bytes=bytes(value.bytes)))
        modern = legacy
        if hasattr(library, "cuDeviceGetUuid_v2"):
            check(library.cuDeviceGetUuid_v2(ctypes.byref(value), device))
            modern = "GPU-" + str(uuid.UUID(bytes=bytes(value.bytes)))
        bus = ctypes.create_string_buffer(64)
        check(library.cuDeviceGetPCIBusId(bus, len(bus), device))
        flags, active = ctypes.c_uint(), ctypes.c_int()
        check(library.cuDevicePrimaryCtxGetState(device, ctypes.byref(flags), ctypes.byref(active)))
        if active.value:
            raise RuntimeError("CUDA metadata subprocess unexpectedly retained a primary context")
        devices.append(
            {
                "ordinal": ordinal,
                "pci_bus_id": normalize_pci(bus.value.decode()),
                "cuda_uuid": legacy,
                "cuda_uuid_v2": modern,
            }
        )
    context = ctypes.c_void_p()
    check(library.cuCtxGetCurrent(ctypes.byref(context)))
    if context.value is not None:
        raise RuntimeError("CUDA metadata subprocess unexpectedly created a context")
    torch_count = None
    try:
        import torch
    except ImportError:
        pass
    else:
        torch_count = torch.cuda.device_count()
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA device-count metadata query initialized Torch CUDA")
    check(library.cuCtxGetCurrent(ctypes.byref(context)))
    if context.value is not None:
        raise RuntimeError("Device-count inspection created a CUDA context")
    return {
        "scope": "CUDA metadata only; no retained/current context",
        "current_context": None,
        "torch_device_count": torch_count,
        "devices": devices,
    }


def join_identities(nvml_rows, cuda_rows):
    by_pci = {}
    wanted = {normalize_pci(row["pci_bus_id"]) for row in nvml_rows}
    for row in cuda_rows:
        pci = normalize_pci(row["pci_bus_id"])
        if pci not in wanted:
            continue
        if pci in by_pci:
            raise ValueError("CUDA inventory repeats a PCI address; this probe requires whole GPUs without MIG")
        by_pci[pci] = row
    devices, seen_pci = {}, set()
    for row in nvml_rows:
        index, pci = int(row["index"]), normalize_pci(row["pci_bus_id"])
        if index in devices or pci in seen_pci or pci not in by_pci:
            raise ValueError("NVML inventory has a duplicate or no matching CUDA PCI address")
        seen_pci.add(pci)
        cuda = by_pci[pci]
        devices[index] = {
            "physical_index": index,
            "pci_bus_id": pci,
            "nvml_uuid": row["uuid"],
            "cuda_uuid": cuda["cuda_uuid"],
            "cuda_uuid_v2": cuda["cuda_uuid_v2"],
            "cuda_ordinal": cuda["ordinal"],
        }
    identity_aliases(devices)  # Reject one alias assigned to two physical GPUs.
    return devices


def identity_aliases(devices):
    result = {}
    for row in devices.values():
        if any(
            not isinstance(row[key], str) or not row[key].strip() for key in ("nvml_uuid", "cuda_uuid", "cuda_uuid_v2")
        ):
            raise ValueError("GPU identities must contain nonempty UUID strings")
        canonical = normalize_uuid(row["nvml_uuid"])
        for key in ("nvml_uuid", "cuda_uuid", "cuda_uuid_v2"):
            alias = normalize_uuid(row[key])
            if not alias or alias in result and result[alias] != canonical:
                raise ValueError("GPU identity alias is empty or points to multiple physical devices")
            result[alias] = canonical
    return result


def matches_device(observed, expected):
    if isinstance(expected, str):
        return normalize_uuid(observed) == normalize_uuid(expected)
    return normalize_uuid(observed) in identity_aliases({0: expected})


def query_devices(indices=(0, 1, 2, 3)):
    if (
        not indices
        or len(set(indices)) != len(indices)
        or any(type(i) is not int or i not in range(4) for i in indices)
    ):
        raise ValueError("Native experiments may select only distinct physical GPUs 0–3")
    text = subprocess.check_output(
        [
            "nvidia-smi",
            "--id=" + ",".join(map(str, indices)),
            "--query-gpu=index,uuid,pci.bus_id",
            "--format=csv,noheader",
        ],
        text=True,
    )
    rows = [
        {"index": int(fields[0]), "uuid": fields[1].strip(), "pci_bus_id": fields[2].strip()}
        for fields in (line.split(",") for line in text.splitlines())
    ]
    env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    env.pop("CUDA_VISIBLE_DEVICES", None)
    raw = subprocess.check_output(
        [sys.executable, str(Path(__file__).resolve()), "--cuda-inventory"], env=env, text=True
    )
    inventory = json.loads(raw)
    devices = join_identities(rows, inventory["devices"])
    if set(devices) != set(indices):
        raise ValueError("Physical GPU inventory differs from the requested devices")
    validate_visibility(devices, ",".join(str(devices[index]["cuda_ordinal"]) for index in indices))
    return devices


def validate_visibility(devices, visible):
    env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": visible}
    raw = subprocess.check_output(
        [sys.executable, str(Path(__file__).resolve()), "--cuda-inventory"], env=env, text=True
    )
    inventory = json.loads(raw)
    selected = inventory["devices"]
    if inventory.get("torch_device_count") not in (None, len(devices)):
        raise ValueError("Torch device visibility differs from the selected CUDA device inventory")
    if len(selected) != len(devices) or {row["pci_bus_id"] for row in selected} != {
        row["pci_bus_id"] for row in devices.values()
    }:
        raise ValueError("CUDA visibility mask does not select exactly the allowed physical PCI devices")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-inventory", action="store_true")
    args = parser.parse_args()
    print(json.dumps(cuda_inventory() if args.cuda_inventory else query_devices()))


if __name__ == "__main__":
    main()

"""Join independent identity namespaces without relaxing physical GPU limits."""

import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
import native_devices as DEVICES


def fixture():
    nvml = [
        {
            "index": i,
            "uuid": f"GPU-nvml-{i}",
            "pci_bus_id": f"00000000:{0x63 + i:02X}:00.0",
        }
        for i in range(2)
    ]
    cuda = [
        {
            "ordinal": i,
            "cuda_uuid": f"GPU-cuda-{i}",
            "cuda_uuid_v2": f"GPU-cuda-v2-{i}",
            "pci_bus_id": f"0000:{0x63 + i:02x}:00.0",
        }
        for i in range(2)
    ]
    return nvml, cuda


class NativeDeviceTests(unittest.TestCase):
    def test_pci_bridge_preserves_distinct_uuid_namespaces_and_rejects_wrong_devices(
        self,
    ):
        nvml, cuda = fixture()
        mapping = DEVICES.join_identities(nvml, list(reversed(cuda)))
        self.assertEqual(mapping[0]["pci_bus_id"], "0000:63:00.0")
        self.assertTrue(DEVICES.matches_device("GPU-cuda-v2-0", mapping[0]))
        self.assertTrue(DEVICES.matches_device("GPU-nvml-0", mapping[0]))
        self.assertFalse(DEVICES.matches_device("GPU-cuda-1", mapping[0]))
        self.assertTrue(DEVICES.matches_device("GPU-legacy", "legacy"))

    def test_missing_duplicate_or_ambiguous_selected_devices_fail(self):
        for kind in (
            "missing",
            "duplicate_cuda",
            "duplicate_nvml",
            "shared_alias",
            "empty_alias",
        ):
            with self.subTest(kind=kind):
                nvml, cuda = fixture()
                if kind == "missing":
                    cuda.pop()
                elif kind == "duplicate_cuda":
                    cuda.append(copy.deepcopy(cuda[0]))
                elif kind == "duplicate_nvml":
                    nvml.append(copy.deepcopy(nvml[0]))
                elif kind == "shared_alias":
                    cuda[1]["cuda_uuid"] = cuda[0]["cuda_uuid"]
                else:
                    cuda[0]["cuda_uuid"] = None
                with self.assertRaises(ValueError):
                    DEVICES.join_identities(nvml, cuda)

    def test_mig_duplicates_on_unselected_cards_do_not_change_allowed_inventory(self):
        nvml, cuda = fixture()
        cuda.extend([{"pci_bus_id": "0000:a3:00.0"}, {"pci_bus_id": "0000:a3:00.0"}])
        self.assertEqual(len(DEVICES.join_identities(nvml, cuda)), 2)

    def test_indices_outside_first_four_fail_before_any_hardware_query(self):
        with patch.object(DEVICES.subprocess, "check_output") as query:
            for indices in ((), (0, 0), (4,), (False,)):
                with self.subTest(indices=indices), self.assertRaises(ValueError):
                    DEVICES.query_devices(indices)
            query.assert_not_called()

    def test_numeric_visibility_roundtrip_is_verified_before_loading_models(self):
        nvml, cuda = fixture()
        csv = "\n".join(f"{r['index']}, {r['uuid']}, {r['pci_bus_id']}" for r in nvml)
        inventory = json.dumps({"devices": cuda, "current_context": None})
        with patch.object(
            DEVICES.subprocess, "check_output", side_effect=[csv, inventory, inventory]
        ) as query:
            mapping = DEVICES.query_devices((0, 1))
            self.assertEqual(set(mapping), {0, 1})
            visible = query.call_args_list[-1].kwargs["env"]["CUDA_VISIBLE_DEVICES"]
            self.assertEqual(visible, "0,1")
        wrong = json.dumps({"devices": cuda[:1]})
        with (
            patch.object(
                DEVICES.subprocess, "check_output", side_effect=[csv, inventory, wrong]
            ),
            self.assertRaisesRegex(ValueError, "visibility mask"),
        ):
            DEVICES.query_devices((0, 1))

    def test_torch_and_driver_visibility_mismatch_is_not_accepted(self):
        nvml, cuda = fixture()
        csv = "\n".join(f"{r['index']}, {r['uuid']}, {r['pci_bus_id']}" for r in nvml)
        inventory = json.dumps({"devices": cuda})
        mismatch = json.dumps({"devices": cuda, "torch_device_count": 0})
        with (
            patch.object(
                DEVICES.subprocess,
                "check_output",
                side_effect=[csv, inventory, mismatch],
            ),
            self.assertRaisesRegex(ValueError, "Torch device visibility"),
        ):
            DEVICES.query_devices((0, 1))


if __name__ == "__main__":
    unittest.main()

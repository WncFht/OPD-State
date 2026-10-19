"""Check real launcher device forwarding with a CPU-only downstream capture."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class MathDeviceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.capture = self.root / "capture.json"
        binary = self.root / "bin"
        binary.mkdir()
        fake = binary / "python3"
        fake.write_text(
            f"#!{sys.executable}\nimport os,json,sys\nfrom pathlib import Path\n"
            "Path(os.environ['CAPTURE_FILE']).write_text(json.dumps({"
            "'args':sys.argv[1:],'devices':os.environ['CUDA_VISIBLE_DEVICES'],"
            "'device_order':os.environ['CUDA_DEVICE_ORDER']}))\n"
        )
        fake.chmod(0o755)
        self.env = {
            **os.environ,
            "OPD_TRAIN_PYTHON": str(fake),
            "OPD_DRY_RUN": "1",
            "OPD_FREEZE_SOURCE": "0",
            "OPD_ROOT": str(self.root / "run"),
            "WANDB_DIR": str(self.root / "wandb"),
            "RAY_TMPDIR": str(self.root / "ray"),
            "CAPTURE_FILE": str(self.capture),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        }

    def run_launcher(self, **settings):
        self.capture.unlink(missing_ok=True)
        process = subprocess.run(
            ["bash", str(ROOT / "recipe/math_baseline.sh")],
            cwd=ROOT,
            env={**self.env, **settings},
            capture_output=True,
            text=True,
        )
        result = json.loads(self.capture.read_text()) if self.capture.exists() else None
        return process, result

    def test_four_and_eight_device_profiles_are_forwarded_without_cuda_initialization(
        self,
    ):
        for count in (4, 8):
            devices = ",".join(map(str, range(count)))
            process, result = self.run_launcher(
                N_GPUS_PER_NODE=str(count), CUDA_VISIBLE_DEVICES=devices
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(result["devices"], devices)
            self.assertEqual(result["device_order"], "PCI_BUS_ID")
            self.assertIn(f"trainer.n_gpus_per_node={count}", result["args"])
            self.assertIn("data.train_batch_size=64", result["args"])

    def test_device_count_duplicates_and_invalid_ordinals_fail_before_downstream(self):
        for count, devices in (
            ("8", "0,1,2,3"),
            ("0", "0"),
            ("4", "0,0,1,2"),
            ("4", "00,1,2,3"),
            ("4", "0,1,2,GPU-bad"),
        ):
            with self.subTest(count=count, devices=devices):
                process, result = self.run_launcher(
                    N_GPUS_PER_NODE=count, CUDA_VISIBLE_DEVICES=devices
                )
                self.assertNotEqual(process.returncode, 0)
                self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()

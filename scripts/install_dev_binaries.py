"""Install pinned Linux x86-64 maintenance binaries inside the project venv."""

import hashlib
import io
import platform
import sys
import tarfile
from pathlib import Path
from urllib.request import urlopen

TOOLS = (
    (
        "autocorrect",
        "https://github.com/huacnlee/autocorrect/releases/download/v2.16.3/autocorrect-linux-amd64.tar.gz",
        "9bef6fef5f30d6cfa2e5399ffc990672621ee8ea699d4feea75830264ed7a32e",
    ),
    (
        "gitleaks",
        "https://github.com/gitleaks/gitleaks/releases/download/v8.30.1/gitleaks_8.30.1_linux_x64.tar.gz",
        "551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb",
    ),
)


def main():
    root = Path(__file__).resolve().parents[1]
    destination = Path(sys.executable).absolute().parent
    if destination != root / ".venv/bin":
        raise SystemExit(
            "Run with this repository's .venv/bin/python; refusing to write another environment"
        )
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise SystemExit(
            "This installer supports Linux x86-64; install autocorrect and gitleaks with your OS package manager"
        )
    for name, url, digest in TOOLS:
        with urlopen(url, timeout=60) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != digest:
            raise SystemExit(f"Checksum mismatch: {name}")
        with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
            member = next(
                item
                for item in archive.getmembers()
                if Path(item.name).name == name and item.isfile()
            )
            with archive.extractfile(member) as stream:
                data = stream.read()
        target = destination / name
        temporary = destination / (name + ".installing")
        temporary.write_bytes(data)
        temporary.chmod(0o755)
        temporary.replace(target)
        print(f"Installed {name} (verified SHA-256) into {destination}")


if __name__ == "__main__":
    main()

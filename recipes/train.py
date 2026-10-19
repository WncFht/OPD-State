"""One native Hydra frontend, with immutable prepared execution."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def dispatch_prepared(directory):
    from recipes.snapshots import frozen_environment, verify_frozen_files

    root = Path(directory).expanduser().resolve(strict=True)
    entry = root / "source/recipes/train.py"
    if Path(__file__).resolve() != entry:
        verify_frozen_files(root)
        environment = frozen_environment(root)
        os.chdir(root / "source")
        os.execve(
            sys.executable,
            [sys.executable, "-m", "recipes.train", "--prepared", str(root)],
            environment,
        )
    from opd_state.processes import enable_subreaper
    from recipes.application import run_prepared

    enable_subreaper()
    return run_prepared(root, subreaper=True)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    selected = parser.add_mutually_exclusive_group(required=True)
    selected.add_argument("--backend", type=Path)
    selected.add_argument("--prepared", type=Path)
    args, overrides = parser.parse_known_args()
    if args.prepared is not None:
        if overrides:
            parser.error(
                "Prepared runs are immutable; overrides require new preparation"
            )
        return dispatch_prepared(args.prepared)
    import hydra

    from recipes.configuration import bind_sources, configure

    repository = Path(__file__).resolve().parents[1]
    backend = args.backend.expanduser().resolve()
    directory = configure(backend, repository=repository)
    sys.argv = [sys.argv[0], *overrides]

    @hydra.main(config_path=str(directory), config_name="train", version_base="1.3")
    def launch(config):
        from recipes.application import prepare

        bind_sources(config, backend=backend, repository=repository)
        manifest = prepare(config, backend=backend, repository=repository)
        if config.run.prepare_only:
            print(
                f"Prepared run {manifest['run_id']}: {Path(config.run.dir).resolve()}",
                flush=True,
            )
            return manifest
        return dispatch_prepared(config.run.dir)

    return launch()


if __name__ == "__main__":
    main()

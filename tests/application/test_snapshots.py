"""Installed package sources and repository helpers share frozen execution."""

from recipes.snapshots import remap_source, snapshot_sources, verify_snapshot


def test_installed_package_helper_is_remapped(tmp_path):
    repository = tmp_path / "repository"
    backend = tmp_path / "backend"
    package = tmp_path / "site-packages/opd_state"
    destination = tmp_path / "frozen"
    for root, name in (
        (repository, "recipes/__init__.py"),
        (backend, "verl/verl/__init__.py"),
        (package, "__init__.py"),
        (package, "scoring.py"),
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen source\n")
    snapshot_sources(repository, backend, destination, package=package)
    mapped = remap_source(
        package / "scoring.py", repository, backend, destination, package=package
    )
    assert mapped == str(destination / "opd_state/scoring.py")
    verify_snapshot(destination)

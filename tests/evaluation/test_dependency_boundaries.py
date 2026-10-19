"""Layer boundaries apply to every current module, without legacy path aliases."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src/opd_state"


def test_algorithm_layers_have_no_backend_or_application_dependencies():
    prohibited_layers = {
        "core": {"kernels", "evaluation", "integrations"},
        "kernels": {"evaluation", "integrations"},
        "evaluation": {"integrations"},
    }
    for layer, prohibited in prohibited_layers.items():
        for path in (ROOT / layer).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    assert not node.level, f"Use explicit package dependencies: {path}"
                    modules = [node.module or ""]
                else:
                    continue
                for module in modules:
                    assert module.split(".")[0] not in {
                        "verl",
                        "ray",
                        "vllm",
                        "recipes",
                        "tools",
                        "eval",
                    }, (path, module)
                    for dependency in prohibited:
                        assert not module.startswith(f"opd_state.{dependency}"), (
                            path,
                            module,
                        )
                    if layer == "core":
                        assert not module.startswith("torch"), (path, module)

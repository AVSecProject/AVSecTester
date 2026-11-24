"""Test imports stay inside the framework's declared public API."""

import ast
from pathlib import Path

from scripts.development.write_api_reference import public_modules


ROOT = Path(__file__).resolve().parents[2]


def _framework_accesses(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name, None
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                yield node.lineno, node.module, alias.name
        elif isinstance(node, ast.Call) and node.args:
            name = getattr(node.func, "attr", getattr(node.func, "id", None))
            target = node.args[0]
            if name in {"patch", "setattr"} and isinstance(target, ast.Constant):
                if isinstance(target.value, str) and "." in target.value:
                    module, attribute = target.value.rsplit(".", 1)
                    yield node.lineno, module, attribute


def test_public_api():
    exports = {module: set(names) for module, (_, _, names) in public_modules().items()}
    errors = []
    for path in sorted((ROOT / "tests").rglob("*.py")):
        for line, module, name in _framework_accesses(ast.parse(path.read_text())):
            if module != "avsectester" and not module.startswith("avsectester."):
                continue
            if module not in exports or name not in (None, "*") and name not in exports[module]:
                target = f"{module}.{name}" if name is not None else module
                errors.append(f"{path.relative_to(ROOT)}:{line}: {target} is not declared public")
    assert not errors, "\n".join(errors)

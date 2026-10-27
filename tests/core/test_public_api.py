"""Test imports stay inside the framework's declared public API."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_public_api():
    exports = {}
    errors = []
    for path in sorted((ROOT / "tests").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if not node.module.startswith("avsectester"):
                continue
            if node.module not in exports:
                source = ROOT.joinpath(*node.module.split(".")).with_suffix(".py")
                if not source.is_file():
                    source = ROOT.joinpath(*node.module.split("."), "__init__.py")
                declared = set()
                for statement in ast.parse(source.read_text()).body:
                    if isinstance(statement, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id == "__all__"
                        for target in statement.targets
                    ):
                        declared.update(ast.literal_eval(statement.value))
                exports[node.module] = declared
            for name in node.names:
                if name.name not in exports[node.module]:
                    errors.append(
                        f"{path.relative_to(ROOT)}:{node.lineno}: "
                        f"{node.module}.{name.name} is not declared public"
                    )
    assert not errors, "\n".join(errors)

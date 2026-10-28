"""Source dependency layers agree with the documented package architecture."""
import ast
import importlib.util
from pathlib import Path

from scripts.development import write_api_reference as api_reference

ROOT = Path(__file__).resolve().parents[2]


def _layer_rules():
    prefixes, exceptions = api_reference.architecture_rules()
    assert prefixes, "Architecture must declare dependency layers"
    for module, _ in prefixes:
        path = ROOT / module.replace(".", "/")
        assert path.is_dir() or path.with_suffix(".py").is_file(), f"Missing module: {module}"
    return prefixes, exceptions


def test_dependency_layers():
    prefixes, exceptions = _layer_rules()

    def level(name):
        matches = [(prefix, rank) for prefix, rank in prefixes
                   if name == prefix or name.startswith(prefix + ".")]
        assert matches, f"Module absent from layer table: {name}"
        return max(matches, key=lambda item: len(item[0]))[1]

    errors = []
    for path in (ROOT / "avsectester").rglob("*.py"):
        module = ".".join(path.relative_to(ROOT).with_suffix("").parts)
        package = module.rsplit(".", 1)[0]
        if module.endswith(".__init__"):
            module = package
        if module == "avsectester":
            continue

        def visit(node, typing_only=False):
            if isinstance(node, ast.If) and (
                isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING"
                or isinstance(node.test, ast.Attribute) and node.test.attr == "TYPE_CHECKING"
            ):
                for child in node.body:
                    visit(child, True)
                for child in node.orelse:
                    visit(child, typing_only)
                return
            targets = []
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                target = "." * node.level + (node.module or "")
                target = importlib.util.resolve_name(target, package) if node.level else target
                targets = [target]
                if target == "avsectester":
                    targets = [target + "." + alias.name for alias in node.names
                               if (ROOT / "avsectester" / alias.name).is_dir()
                               or (ROOT / "avsectester" / (alias.name + ".py")).is_file()]
            for target in targets:
                if not target.startswith("avsectester") or target == "avsectester":
                    continue
                if level(target) > level(module) and not (
                    typing_only and (module, target) in exceptions
                ):
                    errors.append(f"{module}:{node.lineno} -> {target}")
            for child in ast.iter_child_nodes(node):
                visit(child, typing_only)
        visit(ast.parse(path.read_text()))
    assert not errors, "\n".join(errors)

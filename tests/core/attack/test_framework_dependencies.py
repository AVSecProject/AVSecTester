"""Enforce the attack core, independent tools and concrete-attack boundaries."""
import ast
from importlib.util import resolve_name
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
ATTACK = ROOT / "avsectester" / "attack"


def _imports(path):
    module = ".".join(path.relative_to(ROOT).with_suffix("").parts)
    package = module.rsplit(".", 1)[0]
    if path.name == "__init__.py":
        package = module.removesuffix(".__init__")
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            name = "." * node.level + (node.module or "")
            resolved = resolve_name(name, package) if node.level else name
            yield resolved
            yield from (resolved + "." + alias.name for alias in node.names)


@pytest.mark.parametrize("path", sorted(ATTACK.rglob("*.py")),
                         ids=lambda path: str(path.relative_to(ATTACK)))
def test_attack_dependencies_follow_the_three_layers(path):
    relative = path.relative_to(ATTACK)
    layer = relative.parts[0] if len(relative.parts) > 1 else "core"
    own_attack = relative.parts[1] if layer == "implemented_attacks" and len(relative.parts) > 2 else None
    violations = []
    for name in _imports(path):
        parts = name.split(".")
        if parts[:2] != ["avsectester", "attack"] or len(parts) < 3:
            continue
        imported_layer = parts[2]
        if layer == "core" and imported_layer in {"insertion", "optimization", "implemented_attacks"}:
            violations.append(name)
        elif layer in {"insertion", "optimization"} and (
            imported_layer == "implemented_attacks"
            or imported_layer in {"insertion", "optimization"} and imported_layer != layer
        ):
            violations.append(name)
        elif (own_attack is not None and imported_layer == "implemented_attacks"
              and len(parts) > 3 and parts[3] != own_attack):
            violations.append(name)
    assert not violations, f"{relative} imports across a forbidden boundary: {violations}"


def test_framework_imports_respect_attack_dependency_boundaries():
    code = """
import importlib.abc
import sys
class ForbiddenAttackDependency(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        forbidden = ("avsectester.attack.implemented_attacks",
                     "avsectester.attack.insertion", "avsectester.attack.optimization")
        if fullname.startswith(forbidden):
            raise AssertionError("Framework imported " + fullname)
sys.meta_path.insert(0, ForbiddenAttackDependency())
import avsectester.attack.attack
import avsectester.attack.runtime
import avsectester.attack.segments
import avsectester.attack.placement
import avsectester.closed_loop.simulators.carla
import avsectester.closed_loop.simulators.nurec
import avsectester.closed_loop.hooks
import avsectester.closed_loop.avstacks.native_hook
import avsectester.closed_loop.services
import avsectester.experiment
import avsectester.runlog
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)

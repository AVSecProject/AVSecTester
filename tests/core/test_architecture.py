"""Source dependency layers agree with the documented package architecture."""
import ast
import importlib.util
from pathlib import Path

from scripts.development import write_api_reference as api_reference

ROOT = Path(__file__).resolve().parents[2]
CARLA = "closed_loop.simulators.carla"
AUTOWARE_CONTAINER = CARLA + ".autoware.container"
AVSTACKS = "closed_loop.avstacks.implemented"
NUREC = "closed_loop.simulators.nurec"
ROS_INTERCEPTION = "closed_loop.ros_interception"
ALPAMAYO = AVSTACKS + ".alpamayo"
CAMERA_COLLISION = AVSTACKS + ".camera_forward_collision"
MODULAR_STACK = AVSTACKS + ".modular"
DETECTOR = AVSTACKS + ".modules.detector"
SCORERS = AVSTACKS + ".scorers"

OPTIONAL_DEPENDENCY_OWNERS = {
    "carla": (CARLA, "maps.opendrive", "implemented_attacks.patch_attack.online"),
    "avcarla": (CARLA,),
    "avstack": (CAMERA_COLLISION, MODULAR_STACK, "implemented_attacks.phantom_injection_attack.online"),
    "alpasim_controller": ("closed_loop.simulators.ego_dynamics",),
    "alpasim_driver": (ALPAMAYO,),
    "alpasim_grpc": (ALPAMAYO, NUREC, "closed_loop.simulators.ego_dynamics"),
    "alpasim_utils": (ALPAMAYO,),
    "autoware_adapi_v1_msgs": (AUTOWARE_CONTAINER,),
    "geometry_msgs": (AUTOWARE_CONTAINER, ROS_INTERCEPTION),
    "nav_msgs": (AUTOWARE_CONTAINER, ROS_INTERCEPTION),
    "sensor_msgs": (AUTOWARE_CONTAINER, ROS_INTERCEPTION),
    "rclpy": (AUTOWARE_CONTAINER, ROS_INTERCEPTION),
    "rosidl_runtime_py": (AUTOWARE_CONTAINER, ROS_INTERCEPTION),
    "grpc": (NUREC,),
    "embreex": ("geometry.mesh",),
    "huggingface_hub": ("rendering.harmonizers",),
    "mmdet": (CAMERA_COLLISION, SCORERS),
    "mmdet3d": (MODULAR_STACK,),
    "nuscenes": ("datasets.nuscenes",),
    "torch": (
        "attack.optimization", ALPAMAYO, DETECTOR, SCORERS, NUREC + ".scene", "rendering",
        "implemented_attacks.patch_attack.optimization_data",
        "implemented_attacks.stop_sign_patch_attack.detector_optimization",
    ),
    "torchvision": (DETECTOR,),
    "yolox": (DETECTOR,),
}


def _source_modules():
    for path in sorted((ROOT / "avsectester").rglob("*.py")):
        module = ".".join(path.relative_to(ROOT).with_suffix("").parts)
        package = module.rsplit(".", 1)[0]
        yield module.removesuffix(".__init__"), package, ast.parse(path.read_text())


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
    for module, package, source in _source_modules():
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
            elif isinstance(node, ast.Call) and node.args:
                name = getattr(node.func, "attr", getattr(node.func, "id", None))
                target = node.args[0]
                if name == "import_module" and isinstance(target, ast.Constant):
                    if isinstance(target.value, str):
                        targets = [target.value]
            for target in targets:
                if not target.startswith("avsectester") or target == "avsectester":
                    continue
                if level(target) > level(module) and not (
                    typing_only and (module, target) in exceptions
                ):
                    errors.append(f"{module}:{node.lineno} -> {target}")
                if not _allowed_peer_dependency(module, target):
                    errors.append(f"{module}:{node.lineno} -> {target}")
            for child in ast.iter_child_nodes(node):
                visit(child, typing_only)
        visit(source)
    assert not errors, "\n".join(errors)


def _allowed_peer_dependency(module, target):
    source, dependency = module.split("."), target.split(".")
    peers = {"carla", "nurec"}
    if (source[1:3] == dependency[1:3] == ["closed_loop", "simulators"]
            and len(source) > 3 and len(dependency) > 3
            and source[3] in peers and dependency[3] in peers and source[3] != dependency[3]):
        return False
    if source[1] == "maps" and dependency[1] == "datasets":
        return False
    if dependency[1] == "implemented_attacks":
        if source[1] == "attack":
            return False
        if source[1] == "implemented_attacks":
            return source[:3] == dependency[:3]
    return True


def test_optional_dependency_ownership():
    errors = []
    for module, _, source in _source_modules():
        owner = module.removeprefix("avsectester.")
        for node in ast.walk(source):
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                targets = [node.module]
            else:
                continue
            for target in targets:
                allowed = OPTIONAL_DEPENDENCY_OWNERS.get(target.split(".")[0])
                if allowed and not any(owner == prefix or owner.startswith(prefix + ".") for prefix in allowed):
                    errors.append(f"{module}:{node.lineno} imports {target} outside {', '.join(allowed)}")
    assert not errors, "\n".join(errors)

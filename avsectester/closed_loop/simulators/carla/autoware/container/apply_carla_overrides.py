"""Apply the official CARLA integration's three Autoware configuration overrides."""
import argparse
import os
from pathlib import Path
import re


OVERRIDE_VERSION = 1


def _target_files(root: Path) -> dict[str, Path]:
    targets = {
        "ndt_scan_matcher.param.yaml": "",
        "pose_twist_estimator.launch.xml": "tier4_localization_launch",
        "autoware-carla.yaml": "autoware_launch",
    }
    found = {}
    visited = set()
    for directory, children, files in os.walk(root, followlinks=True):
        canonical = os.path.realpath(directory)
        if canonical in visited:
            children.clear()
            continue
        visited.add(canonical)
        for name in targets.keys() - found.keys():
            if name in files and targets[name] in directory:
                found[name] = Path(directory) / name
        if found.keys() == targets.keys():
            return found
    absent = ", ".join(sorted(targets.keys() - found.keys()))
    raise FileNotFoundError(f"Missing CARLA override files under {root}: {absent}")


def _patch(path: Path, pattern: bytes, replacement: bytes) -> bytes:
    content, count = re.subn(pattern, replacement, path.read_bytes())
    if not count:
        raise ValueError(f"No matching CARLA override setting in {path}")
    return content


def _patch_diagnostics(path: Path) -> bytes:
    lines = path.read_text().splitlines(keepends=True)
    targets = {"/autoware/localization/state", "/adapi/mrm_request/delegate"}
    replaced = set()
    current = None
    for index, line in enumerate(lines):
        match = re.match(r"\s*- path: (\S+)", line)
        if match:
            current = match[1]
        elif current in targets and re.match(r"\s*timeout:", line):
            lines[index], count = re.subn(r"timeout:\s*[0-9.]+", "timeout: 30.0", line)
            if count:
                replaced.add(current)
            current = None
    if replaced != targets:
        absent = ", ".join(sorted(targets - replaced))
        raise ValueError(f"Missing ADAPI diagnostic timeouts in {path}: {absent}")
    return "".join(lines).encode()


def _apply(root: Path) -> None:
    targets = _target_files(root)
    ndt = targets["ndt_scan_matcher.param.yaml"]
    twist = targets["pose_twist_estimator.launch.xml"]
    diag = targets["autoware-carla.yaml"]
    patches = {
        ndt: _patch(ndt, rb"(converged_param_nearest_voxel_transformation_likelihood:)[\t ]*[\d.]+",
                    rb"\1 1.0"),
        twist: _patch(twist, rb'(stop_check_enabled"[\t ]*(default|value)=")[^"]*(")',
                      rb"\1false\3"),
        diag: _patch_diagnostics(diag),
    }
    for path, content in patches.items():
        path.write_bytes(content)
        print(f"patched: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("OVERRIDE_ROOT",
                                                                       "/opt/autoware")))
    _apply(parser.parse_args().root)


if __name__ == "__main__":
    main()

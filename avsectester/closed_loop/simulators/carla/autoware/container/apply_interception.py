"""Redirect selected Autoware outputs in the writable layer of a run container."""
import argparse
import json
from pathlib import Path


def apply_rewrites(specification: Path, root: Path) -> None:
    """Require each producer definition exactly once before writing any launch file."""
    rewrites = json.loads(specification.read_text())
    changed: dict[Path, str] = {}
    for rewrite in rewrites:
        path = root / rewrite["path"]
        topic = rewrite["topic"]
        if not path.is_file():
            raise ValueError(f"{topic}: missing Autoware launch file {path}")
        source = changed[path] if path in changed else path.read_text()
        original = rewrite["original"]
        count = source.count(original)
        if count != 1:
            raise ValueError(f"{topic}: expected one producer definition in {path}, found {count}")
        changed[path] = source.replace(original, rewrite["replacement"], 1)
    for path, contents in changed.items():
        path.write_text(contents)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("specification", type=Path)
    parser.add_argument("--root", type=Path, default=Path("/opt/autoware"))
    options = parser.parse_args()
    apply_rewrites(options.specification, options.root)


__all__ = ["apply_rewrites"]


if __name__ == "__main__":
    main()

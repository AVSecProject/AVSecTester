"""Dataset-specific metadata files through their public save/load API."""

from dataclasses import replace
import json

import numpy as np
import pytest

from avsectester.attack import AttackSegment
from avsectester.experiment import load_metadata, save_metadata
from avsectester.serialization import DEFAULT_REGISTRY
from tests.fakes import MemoryDataset


def test_save_metadata(tmp_path):
    frames = MemoryDataset().frames("scene")
    segment = AttackSegment("scene", frames[0], frames[2], {"route": np.zeros((2, 3))})
    path = tmp_path / "metadata" / "approaches.json"
    save_metadata((segment,), path, dataset_type="tests.fakes:MemoryDataset")
    content = path.read_bytes()
    save_metadata((segment,), path, dataset_type="tests.fakes:MemoryDataset")
    assert path.read_bytes() == content
    document = json.loads(content)
    assert document["format"] == "avsectester.metadata"
    assert document["version"] == 1
    assert document["dataset_type"] == "tests.fakes:MemoryDataset"
    assert document["segments"] == [DEFAULT_REGISTRY.encode(segment)]
    changed = replace(segment, stop=frames[3])
    with pytest.raises(ValueError, match="different content"):
        save_metadata((changed,), path, dataset_type="tests.fakes:MemoryDataset")
    assert path.read_bytes() == content
    with pytest.raises(ValueError, match="indices 0 and 1"):
        save_metadata((segment, replace(segment)), tmp_path / "duplicate.json",
                      dataset_type="tests.fakes:MemoryDataset")
    variant = replace(segment, user_data={"variant": 1})
    save_metadata((segment, variant), tmp_path / "variants.json", dataset_type="tests.fakes:MemoryDataset")
    assert len(load_metadata(tmp_path / "variants.json")) == 2


def test_load_metadata(tmp_path):
    frames = MemoryDataset().frames("scene")
    segments = (AttackSegment("scene", frames[0], frames[2], {"route": np.zeros((2, 3))}),
                AttackSegment("scene", frames[2], frames[4], {"key": "second"}))
    path = tmp_path / "selected.json"
    save_metadata(segments, path, dataset_type="tests.fakes:MemoryDataset")
    assert DEFAULT_REGISTRY.encode(load_metadata(path)) == DEFAULT_REGISTRY.encode(segments)
    document = json.loads(path.read_text())
    path.write_text(json.dumps({**document, "format": "other"}))
    with pytest.raises(ValueError, match="format"):
        load_metadata(path)
    path.write_text(json.dumps({**document, "version": 2}))
    with pytest.raises(ValueError, match="version"):
        load_metadata(path)
    path.write_text(json.dumps({**document, "dataset_type": "MemoryDataset"}))
    with pytest.raises(ValueError, match="dataset_type"):
        load_metadata(path)
    path.write_text(json.dumps({**document, "segments": [1]}))
    with pytest.raises(ValueError, match="segment 0.*AttackSegment"):
        load_metadata(path)

    path.write_text(json.dumps({**document, "segments": document["segments"] + [document["segments"][0]]}))
    with pytest.raises(ValueError, match="indices 0 and 2"):
        load_metadata(path)

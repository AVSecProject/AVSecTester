"""Driving inputs selected by the shared concrete-attack template."""

import pytest


@pytest.fixture
def default_driving_options():
    return {}


@pytest.fixture
def native_driving_options():
    pytest.importorskip("avstack.config")
    from avstack.datastructs import DataContainer
    from avstack.geometry import GlobalOrigin3D

    return {"perception_output": DataContainer(0, 0., [], "front", source_reference=GlobalOrigin3D)}

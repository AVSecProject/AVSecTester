"""The driving-system interface owns its typed observation pipeline."""

import pytest

from avsectester.closed_loop.avstacks.base import ObservationPipeline
from tests.fakes import ConstantAVStack


def test_observation_pipeline():
    avstack = ConstantAVStack()
    assert isinstance(avstack.observation_pipeline, ObservationPipeline)
    pipeline = ObservationPipeline()
    avstack.observation_pipeline = pipeline
    assert avstack.observation_pipeline is pipeline
    with pytest.raises(TypeError, match="observation_pipeline must be ObservationPipeline"):
        avstack.observation_pipeline = object()

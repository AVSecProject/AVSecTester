"""Smoke tests for the scaffold — these must pass without the heavy avstack stack."""

from avsectester.config import ATTACKS, BACKENDS
from avsectester.core import EscalationDAG, EscalationEdge, EscalationNode, ExperimentSpec
from avsectester.core.escalation import Stage


def test_registries_populated_by_import():
    # Importing the plugin packages should self-register concrete plugins.
    import avsectester.attacks
    import avsectester.backends.carla_backend
    import avsectester.backends.dataset_backend  # noqa: F401

    assert "LidarSpoofAttack" in ATTACKS
    assert "CarlaBackend" in BACKENDS
    assert "DatasetBackend" in BACKENDS


def test_experiment_spec_roundtrip(tmp_path):
    spec = ExperimentSpec.model_validate(
        {
            "name": "t",
            "system": {"name": "s"},
            "scenario": {"backend": {"type": "DatasetBackend"}},
        }
    )
    p = tmp_path / "spec.yaml"
    spec.to_yaml(str(p))
    assert ExperimentSpec.from_yaml(str(p)).name == "t"


def test_escalation_dag_paths():
    dag = EscalationDAG()
    dag.add_node(EscalationNode("a", Stage.ATTACK_SURFACE, "lidar"))
    dag.add_node(EscalationNode("b", Stage.PERCEPTION, "detector"))
    dag.add_node(EscalationNode("c", Stage.CONSEQUENCE, "ego"))
    dag.add_edge(EscalationEdge("a", "b", kind="triggered"))
    dag.add_edge(EscalationEdge("b", "c", kind="amplified"))
    assert dag.consequence_paths() == [["a", "b", "c"]]
    assert dag.root_cause().id == "a"


def test_example_config_validates():
    spec = ExperimentSpec.from_yaml("configs/example_experiment.yaml")
    assert spec.attack.spec["type"] == "LidarSpoofAttack"

"""User attacks can import optimization tools without knowing implementation modules."""
import subprocess
import sys


def test_public_optimization_exports_are_usable_classes():
    from avsectester.attack.optimization import (
        AdversarialSample, DataSource, GradientOptimizer, HideObject, LinfImage, NES,
        OptimizationObjective, OptimizationResult, PatchPerturbation, Perturbation, PGD,
        Scorer, SessionScorer, SpoofObject, TargetSpec,
    )

    assert all(isinstance(value, type) for value in (
        AdversarialSample, DataSource, GradientOptimizer, HideObject, LinfImage, NES,
        OptimizationObjective, OptimizationResult, PatchPerturbation, Perturbation, PGD,
        Scorer, SessionScorer, SpoofObject, TargetSpec,
    ))
    assert isinstance(PGD(steps=1), GradientOptimizer)
    assert isinstance(NES(steps=1, population_size=2), GradientOptimizer)
    assert isinstance(LinfImage(device="cpu"), Perturbation)
    assert isinstance(PatchPerturbation(device="cpu"), Perturbation)


def test_contract_and_nes_import_without_loading_torch():
    result = subprocess.run([
        sys.executable, "-c",
        "import sys; from avsectester.attack.optimization import DataSource, NES, SessionScorer; "
        "assert 'torch' not in sys.modules; assert NES().population_size == 20",
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

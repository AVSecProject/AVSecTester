"""Epoch-wise PGD batches and batched NES queries preserve their public optimization contracts."""

import numpy as np
import pytest
import torch

from avsectester.attack.optimization import (
    AdversarialSample, DataSource, HideBelowThreshold, HideObject, NES, PGD, Perturbation, Scorer,
    SpoofObject,
)
from avsectester.data_types import TargetSpec


class ImageSource(DataSource):
    def __init__(self, count=1):
        self.count = count
        self.accesses = []

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        self.accesses.append(index)
        return AdversarialSample(None, TargetSpec((index, 0, index + 1, 1)),
                                 {"index": index})


class TensorImage(Perturbation):
    def __init__(self):
        self.updates = 0

    def init(self):
        return torch.full((3, 4, 4), .5, requires_grad=True)

    def apply(self, sample, delta):
        return delta

    def project(self, delta):
        self.updates += 1
        return delta.clamp(0, 1)

    def export(self, delta):
        return (delta.detach().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)


class ArrayImage(Perturbation):
    def init(self):
        return np.full((3, 4, 4), .5, np.float32)

    def apply(self, sample, delta):
        return delta

    def project(self, delta):
        return np.clip(delta, 0, 1)

    def export(self, delta):
        return (delta.transpose(1, 2, 0) * 255).astype(np.uint8)


class TensorBrightnessScore(Scorer):
    def target_scores(self, images, targets):
        assert len(images) == len(targets)
        return images.mean(dim=(1, 2, 3))


class ArrayBrightnessScore(Scorer):
    differentiable = False

    def __init__(self):
        self.batch_sizes = []

    def target_scores(self, images, targets):
        assert len(images) == len(targets)
        self.batch_sizes.append(len(images))
        return np.asarray([image.mean() for image in images])


@pytest.mark.parametrize("optimizer,perturbation,scorer,last_index", [
    pytest.param(PGD(epochs=8, step_size=.1, batch_size=1),
                 TensorImage(), TensorBrightnessScore(), 8, id="PGD"),
    pytest.param(NES(steps=12, population_size=10, learning_rate=.3, sigma=.1, seed=0),
                 ArrayImage(), ArrayBrightnessScore(), 12, id="NES"),
])
@pytest.mark.parametrize("objective", [HideObject(), SpoofObject()], ids=["hide", "spoof"])
def test_optimize(optimizer, perturbation, scorer, last_index, objective):
    observed = []
    result = optimizer.run(
        ImageSource(), perturbation, scorer, objective,
        monitor=lambda index, delta: observed.append((index, delta.shape)),
    )
    assert result.history[-1] < result.history[0]
    assert observed == [(index, (3, 4, 4)) for index in range(last_index + 1)]
    assert result.export.shape == (4, 4, 3) and result.export.dtype == np.uint8
    final_score, = scorer.target_scores(result.delta[None], [TargetSpec((0, 0, 1, 1))])
    assert result.final_score == pytest.approx(float(final_score))


class IdentifiedImage(TensorImage):
    def apply(self, sample, delta):
        image = delta.clone()
        image[0, 0, 0] = sample.context["index"] / 10
        return image


class IdentifiedScore(TensorBrightnessScore):
    def __init__(self):
        self.batch_sizes = []

    def target_scores(self, images, targets):
        self.batch_sizes.append(len(images))
        expected = torch.tensor([target.box[0] / 10 for target in targets])
        torch.testing.assert_close(images[:, 0, 0, 0], expected)
        return super().target_scores(images, targets)


class RaggedImage(TensorImage):
    def apply(self, sample, delta):
        return delta[:, :, :sample.context["index"] + 2]


def test_pgd_epochs():
    source = ImageSource(5)
    perturbation, scorer = IdentifiedImage(), IdentifiedScore()
    observed = []
    result = PGD(epochs=2, batch_size=2, step_size=0., seed=7).run(
        source, perturbation, scorer, HideObject(),
        monitor=lambda epoch, delta: observed.append(epoch),
    )
    expected_rng = np.random.default_rng(7)
    expected = [int(index) for _ in range(2) for index in expected_rng.permutation(5)]
    assert source.accesses == expected + list(range(5))
    assert sorted(source.accesses[:5]) == sorted(source.accesses[5:10]) == list(range(5))
    assert perturbation.updates == 2 * 3
    assert scorer.batch_sizes == [2, 2, 1] * 3
    assert result.history == pytest.approx([(47 * .5 + .2) / 48] * 2)
    assert observed == [0, 1, 2]
    with pytest.raises(IndexError):
        source[5]
    with pytest.raises(ValueError, match="PGD requires a nonempty data source"):
        PGD().run(ImageSource(0), TensorImage(), TensorBrightnessScore(), HideObject())
    with pytest.raises(ValueError, match="Images in the same batch must have equal dimensions"):
        PGD(epochs=1, batch_size=2).run(
            ImageSource(2), RaggedImage(), TensorBrightnessScore(), HideObject())
    for options in ({"epochs": 0}, {"epochs": True}, {"batch_size": 0}, {"batch_size": 1.5}):
        with pytest.raises(ValueError, match="positive integer"):
            PGD(**options)


class OpposingImage(TensorImage):
    def apply(self, sample, delta):
        return delta if sample.context["index"] == 0 else 2 - 2 * delta


def test_pgd_batch_gradient():
    perturbation = OpposingImage()
    result = PGD(epochs=1, batch_size=2, step_size=.1).run(
        ImageSource(2), perturbation, TensorBrightnessScore(), HideObject())
    # The gradients +1 and -2 average to a negative gradient, giving one positive update.
    torch.testing.assert_close(result.delta, torch.full((3, 4, 4), .6))
    assert perturbation.updates == 1
    assert result.history == pytest.approx([.75])
    assert result.final_score == pytest.approx(.7)


def test_nes_candidate_batches():
    scores_one, scores_four = ArrayBrightnessScore(), ArrayBrightnessScore()
    source_one, source_four = ImageSource(5), ImageSource(5)
    options = dict(steps=3, population_size=10, learning_rate=.3, sigma=.1, seed=7)
    one = NES(**options, batch_size=1).run(
        source_one, ArrayImage(), scores_one, HideObject())
    four = NES(**options, batch_size=4).run(
        source_four, ArrayImage(), scores_four, HideObject())
    np.testing.assert_array_equal(one.delta, four.delta)
    assert one.history == four.history
    assert one.final_score == four.final_score
    assert source_one.accesses == source_four.accesses
    assert source_one.accesses == [int(np.random.default_rng(7).integers(5))]
    assert all(size == 1 for size in scores_one.batch_sizes)
    # The initial evaluation, then each step's candidates and its updated variable; no extra query.
    assert scores_four.batch_sizes == [1] + [4, 4, 2, 1] * 3
    with pytest.raises(ValueError, match="NES requires a nonempty data source"):
        NES().run(ImageSource(0), ArrayImage(), ArrayBrightnessScore(), HideObject())
    with pytest.raises(ValueError, match="positive integer"):
        NES(batch_size=False)


class CheckerImage(TensorImage):
    def init(self):
        image = torch.zeros(3, 4, 4)
        image[:, ::2, ::2] = image[:, 1::2, 1::2] = 1.
        return image.requires_grad_(True)


class ConstantScore(Scorer):
    def target_scores(self, images, targets):
        return images.sum(dim=(1, 2, 3)) * 0 + .5


def test_pgd_smoothness():
    checker = CheckerImage()
    assert checker.smoothness(checker.init()).item() == pytest.approx(2.)
    plain = PGD(epochs=4, step_size=.1, batch_size=1).run(
        ImageSource(), checker, ConstantScore(), HideObject())
    smooth = PGD(epochs=4, step_size=.1, batch_size=1, tv_weight=1.).run(
        ImageSource(), checker, ConstantScore(), HideObject())
    torch.testing.assert_close(plain.delta, checker.init().detach())
    assert checker.smoothness(smooth.delta) < checker.smoothness(plain.delta)
    # History records only the objective term, excluding TV.
    assert smooth.history == plain.history == [.5] * 4


def test_objectives():
    scores = torch.tensor([.2, .35, .45, .9], requires_grad=True)
    torch.testing.assert_close(HideObject().loss(scores), scores)
    torch.testing.assert_close(SpoofObject().loss(scores), 1 - scores)
    objective = HideBelowThreshold(.5, margin=.1)
    losses = objective.loss(scores)
    torch.testing.assert_close(losses, torch.tensor([0., 0., .05, .5]))
    losses.sum().backward()
    # Scores below threshold - margin have zero gradient.
    torch.testing.assert_close(scores.grad, torch.tensor([0., 0., 1., 1.]))
    assert objective.loss(np.float64(.7)) == pytest.approx(.3)
    for threshold, margin in ((1.5, .1), (.5, -.1), (True, .1)):
        with pytest.raises(ValueError, match="threshold|margin"):
            HideBelowThreshold(threshold, margin)

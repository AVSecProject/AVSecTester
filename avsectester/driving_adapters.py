"""Driving-system component, scorer and gradient declarations."""
from .services import ComponentDescriptor, DrivingAdapter, ScorerDescriptor


def modular_driving_adapter(factory, *, components=(), scorers=()):
    """Expose explicitly selected native Modular components and existing detector scorers.

    Component inputs are {args: tuple, kwargs: dict}, matching that native component's
    call signature. MMDet scorer inputs follow the existing scorer API and TargetSpec.
    Declaring a scorer is an explicit assertion that the factory supplies that model.
    """
    bindings = []
    for name in components:
        def evaluate(stack, inputs, name=name):
            if name not in stack.native:
                raise ValueError(f"Driving system does not expose component {name!r}")
            return stack.native[name](*inputs.get("args", ()), **inputs.get("kwargs", {}))
        bindings.append((ComponentDescriptor(
            name, {"args": "native positional arguments", "kwargs": "native keyword arguments"},
            {"value": "native component output"}, {"value": "adapter codecs"},
        ), evaluate))
    scores = []
    for name in scorers:
        if name not in {"mmdet.inference", "mmdet.rpn_objectness"}:
            raise ValueError(f"Unknown built-in scorer {name!r}")
        differentiable = name == "mmdet.rpn_objectness"
        def score(stack, inputs, target, differentiable=differentiable):
            from .attacks.optim.scorers import MMDetInferenceScorer, MMDetRPNObjectnessScorer

            model = stack.models["perception"]
            scorer = MMDetRPNObjectnessScorer(model) if differentiable else MMDetInferenceScorer(model)
            return scorer.target_score(inputs, target)
        scores.append((ScorerDescriptor(
            name, {"image": "CHW float tensor [0,1]" if differentiable else "HWC RGB uint8"},
            {"score": "scalar confidence"}, {"score": "tensor" if differentiable else "float"},
            ("image",) if differentiable else (), (), "minimize",
        ), score))
    return DrivingAdapter(factory, components=bindings, scorers=scores)


def alpamayo_driving_adapter(factory):
    """Expose a fresh end-to-end policy decision, including the adapter's history reset."""
    def evaluate(stack, observation):
        stack.reset(observation)
        return stack(observation)

    return DrivingAdapter(factory, components=((ComponentDescriptor(
        "policy", {"observation": "Observation with configured camera IDs and ego pose"},
        {"command": "Control with rig-frame trajectory"}, {"command": "Control"},
    ), evaluate),))

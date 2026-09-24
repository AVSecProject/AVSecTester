"""Natural-language -> formal scenario requirement, via an LLM.

An attack's scenario requirement can be stated in plain English ("a car directly ahead, close and
mostly unoccluded, its rear facing us"). An LLM turns that into the formal
:class:`~avsectester.scenarios.requirement.ScenarioRequirement` (target + constraints) that the
providers execute — so a human writes intent and the machine gets an exact, runnable predicate.

The LLM is **injected** (``llm: Callable[[str], str]``), so this module has no API dependency and is
fully testable with a stub. :func:`build_prompt` grounds the model in the exact constraint vocabulary
(from :func:`~avsectester.scenarios.serialize.constraint_vocabulary`), :func:`parse_response` extracts
the JSON it returns, and :func:`interpret` ties them together into a ``ScenarioRequirement``.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from avsectester.scenarios.requirement import ScenarioRequirement
from avsectester.scenarios.serialize import constraint_vocabulary, requirement_from_dict

LLM = Callable[[str], str]

_EXAMPLE = {
    "name": "physical_patch_hide_vehicle",
    "description": "a vehicle directly ahead, close and mostly unoccluded, its rear facing the ego",
    "target": {"category": "vehicle", "camera": "front", "select": "nearest_ahead"},
    "constraints": [
        {"kind": "InView", "camera": "front"},
        {"kind": "DistanceRange", "min_m": 4.0, "max_m": 25.0},
        {"kind": "ImageAreaFrac", "min_frac": 0.02, "max_frac": 0.5, "camera": "front"},
        {"kind": "ViewpointRear", "max_deg": 35.0},
        {"kind": "MinVisibility", "min_vis": 0.7},
    ],
}


def build_prompt(description: str, name: str = "custom") -> str:
    """Prompt an LLM to emit the JSON requirement for ``description``, grounded in the real vocabulary."""
    vocab = json.dumps(constraint_vocabulary(), indent=2)
    example = json.dumps(_EXAMPLE, indent=2)
    return (
        "You convert a natural-language autonomous-driving scenario requirement into a strict JSON "
        "object describing which scene an attack needs.\n\n"
        "Output ONLY JSON of the form:\n"
        '{"name": str, "description": str, '
        '"target": {"category": str, "camera": str, "select": '
        '"nearest_ahead"|"nearest"|"largest"}, '
        '"constraints": [{"kind": str, ...fields}]}\n\n'
        f"Available constraint kinds and their fields (use ONLY these, with these field names):\n{vocab}\n\n"
        f"Example:\nDescription: {_EXAMPLE['description']}\nJSON:\n{example}\n\n"
        f'Now convert this description (use name "{name}"):\n"""{description}"""\nJSON:'
    )


def parse_response(text: str) -> dict:
    """Extract the JSON object from an LLM response (tolerates ```json fences / surrounding prose)."""
    s = text.strip()
    if "```" in s:  # pull the fenced block
        s = s.split("```", 2)[1]
        s = s[4:] if s.lstrip().lower().startswith("json") else s
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in LLM response: {text[:200]!r}")
    return json.loads(s[start:end + 1])


def interpret(description: str, llm: LLM, name: str = "custom") -> ScenarioRequirement:
    """Interpret a natural-language ``description`` into a :class:`ScenarioRequirement` using ``llm``.

    ``llm`` is any ``str -> str`` completion callable (a real client, or a stub in tests). The response's
    JSON is validated by deserializing through the constraint registry, so an unknown constraint kind or
    bad field raises rather than silently mis-specifying the scenario."""
    d = parse_response(llm(build_prompt(description, name)))
    d.setdefault("name", name)
    d.setdefault("description", description)
    return requirement_from_dict(d)

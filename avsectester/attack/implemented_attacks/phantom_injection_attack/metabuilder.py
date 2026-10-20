"""Select consecutive decision windows for this attack."""
from avsectester.attack.segments import split_segments


def build_metadata(attack, scenario):
    result = split_segments(scenario.frames, lambda frame: True,
                            decision_steps=attack.decision_steps, keep_tail=attack.keep_tail)
    return result if attack.max_segments is None else result[:attack.max_segments]

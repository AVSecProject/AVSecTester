"""Select consecutive decision windows for this attack."""
from avsectester.attack.segments import split_segments


def build_metadata(attack, scenario):
    with scenario.services.session(scenario.seed) as session:
        def eligible(frame):
            if not session.can_start_at(frame.position):
                scenario.record_reason("Frame lacks history or deployment space for a run start")
                return False
            return True

        result = split_segments(scenario.frames, eligible,
                                decision_steps=attack.decision_steps, keep_tail=attack.keep_tail)
    return result if attack.max_segments is None else result[:attack.max_segments]

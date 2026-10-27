"""The run log: one record of everything captured during a run, with Parquet persistence.

`avsectester.runlog.log` holds the records, the recorder and the readers.
Record and native conversions share `avsectester.serialization`.
"""

from avsectester.runlog.log import (
    EventRecord,
    Recorder,
    RunLog,
    StepRecord,
    read_table,
)


__all__ = [
    'EventRecord',
    'Recorder',
    'RunLog',
    'StepRecord',
    'read_table',
]

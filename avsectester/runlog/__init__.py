"""The run log: one record of everything captured during a run, with Parquet persistence.

:mod:`~avsectester.runlog.log` holds the records, the recorder and the readers;
:mod:`~avsectester.runlog.codecs` holds the native-object codecs of the supplied adapters.
"""

from avsectester.runlog.log import (
    CodecRegistry,
    EventRecord,
    Recorder,
    RunLog,
    StepRecord,
    read_table,
)

__all__ = ["CodecRegistry", "EventRecord", "Recorder", "RunLog", "StepRecord", "read_table"]

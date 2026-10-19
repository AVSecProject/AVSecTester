"""Lazy adapter registration scans each concrete value type only once."""

import sys
from types import SimpleNamespace

import avsectester.runlog.codecs as codecs
from avsectester.runlog import Recorder


def test_specs_scanned_once_per_type_across_repeated_captures(monkeypatch):
    original = codecs._SPECS

    class ScanSpy:
        count = 0

        def __iter__(self):
            self.count += 1
            return iter(original)

    spy = ScanSpy()
    monkeypatch.setattr(codecs, "_SPECS", spy)
    recorder = Recorder(codecs=codecs.default_codecs())
    for _ in range(5):
        recorder.capture({"list": [1, 2], "tuple": (3, 4)})
    assert spy.count == 3  # dict, list and tuple; primitives never need codec dispatch


def test_first_value_from_late_module_registers_its_codec(monkeypatch):
    module_name = "avsectester_test_late_codec"
    monkeypatch.setattr(codecs, "_SPECS", (
        (module_name, "Native", lambda value: {"value": value.value}),
    ))
    registry = codecs.default_codecs()
    recorder = Recorder(codecs=registry)
    recorder.capture({"earlier": [1]})

    class Native:
        value = 7

    monkeypatch.setitem(sys.modules, module_name, SimpleNamespace(Native=Native))
    value = Native()
    assert type(recorder.capture(value)) is Native
    label, encode = registry.encoder(value)
    assert label == f"{module_name}.Native.v1" and encode(value) == {"value": 7}


def test_late_subclass_codec_takes_precedence_over_registered_base(monkeypatch):
    module_name = "avsectester_test_subclass_codec"
    class Base:
        pass

    module = SimpleNamespace(Base=Base)
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(codecs, "_SPECS", (
        (module_name, "Base", lambda value: "base"),
        (module_name, "Derived", lambda value: "derived"),
    ))
    registry = codecs.default_codecs()
    assert registry.encoder(Base())[1](Base()) == "base"

    class Derived(Base):
        pass

    module.Derived = Derived
    label, encode = registry.encoder(Derived())
    assert label == f"{module_name}.Derived.v1" and encode(Derived()) == "derived"

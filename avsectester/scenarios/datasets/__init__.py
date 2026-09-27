"""Concrete :class:`~avsectester.scenarios.source.Dataset` adapters — real traces -> ``SceneGT``.

``nuscenes.NuScenesDataset`` reads nuScenes ground-truth annotations. Alpamayo/NuRec has no labels
available here (the render API exposes no actor boxes and no annotation files ship with the clips), so
a real-data filter over it would need externally-provided labels or a 3-D detector's output as SceneGT.
"""

"""Real-trace GT adapters — turning recorded data into filterable :class:`SceneGT`.

Two distinct levels, kept separate on purpose:

* **Datasets that ship 3-D labels.** ``nuscenes.NuScenesDataset`` reads nuScenes ground-truth boxes via
  the devkit — a real labeled :class:`~avsectester.scenarios.source.Dataset`.
* **A frame source + a labeling strategy**, for traces with no accessible 3-D labels. ``frames.ImageFolder``
  is a bare :class:`~avsectester.scenarios.source.FrameSource` (pixels only); ``detector.DetectorLabeler``
  wraps *any* frame source and *derives* ``SceneGT`` with a 2-D detector (monocular distance; viewpoint /
  visibility approximated). This is a strategy that *produces* a labeled ``Dataset`` — not a peer of
  ``NuScenesDataset``. Alpamayo/NuRec falls here: the render API exposes no actor boxes and no annotation
  files ship with the clips, so its frames must be labeled by a detector (or externally-supplied labels).
"""

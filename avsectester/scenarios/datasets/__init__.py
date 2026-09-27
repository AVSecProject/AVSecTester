"""Real-trace GT adapters — labeled datasets turned into filterable :class:`SceneGT`.

Both ship 3-D ground truth, read directly (no detector needed):

* ``nuscenes.NuScenesDataset`` — nuScenes GT boxes via the devkit.
* ``nurec.NuRecDataset`` — nuRec ``.usdz`` cuboid tracks (``sequence_tracks.json`` + the rig trajectory).
  The nre-ga *renderer* exposes only pixels, but the ``.usdz`` itself carries the actor cuboids + ego
  trajectory (scene ids ``clipgt-<uuid>``), so the labels come straight from the artifact.

Pair either with :class:`~avsectester.scenarios.source.DatasetFilter` to keep only the frames satisfying
an attack's :class:`~avsectester.scenarios.requirement.ScenarioRequirement`.
"""

# Python API reference

> **Audience:** Python experiment, attack and adapter authors.
>
> **Outcome:** Find exported names, signatures, constraints and errors.
>
> **Prerequisites:** [Architecture](../concepts/architecture.md).

## Modules

<!-- api:begin module=module-map -->
| Layer | Public module |
|---|---|
| L0 | [`avsectester`](#avsectester) |
| L0 | [`avsectester.geometry`](#avsectestergeometry) |
| L0 | [`avsectester.geometry.boxes`](#avsectestergeometryboxes) |
| L0 | [`avsectester.geometry.cameras`](#avsectestergeometrycameras) |
| L0 | [`avsectester.geometry.mesh`](#avsectestergeometrymesh) |
| L0 | [`avsectester.geometry.poses`](#avsectestergeometryposes) |
| L0 | [`avsectester.geometry.projection`](#avsectestergeometryprojection) |
| L0 | [`avsectester.serialization`](#avsectesterserialization) |
| L0 | [`avsectester.type_paths`](#avsectestertype_paths) |
| L0 | [`avsectester.validation`](#avsectestervalidation) |
| L0′ | [`avsectester.data_types`](#avsectesterdata_types) |
| L1 | [`avsectester.datasets`](#avsectesterdatasets) |
| L1 | [`avsectester.datasets.base`](#avsectesterdatasetsbase) |
| L1 | [`avsectester.datasets.carla`](#avsectesterdatasetscarla) |
| L1 | [`avsectester.datasets.nurec`](#avsectesterdatasetsnurec) |
| L1 | [`avsectester.datasets.nuscenes`](#avsectesterdatasetsnuscenes) |
| L1 | [`avsectester.maps`](#avsectestermaps) |
| L1 | [`avsectester.maps.clipgt`](#avsectestermapsclipgt) |
| L1 | [`avsectester.maps.opendrive`](#avsectestermapsopendrive) |
| L1 | [`avsectester.maps.scenario_map`](#avsectestermapsscenario_map) |
| L1 | [`avsectester.rendering`](#avsectesterrendering) |
| L1 | [`avsectester.rendering.compositing`](#avsectesterrenderingcompositing) |
| L1 | [`avsectester.rendering.harmonizers`](#avsectesterrenderingharmonizers) |
| L1 | [`avsectester.rendering.objects`](#avsectesterrenderingobjects) |
| L1 | [`avsectester.rendering.surfaces`](#avsectesterrenderingsurfaces) |
| L1 | [`avsectester.rendering.textures`](#avsectesterrenderingtextures) |
| L2 | [`avsectester.closed_loop`](#avsectesterclosed_loop) |
| L2 | [`avsectester.closed_loop.avstacks`](#avsectesterclosed_loopavstacks) |
| L2 | [`avsectester.closed_loop.avstacks.base`](#avsectesterclosed_loopavstacksbase) |
| L2 | [`avsectester.closed_loop.avstacks.driving_adapter`](#avsectesterclosed_loopavstacksdriving_adapter) |
| L2 | [`avsectester.closed_loop.avstacks.implemented`](#avsectesterclosed_loopavstacksimplemented) |
| L2 | [`avsectester.closed_loop.avstacks.implemented.alpamayo`](#avsectesterclosed_loopavstacksimplementedalpamayo) |
| L2 | [`avsectester.closed_loop.avstacks.implemented.camera_forward_collision`](#avsectesterclosed_loopavstacksimplementedcamera_forward_collision) |
| L2 | [`avsectester.closed_loop.avstacks.implemented.modular`](#avsectesterclosed_loopavstacksimplementedmodular) |
| L2 | [`avsectester.closed_loop.avstacks.implemented.rule_based`](#avsectesterclosed_loopavstacksimplementedrule_based) |
| L2 | [`avsectester.closed_loop.avstacks.implemented.rule_based.detector`](#avsectesterclosed_loopavstacksimplementedrule_baseddetector) |
| L2 | [`avsectester.closed_loop.avstacks.implemented.rule_based.planner`](#avsectesterclosed_loopavstacksimplementedrule_basedplanner) |
| L2 | [`avsectester.closed_loop.avstacks.implemented.rule_based.ranging`](#avsectesterclosed_loopavstacksimplementedrule_basedranging) |
| L2 | [`avsectester.closed_loop.avstacks.implemented.rule_based.stack`](#avsectesterclosed_loopavstacksimplementedrule_basedstack) |
| L2 | [`avsectester.closed_loop.avstacks.implemented.rule_based.tracker`](#avsectesterclosed_loopavstacksimplementedrule_basedtracker) |
| L2 | [`avsectester.closed_loop.avstacks.implemented.scorers`](#avsectesterclosed_loopavstacksimplementedscorers) |
| L2 | [`avsectester.closed_loop.environment`](#avsectesterclosed_loopenvironment) |
| L2 | [`avsectester.closed_loop.hooks`](#avsectesterclosed_loophooks) |
| L2 | [`avsectester.closed_loop.sensor_corruptions`](#avsectesterclosed_loopsensor_corruptions) |
| L2 | [`avsectester.closed_loop.simulators`](#avsectesterclosed_loopsimulators) |
| L2 | [`avsectester.closed_loop.simulators.base`](#avsectesterclosed_loopsimulatorsbase) |
| L2 | [`avsectester.closed_loop.simulators.carla`](#avsectesterclosed_loopsimulatorscarla) |
| L2 | [`avsectester.closed_loop.simulators.carla.conversions`](#avsectesterclosed_loopsimulatorscarlaconversions) |
| L2 | [`avsectester.closed_loop.simulators.carla.sensors`](#avsectesterclosed_loopsimulatorscarlasensors) |
| L2 | [`avsectester.closed_loop.simulators.carla.simulator`](#avsectesterclosed_loopsimulatorscarlasimulator) |
| L2 | [`avsectester.closed_loop.simulators.ego_dynamics`](#avsectesterclosed_loopsimulatorsego_dynamics) |
| L2 | [`avsectester.closed_loop.simulators.ground_contact`](#avsectesterclosed_loopsimulatorsground_contact) |
| L2 | [`avsectester.closed_loop.simulators.navigation`](#avsectesterclosed_loopsimulatorsnavigation) |
| L2 | [`avsectester.closed_loop.simulators.nurec`](#avsectesterclosed_loopsimulatorsnurec) |
| L2 | [`avsectester.closed_loop.simulators.nurec.renderer`](#avsectesterclosed_loopsimulatorsnurecrenderer) |
| L2 | [`avsectester.closed_loop.simulators.nurec.server`](#avsectesterclosed_loopsimulatorsnurecserver) |
| L2 | [`avsectester.closed_loop.simulators.nurec.simulator`](#avsectesterclosed_loopsimulatorsnurecsimulator) |
| L2 | [`avsectester.closed_loop.simulators.nurec.traffic`](#avsectesterclosed_loopsimulatorsnurectraffic) |
| L2 | [`avsectester.closed_loop.simulators.recorded`](#avsectesterclosed_loopsimulatorsrecorded) |
| L3 | [`avsectester.attack`](#avsectesterattack) |
| L3 | [`avsectester.attack.attack`](#avsectesterattackattack) |
| L3 | [`avsectester.attack.insertion`](#avsectesterattackinsertion) |
| L3 | [`avsectester.attack.insertion.action`](#avsectesterattackinsertionaction) |
| L3 | [`avsectester.attack.insertion.assets`](#avsectesterattackinsertionassets) |
| L3 | [`avsectester.attack.insertion.compositor`](#avsectesterattackinsertioncompositor) |
| L3 | [`avsectester.attack.insertion.placement`](#avsectesterattackinsertionplacement) |
| L3 | [`avsectester.attack.insertion.types`](#avsectesterattackinsertiontypes) |
| L3 | [`avsectester.attack.insertion.visibility`](#avsectesterattackinsertionvisibility) |
| L3 | [`avsectester.attack.offline_report`](#avsectesterattackoffline_report) |
| L3 | [`avsectester.attack.optimization`](#avsectesterattackoptimization) |
| L3 | [`avsectester.attack.optimization.blackbox`](#avsectesterattackoptimizationblackbox) |
| L3 | [`avsectester.attack.optimization.interface`](#avsectesterattackoptimizationinterface) |
| L3 | [`avsectester.attack.optimization.perturbations`](#avsectesterattackoptimizationperturbations) |
| L3 | [`avsectester.attack.optimization.scorers`](#avsectesterattackoptimizationscorers) |
| L3 | [`avsectester.attack.optimization.whitebox`](#avsectesterattackoptimizationwhitebox) |
| L3 | [`avsectester.attack.runtime`](#avsectesterattackruntime) |
| L3 | [`avsectester.attack.segments`](#avsectesterattacksegments) |
| L3 | [`avsectester.runlog`](#avsectesterrunlog) |
| L3 | [`avsectester.runlog.log`](#avsectesterrunloglog) |
| L4 | [`avsectester.execution`](#avsectesterexecution) |
| L4 | [`avsectester.execution.frame_data_cache`](#avsectesterexecutionframe_data_cache) |
| L4 | [`avsectester.execution.runner`](#avsectesterexecutionrunner) |
| L4 | [`avsectester.execution.services`](#avsectesterexecutionservices) |
| L5 | [`avsectester.evaluation`](#avsectesterevaluation) |
| L5 | [`avsectester.evaluation.evaluator`](#avsectesterevaluationevaluator) |
| L5 | [`avsectester.evaluation.metric`](#avsectesterevaluationmetric) |
| L5 | [`avsectester.evaluation.report`](#avsectesterevaluationreport) |
| L5 | [`avsectester.evaluation.visualization`](#avsectesterevaluationvisualization) |
| L6 | [`avsectester.implemented_attacks`](#avsectesterimplemented_attacks) |
| L6 | [`avsectester.implemented_attacks.object_insertion_attack`](#avsectesterimplemented_attacksobject_insertion_attack) |
| L6 | [`avsectester.implemented_attacks.object_insertion_attack.assets`](#avsectesterimplemented_attacksobject_insertion_attackassets) |
| L6 | [`avsectester.implemented_attacks.object_insertion_attack.attack`](#avsectesterimplemented_attacksobject_insertion_attackattack) |
| L6 | [`avsectester.implemented_attacks.patch_attack`](#avsectesterimplemented_attackspatch_attack) |
| L6 | [`avsectester.implemented_attacks.patch_attack.attack`](#avsectesterimplemented_attackspatch_attackattack) |
| L6 | [`avsectester.implemented_attacks.patch_attack.offline`](#avsectesterimplemented_attackspatch_attackoffline) |
| L6 | [`avsectester.implemented_attacks.patch_attack.online`](#avsectesterimplemented_attackspatch_attackonline) |
| L6 | [`avsectester.implemented_attacks.phantom_injection_attack`](#avsectesterimplemented_attacksphantom_injection_attack) |
| L6 | [`avsectester.implemented_attacks.phantom_injection_attack.attack`](#avsectesterimplemented_attacksphantom_injection_attackattack) |
| L6 | [`avsectester.implemented_attacks.speed_feedback_attack`](#avsectesterimplemented_attacksspeed_feedback_attack) |
| L6 | [`avsectester.implemented_attacks.speed_feedback_attack.attack`](#avsectesterimplemented_attacksspeed_feedback_attackattack) |
| L6 | [`avsectester.implemented_attacks.stop_sign_patch_attack`](#avsectesterimplemented_attacksstop_sign_patch_attack) |
| L6 | [`avsectester.implemented_attacks.stop_sign_patch_attack.attack`](#avsectesterimplemented_attacksstop_sign_patch_attackattack) |
| L6 | [`avsectester.implemented_attacks.stop_sign_patch_attack.metrics`](#avsectesterimplemented_attacksstop_sign_patch_attackmetrics) |
| L6 | [`avsectester.implemented_attacks.stop_sign_patch_attack.offline`](#avsectesterimplemented_attacksstop_sign_patch_attackoffline) |
| L7 | [`avsectester.experiment`](#avsectesterexperiment) |
| L7 | [`avsectester.experiment.config`](#avsectesterexperimentconfig) |
| L7 | [`avsectester.experiment.engine`](#avsectesterexperimentengine) |
| L7 | [`avsectester.experiment.html_report`](#avsectesterexperimenthtml_report) |
| L7 | [`avsectester.experiment.metadata_file`](#avsectesterexperimentmetadata_file) |
| L7 | [`avsectester.experiment.models`](#avsectesterexperimentmodels) |
| L7 | [`avsectester.experiment.result_evaluation`](#avsectesterexperimentresult_evaluation) |
| L7 | [`avsectester.experiment.type_registry`](#avsectesterexperimenttype_registry) |
| L8 | [`avsectester.cli`](#avsectestercli) |
<!-- api:end -->

## Public modules

### avsectester

<!-- api:begin module=avsectester -->
Adversarial testing for autonomous driving, organized by internal dependency layer.

| Name and signature | Meaning |
|---|---|
| `__version__` | Framework package version. |
<!-- api:end -->

### avsectester.geometry

<!-- api:begin module=avsectester.geometry -->
Camera models and geometric operations in canonical world coordinates.

| Name and signature | Meaning |
|---|---|
| `Camera` | [Camera](#avsectestergeometrycameras) |
| `CameraCalibration` | [CameraCalibration](#avsectestergeometrycameras) |
| `FThetaCamera` | [FThetaCamera](#avsectestergeometrycameras) |
| `PinholeCamera` | [PinholeCamera](#avsectestergeometrycameras) |
| `WindshieldDistortion` | [WindshieldDistortion](#avsectestergeometrycameras) |
| `MIRROR_Y` | [MIRROR_Y](#avsectestergeometryposes) |
| `box_ray_distances` | [box_ray_distances](#avsectestergeometryboxes) |
| `interpolate_transform` | [interpolate_transform](#avsectestergeometryposes) |
| `make_pose` | [make_pose](#avsectestergeometryposes) |
| `mirror_y_points` | [mirror_y_points](#avsectestergeometryposes) |
| `mirror_y_transform` | [mirror_y_transform](#avsectestergeometryposes) |
| `oriented_boxes_overlap` | [oriented_boxes_overlap](#avsectestergeometryboxes) |
| `quaternion_to_rotation` | [quaternion_to_rotation](#avsectestergeometryposes) |
| `rotation_matrix` | [rotation_matrix](#avsectestergeometryposes) |
| `rotation_to_quaternion` | [rotation_to_quaternion](#avsectestergeometryposes) |
| `transform` | [transform](#avsectestergeometryposes) |
<!-- api:end -->

### avsectester.geometry.boxes

<!-- api:begin module=avsectester.geometry.boxes -->
Ray intersections and top-down overlap for box-centered rigid transforms.

| Name and signature | Meaning |
|---|---|
| `box_ray_distances(world_from_box: ArrayLike, extent: ArrayLike, rays: np.ndarray, world_from_cam: ArrayLike) -> np.ndarray` | Ray/slab intersection distances in meters along camera-frame unit rays. |
| `oriented_boxes_overlap(world_from_box_a: ArrayLike, extent_a: ArrayLike, world_from_box_b: ArrayLike, extent_b: ArrayLike) -> bool` | Whether full-extent boxes overlap in the horizontal plane, including touching. |
<!-- api:end -->

#### Constraints and errors

Boxes use center poses and full local-axis extents. Ray misses are infinity; overlap is
evaluated in top-down geometry for the recorded states.

### avsectester.geometry.cameras

<!-- api:begin module=avsectester.geometry.cameras -->
Camera lens models shared by datasets and simulators.

| Name and signature | Meaning |
|---|---|
| `Camera(Protocol)` | Optical-frame lens contract. Coordinates address pixel centers at (x + .5, y + .5). |
| `Camera.width: int` | Image width in pixels. |
| `Camera.height: int` | Image height in pixels. |
| `Camera.project(pts_cam: np.ndarray) -> np.ndarray` | Project camera-coordinate points into continuous image coordinates. |
| `Camera.unproject(uv: np.ndarray) -> np.ndarray` | Convert continuous image coordinates into unit optical-camera rays. |
| `CameraCalibration(camera: Camera, rig_from_camera: np.ndarray)` | A lens and optical-camera-to-rig transform, in meters. |
| `camera_ray_mask(camera: Camera, rays: np.ndarray) -> np.ndarray` | Finite forward camera rays inside the lens's optional maximum viewing angle. |
| `PinholeCamera(K: np.ndarray, width: int, height: int)` | Ideal pinhole camera with 3x3 intrinsics `K`. |
| `PinholeCamera.project(pts_cam: np.ndarray) -> np.ndarray` | (N,3) camera-frame points -> (N,2) pixels (valid only for z > 0). |
| `PinholeCamera.unproject(uv: np.ndarray) -> np.ndarray` | (N,2) pixels -> (N,3) unit rays in the camera frame. |
| `WindshieldDistortion(horizontal_poly: tuple, vertical_poly: tuple, horizontal_poly_inverse: tuple, vertical_poly_inverse: tuple)` | Refraction of camera rays by the vehicle windshield, as in NVIDIA ncore's bivariate model. |
| `WindshieldDistortion.distort(rays: np.ndarray) -> np.ndarray` | Scene-side rays to the lens-side rays that the f-theta lens projects. |
| `WindshieldDistortion.undistort(rays: np.ndarray) -> np.ndarray` | Lens-side rays to the scene-side rays they observe. |
| `FThetaCamera(cx: float, cy: float, angle_to_pixeldist: tuple, width: int, height: int, pixeldist_to_angle: tuple = (), max_angle: float = math.pi / 2, linear_cde: tuple = (1.0, 0.0, 0.0), windshield: WindshieldDistortion \| None = None)` | NVIDIA f-theta fisheye: `r_px = angle_to_pixeldist(theta)` with `theta` the ray's angle to the optical axis, `r_px` its pixel distance from the principal point (`cx`, `cy`). |
| `FThetaCamera.project(pts_cam: np.ndarray) -> np.ndarray` | Project camera-coordinate points into continuous image coordinates. |
| `FThetaCamera.in_view(pts_cam: np.ndarray) -> np.ndarray` | Which camera-frame points project: in front of the camera and within `max_angle`. |
| `FThetaCamera.unproject(uv: np.ndarray) -> np.ndarray` | Convert continuous image coordinates into unit optical-camera rays. |
| `BODY_FROM_OPTICAL` | Map optical points (x right, y down, z forward) to body axes (x forward, y left, z up). |
<!-- api:end -->

#### Constraints and errors

`Camera` axes are optical: x right, y down, z forward. Pixel centers are (x+0.5, y+0.5).
project/unproject accept N×3 points and N×2 pixels. `BODY_FROM_OPTICAL` maps to body axes (x
forward, y left, z up). `CameraCalibration` does not validate array payloads at construction;
producers and consumers validate transforms. `FThetaCamera.unproject` raises
`NotImplementedError` without `pixeldist_to_angle` and uses that inverse polynomial directly
rather than solving the forward polynomial. Screen correction must be invertible (`c - d * e
!= 0`).

### avsectester.geometry.mesh

<!-- api:begin module=avsectester.geometry.mesh -->
CPU triangle-mesh ray queries requiring Embree through the optional nurec extra.

| Name and signature | Meaning |
|---|---|
| `MeshRaycaster(vertices: np.ndarray, faces: np.ndarray)` | An immutable triangle mesh accelerated by Embree. |
| `MeshRaycaster.check_dependencies() -> None` | Fail early when the optional CPU mesh-query dependencies are unavailable. |
| `MeshRaycaster.distances(origins: np.ndarray, directions: np.ndarray) -> np.ndarray` | First-hit distances for paired `(N, 3)` origins and directions. |
| `camera_depth(camera: Camera, cam_from_world: np.ndarray, raycasters: Sequence[MeshRaycaster]) -> np.ndarray` | Optical-axis z depth at pixel centers, taking the nearest of all supplied meshes. |
<!-- api:end -->

#### Constraints and errors

Mesh vertices are N×3 and faces are indexed triangles. Ray distances are metric and misses are
infinity. `camera_depth` returns optical-axis depth. Embree is an optional
mesh-query dependency.

`MeshRaycaster.check_dependencies` raises `ImportError` when Embree dependencies
are unavailable.

### avsectester.geometry.poses

<!-- api:begin module=avsectester.geometry.poses -->
Canonical world/actor poses in meters and right-handed x-forward, y-left, z-up axes.

| Name and signature | Meaning |
|---|---|
| `MIRROR_Y` | Homogeneous transform that reverses the Y coordinate. |
| `validated_vector(value: ArrayLike, name: str) -> np.ndarray` | Return a finite three-dimensional vector, rejecting invalid shapes or values. |
| `validated_transform(value: ArrayLike) -> np.ndarray` | Return an independent, read-only right-handed rigid transform. |
| `rotation_matrix(rotation_deg: ArrayLike) -> np.ndarray` | Return Rz(yaw) @ Ry(pitch) @ Rx(roll) for right-handed degree angles. |
| `mirror_y_points(points: ArrayLike) -> np.ndarray` | Mirror the y component of points or vectors with trailing coordinate dimension three. |
| `mirror_y_transform(matrix: ArrayLike) -> np.ndarray` | Express a homogeneous transform in the opposite y-axis convention on both sides. |
| `quaternion_to_rotation(w: float, x: float, y: float, z: float) -> np.ndarray` | Return a rotation matrix from a finite, nonzero quaternion in w/x/y/z order. |
| `rotation_to_quaternion(rotation: ArrayLike) -> tuple[float, float, float, float]` | Return a normalized w/x/y/z quaternion, including for rotations close to pi. |
| `interpolate_transform(first: ArrayLike, second: ArrayLike, fraction: float) -> np.ndarray` | Interpolate rigid poses with linear translation and shortest-path quaternion slerp. |
| `make_pose(rotation: np.ndarray, translation: ArrayLike) -> np.ndarray` | Build a homogeneous transform from a rotation matrix and translation vector. |
| `transform(T: np.ndarray, pts: np.ndarray) -> np.ndarray` | Transform an array of three-dimensional points into the destination frame. |
<!-- api:end -->

#### Constraints and errors

Rigid transforms are 4×4 matrices. Quaternion order is (w, x, y, z). `validated_transform`
rejects non-rigid rotations and malformed homogeneous rows.

### avsectester.geometry.projection

<!-- api:begin module=avsectester.geometry.projection -->
Lens-aware projected bounds of boxes, using only geometric data.

| Name and signature | Meaning |
|---|---|
| `project_box(camera: Camera, cam_from_world: np.ndarray, world_from_box: np.ndarray, extent: tuple[float, float, float] \| np.ndarray) -> tuple[float, float, float, float] \| None` | Project a box-centered rigid pose and full dimensions to an image-clipped rectangle. |
<!-- api:end -->

#### Constraints and errors

`project_box` returns visible image-clipped xyxy bounds of a 3-D box, including curved projected
edges. It returns None when no projection intersects the image; extents must be nonnegative.

### avsectester.serialization

<!-- api:begin module=avsectester.serialization -->
Shared JSON-compatible serialization and independent in-memory snapshots.

| Name and signature | Meaning |
|---|---|
| `SerializationRegistry()` | Explicit dataclass reconstruction and data-only native encoding. |
| `SerializationRegistry.register_dataclass(cls_or_path: type \| str) -> None` | Register a dataclass, or a path imported only when that dataclass is decoded. |
| `SerializationRegistry.register_native(module: str, name: str, encode: Callable[[Any], Any], *, snapshot: Callable[[Any, Callable[[Any], Any]], Any] \| None = None) -> None` | Register native conversion without importing the native module. |
| `SerializationRegistry.snapshot(value: Any) -> Any` | Take an independent snapshot, preserving native wrappers and container subclasses. |
| `SerializationRegistry.encode(value: Any, *, encode_child: Callable[[Any], Any] \| None = None, encode_field: Callable[[str, Any], Any] \| None = None, encode_leaf: Callable[[Any], Any] \| None = None) -> Any` | Convert a registered value into tagged JSON-compatible data. |
| `SerializationRegistry.decode(data: Any, *, decode_child: Callable[[Any], Any] \| None = None) -> Any` | Reconstruct registered dataclasses; encoded native values remain ordinary data. |
| `DEFAULT_REGISTRY` | Serialization registry with framework record types and array codecs registered. |
<!-- api:end -->

#### Constraints and errors

Register custom dataclasses before writing and import their defining/registration modules before
reading. Snapshots are independent copies.

#### See also

[Runtime reference](runtime.md).

### avsectester.type_paths

<!-- api:begin module=avsectester.type_paths -->
Resolve callable Python objects from explicit module-and-attribute paths.

| Name and signature | Meaning |
|---|---|
| `split_type_path(value: str) -> tuple[str, str]` | Validate `package.module:name` without importing it and return its two parts. |
| `import_type_path(value: str) -> Any` | Import a callable `package.module:name`, following dotted attributes in `name`. |
<!-- api:end -->

#### Constraints and errors

Paths use `package.module:Name`. Malformed paths raise `ValueError`; non-callable targets
raise `TypeError`. Missing modules and attributes retain Python's import and attribute errors.

#### See also

[Configuration reference](configuration.md).

### avsectester.validation

<!-- api:begin module=avsectester.validation -->
Reusable value checks for framework components and user implementations.

| Name and signature | Meaning |
|---|---|
| `require_nonnegative_integer(value: int, name: str) -> None` | Require an integer greater than or equal to zero, excluding booleans. |
| `require_positive_integer(value: int, name: str) -> None` | Require an integer greater than zero, excluding booleans, without converting it. |
| `require_nonnegative_number(value: float, name: str) -> None` | Require a finite real number greater than or equal to zero, excluding booleans. |
| `require_positive_number(value: float, name: str) -> None` | Require a finite real number greater than zero, excluding booleans. |
<!-- api:end -->

#### Constraints and errors

Integers exclude bool. Numeric validators reject non-finite values; positive and nonnegative
bounds are checked separately.

Validation helpers raise `ValueError` naming the invalid parameter.

### avsectester.data_types

<!-- api:begin module=avsectester.data_types -->
Data types exchanged between framework components.

| Name and signature | Meaning |
|---|---|
| `ActorState(pose: Any, velocity: tuple[float, float, float] \| None = None, category: str \| None = None, extent: tuple[float, float, float] \| None = None, box_center: tuple[float, float, float] = (0.0, 0.0, 0.0), box_rotation: tuple[float, float, float] = (0.0, 0.0, 0.0))` | An actor's world-from-origin pose, world velocity and optional bounding-box geometry. |
| `ActorState.box_transform() -> np.ndarray` | World-from-box-center transform, including local roll/pitch/yaw degree rotation. |
| `Unavailable(reason: str)` | A dataset cannot supply the requested evidence, with an explicit reason. |
| `TargetSpec(box: tuple[float, float, float, float], label: str \| None = None)` | The object to attack, with an image-space xyxy `box` and an optional class label. |
| `Control(throttle: float = 0.0, steer: float = 0.0, brake: float = 0.0, trajectory: Any = None)` | The driving system's output — a vehicle control *command* (the decision). |
| `Observation(t: float, frame: int, sensor_data: dict[str, Any] = dict(), calibration: dict[str, Any] = dict(), vehicle_state: Any = None, ego_speed: float = 0.0, route: np.ndarray \| None = None)` | What the simulator sends the driving system each frame — and the only thing it sends. |
| `Observation.set_calibration(sensor_id: str, calibration: Any) -> None` | Replace model-visible calibration and its native payload reference together. |
| `WorldSnapshot(t: float, frame: int, vehicle_state: Any = None, ego_speed: float = 0.0, actors: Mapping[str, Any] = dict(), ego: ActorState \| None = None, traffic_lights: Mapping[str, str] = dict(), objects: Mapping[str, ActorState] = dict())` | An independent snapshot of the simulator's actual ego state and other actors' states. |
| `StateEstimate(vehicle_state: Any = None, ego_speed: float = 0.0)` | Model-visible ego state and speed produced by a localization adapter. |
| `RecordedOutput(name: str, value: Any, step: int, stage: str, frame: int, t: float, sensor: str \| None = None)` | A named output captured by an attack, defense or localization callback. |
| `Detection(position: tuple[float, float, float], extent: tuple[float, float, float], yaw: float, category: str \| None = None, score: float \| None = None)` | One 3D detection in the capture-time true ego frame (x forward, y left, z up). |
| `validate_command_modes(command: Control, modes: frozenset[str] \| None) -> None` | Validate the fields consumed by a simulator or independently called dynamics model. |
| `read_camera_rgb(observation: Observation, camera: str \| None = None) -> tuple[str, np.ndarray] \| None` | Return the selected sensor ID and H×W×3 uint8 RGB pixels, or `None` if unavailable. |
| `write_camera_rgb(observation: Observation, key: str, rgb: np.ndarray) -> Observation` | Return an observation copy with sensor `key` replaced by pixels cast to uint8. |
| `write_sensor_data(observation: Observation, sensor: str, data: Any) -> Observation` | Return an observation copy with one sensor's data replaced. |
<!-- api:end -->

#### Constraints and errors

`ActorState.pose` is world-from-actor. `extent` is full length/width/height in meters or `None`;
zero dimensions are distinct from unknown dimensions. `box_center` is local; `box_rotation` is
local right-handed roll/pitch/yaw in degrees. `box_transform()` composes the world-from-box pose.
`write_camera_rgb` casts input to RGB uint8; generic sensor writes preserve transport conventions.

#### See also

[Runtime reference](runtime.md).

### avsectester.datasets

<!-- api:begin module=avsectester.datasets -->
Stateless recorded-data and recipe providers; no simulator ownership.

| Name and signature | Meaning |
|---|---|
| `SensorDataReference` | [SensorDataReference](#avsectesterdatasetsbase) |
| `Dataset` | [Dataset](#avsectesterdatasetsbase) |
| `FrameReference` | [FrameReference](#avsectesterdatasetsbase) |
| `FrameData` | [FrameData](#avsectesterdatasetsbase) |
| `ScenarioDescriptor` | [ScenarioDescriptor](#avsectesterdatasetsbase) |
| `ScenarioStart` | [ScenarioStart](#avsectesterdatasetsbase) |
<!-- api:end -->

#### See also

[Datasets and semantic maps](../guides/datasets.md).

### avsectester.datasets.base

<!-- api:begin module=avsectester.datasets.base -->
Dataset interfaces, frame references and recorded scenario starts.

| Name and signature | Meaning |
|---|---|
| `FrameReference(scenario_id: str, position: int, frame_id: str \| int, timestamp: float)` | One frame of a scenario's timeline: its position, native frame ID and scenario time (s). |
| `ScenarioDescriptor(id: str, kind: str, payload: Mapping[str, Any] = dict())` | A scenario's `id`, dataset class name in `kind`, and adapter-specific `payload`. |
| `SensorDataReference(uri: str, sensor: str, timestamp: float, metadata: Mapping[str, Any] = dict())` | Location of one recorded sensor payload; adapters decide how it is decoded. |
| `FrameData(frame: FrameReference, actors: Mapping[str, data_types.ActorState] = dict(), sensor_references: Mapping[str, SensorDataReference] = dict(), provenance: Mapping[str, Any] = dict(), ego: data_types.ActorState \| None = None)` | World-frame actor states and recorded sensor references at one dataset position. |
| `ScenarioStart(kind: str, scenario_id: str, frame: FrameReference, payload: Mapping[str, Any], recorded_ego: tuple[data_types.ActorState, ...] = (), scenario_map: ScenarioMap \| data_types.Unavailable = data_types.Unavailable("The dataset provides no scenario map"))` | What a simulator needs to start a scenario at a frame; no attack or selection metadata. |
| `Dataset(ABC)()` | Read scenarios with identical behavior before and after simulator execution. |
| `Dataset.scenarios() -> Iterator[ScenarioDescriptor]` | Abstract. Enumerate available scenarios in dataset order. |
| `Dataset.frames(scenario_id: str) -> Sequence[FrameReference]` | Abstract. Return the ordered frame references for a scenario. |
| `Dataset.frame_data(scenario_id: str, frame: FrameReference) -> FrameData \| data_types.Unavailable` | Abstract. Load frame data for the requested scenario reference. |
| `Dataset.native(scenario_id: str) -> Any` | Abstract. Return the native record for `scenario_id`. |
| `Dataset.scenario_start(scenario_id: str, frame: FrameReference) -> ScenarioStart` | Abstract. Build the recorded ego, map and simulator payload for a scenario reference. |
| `Dataset.scenario_map(scenario_id: str) -> ScenarioMap \| data_types.Unavailable` | Return immutable semantic map data, or explain why this dataset has none. |
| `Dataset.checked_frame(scenario_id: str, frame: FrameReference) -> FrameReference` | Validate a frame reference against the scenario timeline and return the canonical reference. |
| `Dataset.decision_window(scenario_id: str, start: FrameReference, decision_steps: int) -> tuple[FrameReference, ...]` | Require N+1 actual frames (the last is the terminal state); never pad or truncate. |
<!-- api:end -->

#### Constraints and errors

N decisions require N+1 supported frame references. `FrameReference.position` is a canonical
timeline position, separate from native `frame_id`. `Dataset` is stateless; unavailable source
values are explicit. `FrameData.actors` excludes ego.

#### See also

[Datasets and semantic maps](../guides/datasets.md).

### avsectester.datasets.carla

<!-- api:begin module=avsectester.datasets.carla -->
CARLA scenario recipes and semantic-map-backed dataset access.

| Name and signature | Meaning |
|---|---|
| `PrescribedMotion(controls: tuple[Control, ...] = (), trajectory: tuple[ActorState, ...] = (), constant_control: Control \| None = None)` | One control sequence, trajectory, or constant actuator control. |
| `PrescribedMotion.expanded(decision_steps: int) -> PrescribedMotion` | Expand `constant_control` into one control per decision and validate the motion. |
| `PrescribedMotion.validate(decision_steps: int) -> None` | Require N actuator controls or N+1 states with velocity for N=`decision_steps`. |
| `CarlaRecipe(scenario: dict, decision_steps: int, time_step: float, motion: Mapping[str, PrescribedMotion], metadata: Mapping = dict())` | Serializable CARLA world, ego and recorded-actor scenario recipe. |
| `CarlaDataset(Dataset)(recipes: Mapping[str, CarlaRecipe], map_directory: str \| Path \| None = None)` | Dataset of CARLA recipes and optional exported semantic maps. |
| `CarlaDataset.scenario_map(scenario_id: str) -> ScenarioMap \| Unavailable` | Read an exported OpenDRIVE map without connecting to a CARLA server. |
<!-- api:end -->

#### Constraints and errors

`CarlaRecipe` raises `ValueError` for zero decisions or nonpositive/nonfinite `time_step`.
Every prescribed control sequence needs N controls; trajectories need N+1 states with velocity.
Generated frame data is owned by preparation caches rather than `Dataset`.

#### See also

[Datasets and semantic maps](../guides/datasets.md).

### avsectester.datasets.nurec

<!-- api:begin module=avsectester.datasets.nurec -->
Read NuRec USDZ annotations and recorded initial sequences without a rendering service.

| Name and signature | Meaning |
|---|---|
| `read_scene_mesh(usdz_path: str \| Path) -> tuple[np.ndarray, np.ndarray]` | Merge mesh.ply and mesh_ground.ply without loading camera or trajectory metadata. |
| `NuRecDataset(Dataset)(usdz_paths: list[str \| Path] \| None = None, camera_id: str = "camera_front_wide_120fov", vehicle_labels: tuple[str, ...] = _VEHICLE_LABELS, time_step: float = 0.1, *, dataroot: str \| Path \| None = None)` | Labeled NuRec scenes on a decision timeline generated at `time_step`. |
| `NuRecDataset.usdz_paths: list[str]` | Scenario archive paths: the configured list, or every archive below dataroot. |
| `NuRecDataset.open_archive(path: str \| Path) -> zipfile.ZipFile` | Open the scene archive read-only. The caller closes it, preferably with `with`. |
| `NuRecDataset.raw_metadata(scenario_id: str) -> dict[str, Any]` | Return an independent copy of this scenario's native annotations. |
| `NuRecDataset.camera_geometry(scenario_id: str)` | Return an independent optical camera model and its rig-from-camera pose. |
| `NuRecDataset.camera_timestamps(scenario_id: str) -> np.ndarray` | Recorded video-frame timestamps in seconds, distinct from decision positions. |
| `NuRecDataset.usdz_path(scenario_id: str) -> str` | Return this scenario's source archive path without creating a simulator. |
| `NuRecDataset.scene_mesh(scenario_id: str) -> tuple[np.ndarray, np.ndarray]` | Merge the static and ground PLY meshes in canonical scene-world coordinates. |
| `NuRecDataset.scenario_map(scenario_id: str) -> ScenarioMap \| Unavailable` | Read canonical-world clipgt geometry, preserving missing-layer reasons. |
| `NuRecDataset.ego_state(scenario_id: str, timestamp_us: int) -> ActorState` | Interpolate the recorded rig pose and world velocity within its support. |
| `NuRecDataset.track_trajectories(scenario_id: str) -> dict[str, dict[str, Any]]` | Return independent complete box-center tracks without smoothing or cropping. |
| `NuRecDataset.actor_states(scenario_id: str, timestamp_us: int) -> dict[str, ActorState]` | Recorded actors at this timestamp: world pose, velocity and category, by track ID. |
<!-- api:end -->

#### Constraints and errors

Exactly one of `usdz_paths` and `dataroot` must be supplied. Timelines interpolate only within
recorded support. `camera_id` selects the camera whose timestamps define the decision timeline.
`read_scene_mesh` returns world-coordinate vertices and indexed triangle faces. `clipgt` shares
the recorded rig world frame. An OpenDRIVE fallback requires a confirmed transform into that
frame; geographic reference alone is insufficient and unconfirmed alignment
returns `Unavailable`.

Each `track_trajectories` entry contains `trajectory` as timestamp-microsecond/world-from-box
pairs, boolean `controllable`, native `label_class`, full `extent` and canonical `category`.

#### See also

[Datasets and semantic maps](../guides/datasets.md).

### avsectester.datasets.nuscenes

<!-- api:begin module=avsectester.datasets.nuscenes -->
Read nuScenes records without simulator dependencies.

| Name and signature | Meaning |
|---|---|
| `NuScenesDataset(Dataset)(dataroot: str \| Path, version: str = "v1.0-trainval", camera: str = "CAM_FRONT")` | nuScenes scenes as scenario timelines of linked keyframes. |
<!-- api:end -->

#### Constraints and errors

The nuScenes SDK and requested dataset version must be available. Sensor names remain native
nuScenes names.

#### See also

[Datasets and semantic maps](../guides/datasets.md).

### avsectester.maps

<!-- api:begin module=avsectester.maps -->
Semantic scenario maps and offline source readers.

| Name and signature | Meaning |
|---|---|
| `Lane` | [Lane](#avsectestermapsscenario_map) |
| `ScenarioMap` | [ScenarioMap](#avsectestermapsscenario_map) |
| `STOP_LINE_KIND_BY_CONTROL_KIND` | [STOP_LINE_KIND_BY_CONTROL_KIND](#avsectestermapsscenario_map) |
| `StopLine` | [StopLine](#avsectestermapsscenario_map) |
| `TrafficControl` | [TrafficControl](#avsectestermapsscenario_map) |
| `read_opendrive` | [read_opendrive](#avsectestermapsopendrive) |
<!-- api:end -->

#### See also

[Datasets and semantic maps](../guides/datasets.md).

### avsectester.maps.clipgt

<!-- api:begin module=avsectester.maps.clipgt -->
Read NuRec clipgt semantic geometry without dataset or simulator dependencies.

| Name and signature | Meaning |
|---|---|
| `read_clipgt(archive: zipfile.ZipFile) -> ScenarioMap` | Read canonical semantic layers from clipgt parquet tables in an open scene archive. |
<!-- api:end -->

#### Constraints and errors

Read an open USDZ ZIP archive. `Lane` rails remain recorded boundaries; centerline construction
resamples both at matching normalized arc-length fractions, using the larger point count, then
takes paired midpoints. Connectivity and control/line associations require source records.
`Lane` classification applies in order: `SHOULDER_LANE` →shoulder; `PARKING_LANE` →parking;
`vehicle_types` containing CAR→driving; only BICYCLE→bike; otherwise other. `HOV_LANE` and
`SERVICE_ROAD` with CAR remain driving. `TrafficControl` poses retain raw centers and XYZ dimensions
with `face_center`. Exact traffic-sign categories `TRAFFIC_SIGN_REGULATORY_R1_STOP`,
`TRAFFIC_SIGN_REGULATORY_R1_YIELD` and `TRAFFIC_SIGN_REGULATORY_R2_SPEED_LIMIT` map to
`stop_sign`, `yield_sign` and `speed_limit`; other categories remain other and retain their
source `native_category`. Only wait-line rows explicitly marked `is_implicit=False` are
retained. All-implicit rows mean a known empty `stop_lines` layer. `native_category` keeps paint
labels such as STOP; subtypes ENTRY, `NOT_APPLICABLE` and `CROSSWALK_ENTRY` are not categories
or filters. Kind is derived only from explicit control/lane associations, never paint or nearest
geometry. Multiple controls of one recognized kind may establish kind; `control_id` requires one
unique control. Missing/ambiguous evidence gives unknown. Missing or unusable geometry is
reported in `unavailable_layers`. An empty or unsupported traffic-light table does not make the
whole mixed `traffic_controls` layer unavailable. `read_clipgt` returns the archive coordinates
without applying a coordinate transform.

#### See also

[Datasets and semantic maps](../guides/datasets.md).

### avsectester.maps.opendrive

<!-- api:begin module=avsectester.maps.opendrive -->
Read canonical semantic layers through CARLA's offline OpenDRIVE client parser.

| Name and signature | Meaning |
|---|---|
| `read_opendrive(xodr_text: str, *, name: str) -> ScenarioMap \| Unavailable` | Read OpenDRIVE without a simulator connection, or report a missing CARLA client package. |
<!-- api:end -->

#### Constraints and errors

The optional CARLA parser reads exported OpenDRIVE without a server. Canonical conversion
reflects the Y axis. Every lane type is sampled at spacing at most two meters on the center and
both boundaries. Native midpoint refinement handles continuous geometry; unresolved source
discontinuities subdivide the existing straight connector while preserving its endpoints.
Left/right neighbours refer only to the same travel direction. Driving-lane polygons form the
drivable area. A junction intersection is the convex hull of its driving lanes. Landmark types
206, 205, 274 and 1000001 map to `stop_sign`, `yield_sign`, `speed_limit` and `traffic_light`.
Each signal reference resolves its own road and governed lanes. A stop line spans each
controlled lane perpendicular to its local heading, with ID
`stop-line:<signal_id>:<road_id>:<lane_id>`. `TrafficControl.lane_ids` unions those references;
`stop_line_id` is set only for one line; otherwise follow `StopLine.control_id`. Generated lines
approximate reference positions, not surveyed painted markings; `native_category` is
`opendrive_signal_reference`. Positive-height non-light landmarks shift up half their height and
use `face_center`; lights and zero-height landmarks retain `signal_position`. `Orientation`
opposes the mean normalized governed-lane direction when its horizontal magnitude is at least
0.5, otherwise uses the converted landmark heading plus 180° local Z. Extent is
(0,width,height); unavailable thickness and unconfirmed speed limits are reported explicitly.

#### See also

[Datasets and semantic maps](../guides/datasets.md).

### avsectester.maps.scenario_map

<!-- api:begin module=avsectester.maps.scenario_map -->
Immutable semantic map records in canonical world coordinates, measured in meters.

| Name and signature | Meaning |
|---|---|
| `STOP_LINE_KIND_BY_CONTROL_KIND` | Mapping from traffic-control kind to the corresponding stop-line kind. |
| `Lane(id: str, centerline: np.ndarray, left_boundary: np.ndarray, right_boundary: np.ndarray, lane_type: Literal["driving", "shoulder", "bike", "parking", "other"] = "driving", predecessors: tuple[str, ...] = (), successors: tuple[str, ...] = (), left_neighbor: str \| None = None, right_neighbor: str \| None = None, speed_limit_mps: float \| None = None)` | One lane with direction-ordered centerline and left/right boundary polylines. |
| `TrafficControl(id: str, kind: Literal["stop_sign", "yield_sign", "speed_limit", "traffic_light", "other"], native_category: str, pose: np.ndarray, extent: tuple[float, float, float], pose_reference: Literal["face_center", "signal_position"], lane_ids: tuple[str, ...] = (), stop_line_id: str \| None = None)` | A static control facing approaching traffic, with (thickness, width, height) extent. |
| `StopLine(id: str, points: np.ndarray, native_category: str, kind: Literal["stop_sign", "traffic_light", "yield_sign", "unknown"] = "unknown", lane_ids: tuple[str, ...] = (), control_id: str \| None = None)` | A world-coordinate transverse line associated with lanes and optionally a control. |
| `ScenarioMap(lanes: Mapping[str, Lane] = dict(), drivable_area: tuple[np.ndarray, ...] = (), road_boundaries: tuple[np.ndarray, ...] = (), crosswalks: tuple[np.ndarray, ...] = (), intersections: tuple[np.ndarray, ...] = (), traffic_controls: Mapping[str, TrafficControl] = dict(), stop_lines: Mapping[str, StopLine] = dict(), unavailable_layers: Mapping[str, str] = dict())` | Semantic layers in canonical right-handed world axes, with explicit missing-layer reasons. |
<!-- api:end -->

#### Constraints and errors

All four records are frozen; geometry inputs are copied and made read-only in canonical world
meters. `ScenarioMap` contains lanes, drivable polygons, road boundaries, crosswalks,
intersections, traffic controls, stop lines and `unavailable_layers`. `Lane` geometry is N×3 and
speed limits are m/s; `StopLine.points` is N×3. `TrafficControl.pose` has +X pointing out of its
front face toward approaching traffic. extent is full (thickness, width, height) on local X/Y/Z.
`pose_reference` is required:

| Source | `pose_reference` | Origin and extent |
|---|---|---|
| NuRec signs/lights | `face_center` | Recorded box center and XYZ dimensions |
| CARLA positive-height non-light landmark | `face_center` | Signal position plus half its height in world Z; thickness is unavailable |
| CARLA traffic light or zero-height landmark | `signal_position` | OpenDRIVE signal position; a light head or face size is not established |

Only `face_center` puts the extent center at the pose origin. `Lane` kinds are driving,
shoulder, bike, parking or other. `TrafficControl` kinds are `stop_sign`, `yield_sign`, `speed_limit`,
`traffic_light` or other; stop-line kinds are `stop_sign`, `traffic_light`, `yield_sign` or
unknown. `STOP_LINE_KIND_BY_CONTROL_KIND` maps recognized stopping-control associations.
`native_category` retains source text independently. `unavailable_layers` keys are a layer,
`<layer>.partial`, `<layer>.<record field>`, or `traffic_controls.<kind>`. They identify absent
layers, partially readable records, absent fields or missing control kinds. Other keys raise
`ValueError`; missing evidence is never filled with invented associations.

#### See also

[Datasets and semantic maps](../guides/datasets.md).

### avsectester.rendering

<!-- api:begin module=avsectester.rendering -->
Simulator-independent world objects, planar rendering and appearance compositing.

| Name and signature | Meaning |
|---|---|
| `ObjectAsset` | [ObjectAsset](#avsectesterrenderingobjects) |
| `PlacedObject` | [PlacedObject](#avsectesterrenderingobjects) |
| `PlaneSurface` | [PlaneSurface](#avsectesterrenderingobjects) |
| `ClassicHarmonizer` | [ClassicHarmonizer](#avsectesterrenderingharmonizers) |
| `ExposureHarmonizer` | [ExposureHarmonizer](#avsectesterrenderingharmonizers) |
| `Harmonizer` | [Harmonizer](#avsectesterrenderingharmonizers) |
| `PCTNetHarmonizer` | [PCTNetHarmonizer](#avsectesterrenderingharmonizers) |
| `render_surfaces` | [render_surfaces](#avsectesterrenderingsurfaces) |
| `CompositePlan` | [CompositePlan](#avsectesterrenderingcompositing) |
| `CompositeResult` | [CompositeResult](#avsectesterrenderingcompositing) |
| `ObjectComposition` | [ObjectComposition](#avsectesterrenderingcompositing) |
| `ObjectVisibility` | [ObjectVisibility](#avsectesterrenderingcompositing) |
| `blend_rgba` | [blend_rgba](#avsectesterrenderingcompositing) |
| `composite_objects` | [composite_objects](#avsectesterrenderingcompositing) |
| `plan_composite` | [plan_composite](#avsectesterrenderingcompositing) |
| `STOP_SIGN_FACE` | [STOP_SIGN_FACE](#avsectesterrenderingtextures) |
| `CheckerboardTexture` | [CheckerboardTexture](#avsectesterrenderingtextures) |
| `Texture` | [Texture](#avsectesterrenderingtextures) |
| `load_rgba_image` | [load_rgba_image](#avsectesterrenderingtextures) |
<!-- api:end -->

#### See also

[Implement an attack](../guides/attacks.md).

### avsectester.rendering.compositing

<!-- api:begin module=avsectester.rendering.compositing -->
Depth-aware composition of world objects, differentiable in surface colors, and
visible-region blending.

| Name and signature | Meaning |
|---|---|
| `SurfaceFootprint(object_index: int, surface_index: int, sampling: PlaneSampling, alpha: np.ndarray, premultiplied: np.ndarray, owned: np.ndarray)` | One surface's sampling over its clipped image region and the visible pixels it owns. |
| `ObjectVisibility(occlusion_source: str, ignored_occluders: tuple[str, ...], visible_pixels: int)` | One world object's camera visibility and the evidence used to measure it. |
| `CompositeResult(image: np.ndarray, depth: np.ndarray, visible_mask: np.ndarray, occlusion_source: str, ignored_occluders: tuple[str, ...] = (), object_visibility: Mapping[str, ObjectVisibility] = dict())` | Composite pixels, optical z-depth, visibility, occlusion source and ignored_occluders. |
| `blend_rgba(image: np.ndarray, rgba: np.ndarray, visible_mask: np.ndarray, *, harmonizer: Harmonizer \| None = None, soften: float = 0.0) -> np.ndarray` | Blend sampled RGBA only at visible pixels, optionally softening and harmonizing there. |
| `CompositePlan(objects: tuple[PlacedObject, ...], depth: np.ndarray, visible_mask: np.ndarray, occlusion_source: str, ignored_occluders: tuple[str, ...], object_visibility: Mapping[str, ObjectVisibility], footprints: tuple[SurfaceFootprint, ...])` | Texture-independent geometry and composition for one camera image. |
| `CompositePlan.composite(image: torch.Tensor, textures: Mapping[tuple[str, int], torch.Tensor] \| None = None) -> torch.Tensor` | Blend surfaces and differentiable harmonizers over a (3, H, W) image in [0, 1]. |
| `CompositePlan.apply(image: np.ndarray) -> CompositeResult` | Deploy uint8 RGB, rounding once after differentiable composition. |
| `ObjectComposition(background: np.ndarray, plan: CompositePlan)` | How placed objects were composed into one capture: the image beneath them and the plan. |
| `plan_composite(camera: Camera, cam_from_world: np.ndarray, objects: Sequence[PlacedObject], *, scene_depth: np.ndarray \| None = None, occluders: Mapping[str, ActorState] \| None = None, occlusion_tolerance_m: float = 0.1, scene_visibility: np.ndarray \| None = None) -> CompositePlan` | Plan the nearest object surfaces and their unoccluded pixels in one camera. |
| `composite_objects(image: np.ndarray, camera: Camera, cam_from_world: np.ndarray, objects: Sequence[PlacedObject], *, scene_depth: np.ndarray \| None = None, occluders: Mapping[str, ActorState] \| None = None, occlusion_tolerance_m: float = 0.1) -> CompositeResult` | Plan and apply a composite; see `plan_composite` and `CompositePlan.apply`. |
<!-- api:end -->

#### Constraints and errors

`CompositePlan` shares texture sampling, alpha, visibility and differentiable harmonization
between deployment and optimization; `apply(image)` equals `round(composite(image/255) * 255)`
when all harmonizers are differentiable. `composite` raises `ValueError` for a
non-differentiable harmonizer, identifying its object.

Each object's harmonizer sees the same unadjusted composite, original background, its visible
mask and a nominal composite. Surface `nominal_texture` values replace candidate colors in that
nominal view, so exposure estimation can retain the unpatched face as reference.

`scene_visibility` is the sole occlusion evidence when provided, excluding scene depth and
occluders; the plan depth then contains only visible object surfaces. Otherwise supplied scene
optical-axis depth is the sole scene-occlusion evidence; without depth, known actor cuboids
approximate occlusion in candidate regions. Missing actor extents are listed as
`ignored_occluders`. The nearest surface wins; surface order breaks exact depth ties.

Surfaces without `nominal_texture` keep their own candidate colors in the nominal blend.
Non-differentiable harmonizers run through NumPy after uint8 rounding during `apply`; `composite`
rejects them. Both differentiable paths round only once at final uint8 conversion.

`CompositePlan.composite` accepts replacement colors keyed by `(object_id, surface_index)`; an
unknown planned surface raises `KeyError`. `occlusion_source` is `scene_visibility`, `depth` or
`actor_cuboids`, according to the supplied evidence.

#### See also

[Simulators](../guides/simulators.md).

### avsectester.rendering.harmonizers

<!-- api:begin module=avsectester.rendering.harmonizers -->
Image harmonization with optional in-process PCTNet inference.

| Name and signature | Meaning |
|---|---|
| `Harmonizer(ABC)()` | Adjust the pasted foreground to fit the background. Must preserve the patch's texture. |
| `Harmonizer.differentiable: ClassVar[bool] = False` | Whether input-image gradients are supported by this component. |
| `Harmonizer.harmonize_tensor(composite: torch.Tensor, mask: torch.Tensor, background: torch.Tensor, nominal_composite: torch.Tensor) -> torch.Tensor` | Adjust CHW floating sRGB only within the HW boolean visible mask. |
| `Harmonizer.__call__(composite_rgb: np.ndarray, mask: np.ndarray, background_rgb: np.ndarray) -> np.ndarray` | Abstract. Return H×W×3 uint8 RGB adjusted only inside the H×W boolean visible mask. |
| `ExposureHarmonizer(Harmonizer)(reference: str = "ring", ring_px: int = 12, gain_range: tuple[float, float] = (.05, 1.2), feather_sigma: float = 1.2)` | Match visible foreground exposure to a background reference in linear RGB. |
| `ExposureHarmonizer.differentiable: ClassVar[bool] = True` | Whether input-image gradients are supported by this component. |
| `ExposureHarmonizer.exposure_gain(composite: torch.Tensor, mask: torch.Tensor, background: torch.Tensor) -> torch.Tensor \| None` | Return the clipped gain from `composite`'s visible luminance to the reference. |
| `ExposureHarmonizer.__call__(composite_rgb: np.ndarray, mask: np.ndarray, background_rgb: np.ndarray) -> np.ndarray` | Apply exposure matching and return uint8 RGB without changing mask-exterior pixels. |
| `ClassicHarmonizer(Harmonizer)(color_transfer: bool = True, poisson: bool = True, preserve_chroma: bool = False, blend: str = "poisson", feather: float = 1.2, gain_range: tuple = (0.35, 1.2))` | Reinhard color transfer (match the patch's Lab mean/std to the scene) + Poisson seamless blend. |
| `ClassicHarmonizer.__call__(composite_rgb, mask, background_rgb)` | Transfer foreground color statistics and optionally blend its boundary with Poisson editing. |
| `PCTNetHarmonizer(Harmonizer)(libcom_directory: str \| Path, device: str = "cuda:0", weights: str \| Path \| None = None, strict: bool = False)` | Learned harmonization via libcom's **PCTNet**, run **in-process**. |
| `PCTNetHarmonizer.__call__(composite_rgb, mask, background_rgb)` | Harmonize visible foreground with PCTNet, applying the configured strict error policy. |
<!-- api:end -->

#### Constraints and errors

`ExposureHarmonizer` converts sRGB c to linear l: l=c/12.92 for c≤0.04045;
otherwise l=((c+0.055)/1.055)^2.4.

Luminance is Y=0.2126R+0.7152G+0.0722B. Gain is
`g = clip(mean(Y_reference) / (mean(Y_nominal_visible) + 1e-6), gain_range)`.
The reference is
the outside ring or the original background under the object. `CompositePlan` obtains nominal
visible color from each surface's `nominal_texture` when available; without a separate nominal
view, NumPy calls estimate gain from the composite itself.

Multiply visible candidate linear RGB by g. Encode sRGB: c=12.92l for l≤0.0031308; otherwise
c=1.055l^(1/2.4)−0.055. Pixels outside the mask retain input; empty visible/reference regions
return input unchanged.

`ClassicHarmonizer` and `PCTNetHarmonizer` are non-differentiable. PCTNet accepts a string
device and falls back to CPU when CUDA is unavailable.

Tensor harmonizers receive CHW float `composite/background/nominal_composite` in [0,1] and an HW
boolean visible mask; the nominal tensor matches composite shape, dtype and device.
`ExposureHarmonizer.exposure_gain` returns the clipped scalar gain or None for an empty
visible/reference region. Feathering is output=a×adjusted+(1−a)×background inside the mask,
where a is the Gaussian-blurred float mask; `feather_sigma=0` disables it.

Custom tensor harmonizers declare differentiable=True, implement `harmonize_tensor` and preserve
mask-exterior pixels. The base tensor method raises `NotImplementedError`.

PCTNet construction neither loads a model nor downloads weights. First harmonization loads only
libcom's source modules. Explicit weights take precedence; otherwise it
uses `<libcom_directory>/libcom/image_harmonization/pretrained_models/PCTNet.pth`. If that
file is absent, first use requests `PCTNet.pth` from `BCMIZB/Libcom_pretrained_models` through
`huggingface_hub` into the same directory. strict=True wraps loading/inference failure in
`RuntimeError`; strict=False uses `ClassicHarmonizer` fallback.

#### Extension contract

Implement `Harmonizer.__call__(composite, mask, background)` for HWC RGB arrays.
Return matching shape and dtype, preserve pixels outside the mask and leave inputs unchanged.
For differentiable composition, declare `differentiable=True` and implement `harmonize_tensor`
with CHW floating RGB, an HW mask and gradients connected to candidate colors.

#### See also

[Simulators](../guides/simulators.md).

### avsectester.rendering.objects

<!-- api:begin module=avsectester.rendering.objects -->
Textured world-object descriptions, independent of simulators and attacks.

| Name and signature | Meaning |
|---|---|
| `PlaneSurface(corners: np.ndarray, texture: np.ndarray, nominal_texture: np.ndarray \| None = None)` | A textured rectangle, with corners ordered TL, TR, BR, BL from its front. |
| `ObjectAsset(surfaces: tuple[PlaneSurface, ...], extent: tuple[float, float, float], category: str)` | Local rectangular faces in x-forward, y-left, z-up axes, with bottom-center origin. |
| `ObjectAsset.from_surfaces(surfaces: Sequence[PlaneSurface], category: str) -> ObjectAsset` | Infer full extent from local corners without moving the bottom-center origin. |
| `ObjectAsset.from_panel(texture: np.ndarray \| str \| Path \| Texture, *, category: str, width_m: float, mount_height_m: float = 0.0, post_count: int = 0, post_width_m: float = 0.07, nominal_texture: np.ndarray \| str \| Path \| Texture \| None = None) -> ObjectAsset` | Build a +X-facing panel and optional support posts above a ground-level origin. |
| `PlacedObject(object_id: str, asset: ObjectAsset, world_from_object: np.ndarray, harmonizer: Harmonizer \| None = None)` | One asset at a rigid world-from-object transform with its own optional harmonizer. |
| `PlacedObject.world_surfaces() -> tuple[PlaneSurface, ...]` | The asset's surfaces with corners in world coordinates. |
<!-- api:end -->

#### Constraints and errors

RGBA surfaces are uint8 in object-local coordinates. `ObjectAsset.extent` is (thickness, width,
height); `from_panel` uses +X as the face normal. Transparent texture pixels do not cover
the background.

`PlaneSurface.nominal_texture` is optional RGBA with the same shape and alpha as texture. It
records the unmodified appearance used for exposure estimation while candidate colors
can change.

`ObjectAsset.from_panel` lists support posts first and the face last. `mount_height_m` is the
height of the face bottom above the ground-level object origin; posts reach the face center.

#### See also

[Implement an attack](../guides/attacks.md).

### avsectester.rendering.surfaces

<!-- api:begin module=avsectester.rendering.surfaces -->
Prefiltered planar texture sampling shared by composition, visibility and optimization.

| Name and signature | Meaning |
|---|---|
| `SurfaceSampler(corners: np.ndarray, texture: np.ndarray, bounds: tuple[int, int, int, int] \| None, reason: str = "")` | A camera-space rectangle, its RGBA texture and its full projected pixel bounds. |
| `SurfaceSampler.intersect(rays: np.ndarray) -> tuple[np.ndarray, np.ndarray]` | Distances to the unbounded plane along each ray and its (s, t) plane coordinates. |
| `SurfaceSampler.sample(rays: np.ndarray) -> tuple[np.ndarray, np.ndarray]` | Distances to opaque texels and the full-resolution RGBA they hit, for any rays. |
| `SurfaceSampler.image_bounds(camera: Camera) -> tuple[int, int, int, int] \| None` | The projection bounds clipped to the image, the whole image when they are unavailable, or None when the surface is behind the camera or projects outside the image. |
| `PlaneSampling(bounds: tuple[int, int, int, int], coordinates: np.ndarray, derivatives: np.ndarray, depth: np.ndarray)` | Footprint geometry of one plane over an integer pixel region, independent of texture size. |
| `plane_sampling(sampler: SurfaceSampler, camera: Camera, bounds: tuple[int, int, int, int]) -> PlaneSampling` | Unproject the region's subpixel rays through the lens and intersect them with the plane. |
| `prepare_surface(surface: PlaneSurface, camera: Camera, cam_from_world) -> SurfaceSampler` | Transform a surface into the camera and bound its full projection, including off-image pixels. |
| `sample_texture(texture: torch.Tensor, sampling: PlaneSampling) -> torch.Tensor` | Filtered premultiplied RGBA `(4, H, W)` of a `(4, h, w)` premultiplied texture. |
| `render_surfaces(surfaces: Sequence[PlaneSurface], camera: Camera, cam_from_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]` | Image-sized uint8 RGBA and optical-axis depth of the nearest covering surface per pixel. |
<!-- api:end -->

#### Constraints and errors

The public sampling functions expose `PlaneSampling`, surface preparation, differentiable
texture sampling and surface rendering. Tensor textures use premultiplied CHW RGBA; RGB channels
must already be multiplied by alpha. `SurfaceSampler` is the public result of `prepare_surface`;
its image bounds feed `plane_sampling`. Rasterization helpers remain internal.

A rendered surface sample counts as an occlusion hit only when normalized alpha exceeds
`0.5 / 255`; lower alpha still contributes its original blending weight.

#### See also

[Implement an attack](../guides/attacks.md).

### avsectester.rendering.textures

<!-- api:begin module=avsectester.rendering.textures -->
Texture descriptions and independent RGBA image loading for rendering assets.

| Name and signature | Meaning |
|---|---|
| `STOP_SIGN_FACE` | Path to the bundled nominal stop-sign face texture. |
| `Texture(Protocol)` | A user-provided texture that returns an RGB or RGBA uint8 image. |
| `Texture.rgba() -> np.ndarray` | Produce an H×W×3 RGB or H×W×4 RGBA uint8 image. |
| `load_rgba_image(value: np.ndarray \| str \| Path \| Texture) -> np.ndarray` | Load one RGB/RGBA image as an independent uint8 RGBA array. |
| `CheckerboardTexture(size: int = 256, cells: int = 8, colors: tuple = ((255, 0, 0), (255, 255, 255)))` | An opaque square checkerboard with an explicit pixel size and cell count. |
| `CheckerboardTexture.rgba() -> np.ndarray` | Produce the texture as an H×W×4 uint8 RGBA image. |
<!-- api:end -->

#### Constraints and errors

Textures produce RGB/RGBA pixels at requested sizes. Invalid shapes and unsupported color
specifications raise `ValueError`.

`STOP_SIGN_FACE` points to the packaged 512×512 RGBA stop-sign face.

#### See also

[Implement an attack](../guides/attacks.md).

### avsectester.closed_loop

<!-- api:begin module=avsectester.closed_loop -->
Closed-loop components and their shared hooks and environment.

No exports; import from submodules.
<!-- api:end -->

#### See also

[Runtime reference](runtime.md).

### avsectester.closed_loop.avstacks

<!-- api:begin module=avsectester.closed_loop.avstacks -->
Driving-system interfaces; import implementations from their own modules.

| Name and signature | Meaning |
|---|---|
| `AVStack` | [AVStack](#avsectesterclosed_loopavstacksbase) |
| `ObservationPipeline` | [ObservationPipeline](#avsectesterclosed_loopavstacksbase) |
| `CommandCompatibilityError` | [CommandCompatibilityError](#avsectesterclosed_loopavstacksbase) |
| `validate_command_compatibility` | [validate_command_compatibility](#avsectesterclosed_loopavstacksbase) |
| `DrivingAdapter` | [DrivingAdapter](#avsectesterclosed_loopavstacksdriving_adapter) |
| `ComponentDescriptor` | [ComponentDescriptor](#avsectesterclosed_loopavstacksdriving_adapter) |
| `ScorerDescriptor` | [ScorerDescriptor](#avsectesterclosed_loopavstacksdriving_adapter) |
<!-- api:end -->

#### See also

[Driving systems](../guides/driving-systems.md).

### avsectester.closed_loop.avstacks.base

<!-- api:begin module=avsectester.closed_loop.avstacks.base -->
The driving-system interface: an Observation in, a Control out.

| Name and signature | Meaning |
|---|---|
| `ObservationPipeline(localizer: Any = None)` | Process driving-system inputs without changing output commands. |
| `ObservationPipeline.process(observation: Observation, dispatcher: HookDispatcher) -> Observation` | Apply localization and plugins, emitting their supported hook stages. |
| `AVStack(ABC)()` | The driving system: an Observation in, a Control out. |
| `AVStack.provides_detections: bool \| None = None` | Whether decisions expose detections compatible with detection metrics. |
| `AVStack.past_steps: ClassVar[int] = 0` | Recorded decision steps required before the selected start. |
| `AVStack.decision_time_step: ClassVar[float \| None] = None` | Decision interval in seconds; None leaves the interval unrestricted. |
| `AVStack.observation_pipeline: ObservationPipeline` | Driving-system input processing, including its localizer lifecycle. |
| `AVStack.reset(observation: Observation, past: Sequence[Observation] = ()) -> None` | Initialize from the first observation and recorded past, ordered oldest to newest. |
| `AVStack.command_modes: frozenset[str] \| None` | Possible output command representations, or None for an undeclared adapter. |
| `AVStack.models: Mapping[str, Any]` | Loaded native models available to user handlers. Adapters expose their own names. |
| `AVStack.component_log() -> Mapping[str, Any] \| None` | Already-produced component outputs for the current decision, or None. |
| `AVStack.detections() -> Sequence[Detection] \| None` | Current decision's ego-frame detections; None means no detection output. |
| `AVStack.close() -> None` | Release driving-system resources. The default implementation owns none. |
| `AVStack.__call__(observation: Observation) -> Control` | Abstract. Advance driving inference once and return a command in `command_modes`. |
| `AVStack.supported_stages: frozenset[str]` | Return the hook stages supported by this adapter. |
| `AVStack.native: Mapping[str, Any]` | Expose native components available to callbacks, keyed by component name. |
| `AVStack.bind_hooks(dispatcher: StageEmitter) -> Iterator[None]` | Context manager. Bind one dispatcher for the scope; reject an already active binding. |
| `CommandCompatibilityError(ValueError)` | Declared AVStack commands cannot be consumed by the selected simulator. |
| `validate_command_compatibility(simulator_runtime: SimulatorRuntime, avstack: AVStack, hooks: HookSet \| None = None) -> None` | Check declared commands before world setup, allowing explicit hook conversion. |
<!-- api:end -->

#### Constraints and errors

`AVStack` declares `past_steps` and optional `decision_time_step` without model construction.
reset receives the current processed observation and chronological past observations. None
detections means unsupported; an empty sequence means zero detections. `ObservationPipeline`
handles inputs and localization, not output commands.

#### Extension contract

Implement `AVStack.__call__(observation) -> Control`. The runner calls it once per decision
after `reset` with processed model-visible input. Retain history only within that run and
return a declared command mode. Trajectory poses use rig axes and absolute run timestamps;
do not mutate the supplied observation.

#### See also

[Driving systems](../guides/driving-systems.md).

### avsectester.closed_loop.avstacks.driving_adapter

<!-- api:begin module=avsectester.closed_loop.avstacks.driving_adapter -->
Driving-system component schemas, scorer capabilities and fresh instance bindings.

| Name and signature | Meaning |
|---|---|
| `ComponentDescriptor(id: str, inputs: Mapping[str, str], outputs: Mapping[str, str], codecs: Mapping[str, str], differentiable_inputs: tuple[str, ...] = ())` | Description of an evaluable driving-system component and its input/output formats. |
| `ScorerDescriptor(ComponentDescriptor)(id: str, inputs: Mapping[str, str], outputs: Mapping[str, str], codecs: Mapping[str, str], differentiable_inputs: tuple[str, ...] = (), output_shape: tuple[int, ...] = (), direction: str = "minimize", sensor_id: str \| None = None, decision_threshold: float \| None = None)` | Description of a scoring component, gradient support and optional sensor decision threshold. |
| `DrivingAdapter(factory: Callable[[int], AVStack], *, components: Sequence[tuple[ComponentDescriptor, Callable[..., Any]]] = (), scorers: Sequence[tuple[ScorerDescriptor, Callable[..., Any]]] = (), past_steps: int = 0, decision_time_step: float \| None = None)` | Bind component callbacks `(avstack, inputs)` and scorer callbacks `(avstack, inputs, target)` to a fresh AVStack factory. |
| `DrivingAdapter.past_steps: int` | Required recorded history, declared by the AVStack type or this adapter. |
| `DrivingAdapter.decision_time_step: float \| None` | Required decision interval, read without constructing the driving system. |
<!-- api:end -->

#### Constraints and errors

Factory `avstack_type` declarations take precedence for `past_steps` and `decision_time_step`.
Bindings need unique IDs, callable callbacks, schemas/codecs and valid differentiable inputs.
`ScorerDescriptor.sensor_id` identifies the image sensor; `decision_threshold` is the
driving-system threshold in [0,1].

A driving-system factory is called as `factory(seed)`.

#### Extension contract

Component callbacks use `(avstack, inputs)`; scorer callbacks use `(avstack, inputs, target)`.
They operate on the session's fresh stack and must match their declared schemas.
Differentiable scorers preserve the input gradient when gradient recording is enabled.

#### See also

[Driving systems](../guides/driving-systems.md).

### avsectester.closed_loop.avstacks.implemented

<!-- api:begin module=avsectester.closed_loop.avstacks.implemented -->
Driving systems provided with the framework; import each from its own module.

No exports; import from submodules.
<!-- api:end -->

#### See also

[Driving systems](../guides/driving-systems.md).

### avsectester.closed_loop.avstacks.implemented.alpamayo

<!-- api:begin module=avsectester.closed_loop.avstacks.implemented.alpamayo -->
Alpamayo end-to-end camera policy as an `avsectester.closed_loop.avstacks.base.AVStack`.

| Name and signature | Meaning |
|---|---|
| `alpamayo_driving_adapter(factory)` | Bind one `policy` component (observation + past → Control) and no scorers. |
| `AlpamayoAVStack(AVStack)(checkpoint_path: str \| Path \| None = None, camera_ids: list[str] \| None = None, device: str = "cuda:0", context_length: int = 4, frame_subsample: int = 1, output_frequency_hz: int = 10, model: Any \| None = None, seed: int \| None = None)` | A policy using real 10 Hz observations, image context and model-visible navigation. |
| `AlpamayoAVStack.provides_detections = False` | Whether decisions expose detections compatible with detection metrics. |
| `AlpamayoAVStack.past_steps: ClassVar[int] = 15` | Recorded decision steps required before the selected start. |
| `AlpamayoAVStack.decision_time_step: ClassVar[float \| None] = 0.1` | Decision interval in seconds; None leaves the interval unrestricted. |
| `AlpamayoAVStack.supported_stages: frozenset[str]` | Expose `policy.pre` and `policy.post` around the end-to-end policy. |
| `AlpamayoAVStack.native: Mapping[str, Any]` | Expose native components available to callbacks, keyed by component name. |
| `AlpamayoAVStack.models: Mapping[str, Any]` | The loaded policy model. Empty until reset or the first inference loads it. |
| `AlpamayoAVStack.reset(observation: Observation, past: Sequence[Observation] = ()) -> None` | Initialize policy context from recorded observations before the selected origin. |
| `AlpamayoAVStack.close() -> None` | Release this adapter's references and any model it loaded itself. |
| `AlpamayoAVStack.EPOCH_US = 10_000_000` | Timestamp offset in microseconds used by the trajectory interface. |
| `AlpamayoAVStack.__call__(observation: Observation) -> Control` | Append the current capture and infer a rig-frame trajectory from camera and ego history. |
| `AlpamayoAVStack.component_log() -> dict[str, Any] \| None` | Already-produced policy prediction and Control under `policy` and `command`. |
| `AlpamayoAVStack.bind_hooks(dispatcher: StageEmitter) -> Iterator[None]` | Context manager. Bind one dispatcher for the scope; reject an already active binding. |
<!-- api:end -->

#### Constraints and errors

Alpamayo requires 0.1 s decisions and recorded history. Its binding reads declarations without
creating a model. Returned plans use rig-frame coordinates and integer-microsecond timestamps.

History sampling requires `(context_length - 1) * frame_subsample <= 15`.

#### See also

[Driving systems](../guides/driving-systems.md).

### avsectester.closed_loop.avstacks.implemented.camera_forward_collision

<!-- api:begin module=avsectester.closed_loop.avstacks.implemented.camera_forward_collision -->
Camera detection with the box-area forward-collision braking rule.

| Name and signature | Meaning |
|---|---|
| `CameraForwardCollisionAVStack(AVStack)(gpu: int = 0, camera: str = "front", area_brake: float = .03, throttle: float = .55, detector_dataset: str = "carla-vehicle", threshold: float = .3, area_max: float = .3, max_height_frac: float = .6)` | Brake for a plausible detected vehicle whose image area exceeds `area_brake`. |
| `CameraForwardCollisionAVStack.provides_detections = False` | Whether decisions expose detections compatible with detection metrics. |
| `CameraForwardCollisionAVStack.command_modes = frozenset({"actuator"})` | Control command modes accepted or produced by this component. |
| `CameraForwardCollisionAVStack.supported_stages = frozenset({"perception.pre", "perception.post"})` | Hook stages supported by the component. |
| `CameraForwardCollisionAVStack.detect(rgb: Any) -> list` | Return current image detections as `(xyxy, score, vehicle)` tuples. |
| `CameraForwardCollisionAVStack.__call__(observation: Observation) -> Control` | Detect the selected RGB image and choose braking from the largest image-box area. |
| `CameraForwardCollisionAVStack.native: Mapping[str, Any]` | Expose native components available to callbacks, keyed by component name. |
| `CameraForwardCollisionAVStack.bind_hooks(dispatcher: StageEmitter) -> Iterator[None]` | Context manager. Bind one dispatcher for the scope; reject an already active binding. |
<!-- api:end -->

#### Constraints and errors

Camera inputs use their declared RGB transport and native detector preprocessing. Component
logs expose produced perception output without additional inference.

#### See also

[Driving systems](../guides/driving-systems.md).

### avsectester.closed_loop.avstacks.implemented.modular

<!-- api:begin module=avsectester.closed_loop.avstacks.implemented.modular -->
ModularAVStack — an AVStack backed by an avstack `ModularDrivingPipeline`.

| Name and signature | Meaning |
|---|---|
| `NativeHook(Plugin)(config: Mapping[str, Any], *, phase: str = "post")` | Adapt a registered avstack hook's tuple convention to an ordered stage hook. |
| `register_carla_dataset() -> type` | Return the CARLA dataset class, registering its vendored declaration if needed. |
| `modular_driving_adapter(factory, *, components=(), scorers=())` | Expose explicitly selected native Modular components and existing detector scorers. |
| `ModularAVStack(AVStack)(pipeline_cfg: dict)` | An `avsectester.closed_loop.avstacks.base.AVStack` backed by an avstack `ModularDrivingPipeline`. |
| `ModularAVStack.provides_detections = True` | Whether decisions expose detections compatible with detection metrics. |
| `ModularAVStack.reset(observation: Observation, past: Sequence[Observation] = ()) -> None` | Clear telemetry. Use a fresh AVStack for independent pipeline histories. |
| `ModularAVStack.supported_stages: frozenset[str]` | Return the hook stages supported by this adapter. |
| `ModularAVStack.native: Mapping[str, Any]` | The pipeline and its actual modules, including non-hookable modules. |
| `ModularAVStack.models: Mapping[str, Any]` | Public handles for the built components and their loaded models. |
| `ModularAVStack.bind_hooks(dispatcher: StageEmitter) -> Iterator[None]` | Context manager. Bridge hook callbacks for one run, preserving native hook order. |
| `ModularAVStack.instrument(stages: tuple[str, ...] = ("perception", "tracking", "planning", "control")) -> None` | Capture named stage outputs after attack hooks, enabling `component_log`. |
| `ModularAVStack.component_log() -> dict[str, Any] \| None` | Return the most recent per-stage outputs after `instrument`, or None otherwise. |
| `ModularAVStack.detections() -> Sequence[Detection] \| None` | Boxes from the current decision in observation-time ego coordinates. |
| `ModularAVStack.__call__(observation: Observation) -> Control` | Run the native perception-to-control pipeline once using the current capture reference. |
<!-- api:end -->

#### Constraints and errors

The native avstack configuration and optional SDK must be available. `NativeHook` adapts
explicitly registered native stages; `modular_driving_adapter` declares component/scorer schemas
without changing the pipeline.

#### See also

[Driving systems](../guides/driving-systems.md).

### avsectester.closed_loop.avstacks.implemented.rule_based

<!-- api:begin module=avsectester.closed_loop.avstacks.implemented.rule_based -->
Detect and range signs, track bearings with an EKF, then cruise and brake to stops.

| Name and signature | Meaning |
|---|---|
| `RuleBasedAVStack` | [RuleBasedAVStack](#avsectesterclosed_loopavstacksimplementedrule_basedstack) |
| `CocoStopSignDetector` | [CocoStopSignDetector](#avsectesterclosed_loopavstacksimplementedrule_baseddetector) |
| `StopSignPlanner` | [StopSignPlanner](#avsectesterclosed_loopavstacksimplementedrule_basedplanner) |
| `StopSignTracker` | [StopSignTracker](#avsectesterclosed_loopavstacksimplementedrule_basedtracker) |
| `SignTrack` | [SignTrack](#avsectesterclosed_loopavstacksimplementedrule_basedtracker) |
| `rule_based_driving_adapter` | [rule_based_driving_adapter](#avsectesterclosed_loopavstacksimplementedrule_basedstack) |
<!-- api:end -->

#### See also

[Driving systems](../guides/driving-systems.md).

### avsectester.closed_loop.avstacks.implemented.rule_based.detector

<!-- api:begin module=avsectester.closed_loop.avstacks.implemented.rule_based.detector -->
Local COCO stop-sign detection shared by rule driving and detector scorers.

| Name and signature | Meaning |
|---|---|
| `CocoStopSignDetector(weights_path: str \| Path, device: str = "cuda:0", score_threshold: float = .5, model: Any = None)` | Detect only COCO category 13 in uint8 RGB images using a local checkpoint. |
| `CocoStopSignDetector.detect(rgb: np.ndarray) -> list[tuple[np.ndarray, float]]` | Return original-image xyxy boxes and scores at or above score_threshold. |
| `CocoStopSignDetector.target_score(image: torch.Tensor, target: TargetSpec) -> torch.Tensor` | Return a scalar confidence for a float RGB `(3, H, W)` image in [0, 1]. |
| `CocoStopSignDetector.close() -> None` | Release the model; a closed detector cannot detect or score again. |
<!-- api:end -->

#### Constraints and errors

Only local `COCO_V1` Faster R-CNN weights are loaded, lazily on first inference. detect filters
stop-sign class 13 by `score_threshold`. `target_score` requires a finite floating RGB (3,H,W)
image in [0,1] and a `TargetSpec` with label None or stop sign and finite positive-area xyxy
coordinates. It selects the highest confidence among class-13 boxes with `IoU` ≥0.3. Box/NMS
selection is discrete; gradients flow through the selected confidence. No match returns an
input-connected zero.

A closed detector rejects `detect` and `target_score` with `RuntimeError`.

#### See also

[Driving systems](../guides/driving-systems.md).

### avsectester.closed_loop.avstacks.implemented.rule_based.planner

<!-- api:begin module=avsectester.closed_loop.avstacks.implemented.rule_based.planner -->
Route-following cruise and constant-deceleration stops at tracked signs.

| Name and signature | Meaning |
|---|---|
| `StopSignPlanner(stop_before_sign_m: float = 5.5, stop_duration_s: float = 2., stop_speed_mps: float = .1, stop_position_tolerance_m: float = 2., sign_lookahead_m: float = 60., max_sign_lateral_offset_m: float = 10., cruise_speed_mps: float \| None = None, min_cruise_speed_mps: float = 5., max_acceleration_mps2: float = 1., comfortable_deceleration_mps2: float = 3., max_deceleration_mps2: float = 6., acceleration_exponent: float = 4., trajectory_horizon_s: float = 5., trajectory_step_s: float = .1)` | Plan rig-frame trajectories from bearing-EKF sign tracks. |
| `StopSignPlanner.component_log: Mapping[str, Any]` | State, target track ID, target distance and acceleration/speed at the first planned step. |
| `StopSignPlanner.reset(observation: Observation) -> None` | Forget served track IDs and initialize cruise speed from the observation. |
| `StopSignPlanner.step(observation: Observation, tracks: Sequence[SignTrack]) -> Control` | Use horizontal world tracks to return an absolute-time rig-frame plan. |
<!-- api:end -->

#### Constraints and errors

Cruise uses the IDM free-road term. Braking latches when required deceleration reaches the
comfortable threshold; it clears when the target ID changes, disappears or reaches STOPPED.
Deceleration is limited by `max_deceleration_mps2`. Each track ID is served once per reset.

`trajectory_horizon_s` must be at least `trajectory_step_s`; rounding the step to integer
microseconds must yield a positive interval. See [Driving systems](../guides/driving-systems.md)
for cruise-speed and braking behavior.

| Parameter | Meaning |
|---|---|
| `stop_before_sign_m` | Route distance from the sign projection back to the stop target |
| `stop_duration_s` | Time held after entering STOPPED |
| `stop_speed_mps` | Strict upper speed bound for declaring the vehicle stopped |
| `stop_position_tolerance_m` | Maximum remaining target distance when declaring a stop |
| `sign_lookahead_m` | Farthest sign projection considered ahead along the route |
| `max_sign_lateral_offset_m` | Maximum distance from sign to the finite route |
| `cruise_speed_mps` | Fixed cruise reference; `None` derives it at reset |
| `min_cruise_speed_mps` | Lower bound for the cruise reference derived at reset |
| `max_acceleration_mps2` | Upper acceleration bound and free-road acceleration scale |
| `comfortable_deceleration_mps2` | Required-deceleration threshold that latches braking |
| `max_deceleration_mps2` | Maximum braking magnitude, including late detections |
| `acceleration_exponent` | Exponent of the cruise-speed ratio |
| `trajectory_horizon_s` | Duration of the forecast trajectory |
| `trajectory_step_s` | Forecast integration and waypoint interval |

#### See also

[Driving systems](../guides/driving-systems.md).

### avsectester.closed_loop.avstacks.implemented.rule_based.ranging

<!-- api:begin module=avsectester.closed_loop.avstacks.implemented.rule_based.ranging -->
Monocular stop-sign position from calibrated angular height.

| Name and signature | Meaning |
|---|---|
| `estimate_stop_sign_position(box_xyxy: tuple[float, float, float, float] \| np.ndarray, calibration: CameraCalibration, sign_height_m: float = .762) -> np.ndarray` | Estimate the sign center in rig meters from its angular image height. |
<!-- api:end -->

#### Constraints and errors

Known-size range is `sign_height_m / (2 tan(θ/2))`, where θ is the angle between box-top and
box-bottom rays. The point is transformed from optical camera to rig coordinates. Error scales
with actual sign-size error.

#### See also

[Driving systems](../guides/driving-systems.md).

### avsectester.closed_loop.avstacks.implemented.rule_based.stack

<!-- api:begin module=avsectester.closed_loop.avstacks.implemented.rule_based.stack -->
Detect signs, range by known size, track bearings and plan route-following stops.

| Name and signature | Meaning |
|---|---|
| `RuleBasedAVStack(AVStack)(detector: CocoStopSignDetector, tracker: StopSignTracker \| None = None, planner: StopSignPlanner \| None = None, camera_id: str = DEFAULT_CAMERA_ID, sign_height_m: float = .762)` | Follow the route and stop for signs using calibrated camera observations. |
| `RuleBasedAVStack.command_modes = frozenset({"trajectory"})` | Control command modes accepted or produced by this component. |
| `RuleBasedAVStack.provides_detections = False` | Whether decisions expose detections compatible with detection metrics. |
| `RuleBasedAVStack.reset(observation: Observation, past: Sequence[Observation] = ()) -> None` | Clear tracks and the stop state machine for a fresh run. |
| `RuleBasedAVStack.__call__(observation: Observation) -> Control` | Detect, range and track stop signs, then return the planner's rig-frame trajectory. |
| `RuleBasedAVStack.component_log() -> Mapping[str, Any] \| None` | Return detections, ranges, world tracks and current planner outputs. |
| `RuleBasedAVStack.close() -> None` | Release the detection model and clear recorded decision outputs. |
| `RuleBasedAVStack.supported_stages: frozenset[str]` | Return the hook stages supported by this adapter. |
| `RuleBasedAVStack.native: Mapping[str, Any]` | Expose native components available to callbacks, keyed by component name. |
| `RuleBasedAVStack.bind_hooks(dispatcher: StageEmitter) -> Iterator[None]` | Context manager. Bind one dispatcher for the scope; reject an already active binding. |
| `rule_based_driving_adapter(factory: Callable[[int], AVStack]) -> DrivingAdapter` | Bind the configured rule driver's differentiable stop-sign confidence scorer. |
<!-- api:end -->

#### Constraints and errors

The detector must be a `CocoStopSignDetector`. tracker and planner must be `StopSignTracker` and
`StopSignPlanner` instances or None; None constructs their defaults. Invalid component types
raise `TypeError`. `CameraCalibration` is required for the selected camera. Logged acceleration
and speed describe the first planned trajectory step; the controller follows the
whole trajectory.

#### See also

[Driving systems](../guides/driving-systems.md).

### avsectester.closed_loop.avstacks.implemented.rule_based.tracker

<!-- api:begin module=avsectester.closed_loop.avstacks.implemented.rule_based.tracker -->
Static stop-sign tracking from known-size ranges and bearing EKF updates.

| Name and signature | Meaning |
|---|---|
| `SignTrack(track_id: int, position: np.ndarray, covariance: np.ndarray)` | A stationary sign's ID, horizontal world position and 2×2 covariance. |
| `StopSignTracker(range_sigma_fraction: float = .3, bearing_sigma_rad: float = .007, min_position_sigma_m: float = .3, max_association_mahalanobis_distance: float = 3.5)` | Associate range measurements, then refine static sign positions by bearing EKF. |
| `StopSignTracker.reset() -> None` | Forget all signs and restart track IDs at zero. |
| `StopSignTracker.tracks: tuple[SignTrack, ...]` | Return detached snapshots of the horizontal world tracks. |
| `StopSignTracker.update(vehicle_state: PlanarEgoState, camera_position_rig: np.ndarray, sign_positions_rig: Sequence[np.ndarray]) -> None` | Associate rig-frame measurements and update matched tracks by bearing. |
<!-- api:end -->

#### Constraints and errors

Tracks are stationary horizontal world positions. Range initializes and associates tracks;
bearing alone performs EKF updates. Association uses joint covariance and a Mahalanobis gate.
Reset clears tracks and restarts IDs.

#### See also

[Driving systems](../guides/driving-systems.md).

### avsectester.closed_loop.avstacks.implemented.scorers

<!-- api:begin module=avsectester.closed_loop.avstacks.implemented.scorers -->
Differentiable RPN objectness and inference-confidence scorers for mmdet detectors.

| Name and signature | Meaning |
|---|---|
| `MMDetRPNObjectnessScorer(model, strides=FPN_STRIDES)` | RPN-objectness confidence in the target box for an mmdet two-stage detector (white-box). |
| `MMDetRPNObjectnessScorer.differentiable = True` | Whether input-image gradients are supported by this component. |
| `MMDetRPNObjectnessScorer.target_score(image: torch.Tensor, target: data_types.TargetSpec) -> torch.Tensor` | Return differentiable RPN objectness within the target region. |
| `MMDetInferenceScorer(model)` | Black-box (non-differentiable) scorer: the detector's actual max confidence in the target box. |
| `MMDetInferenceScorer.differentiable = False` | Whether input-image gradients are supported by this component. |
| `MMDetInferenceScorer.target_score(image, target: data_types.TargetSpec) -> float` | Return matched detection confidence without requiring input gradients. |
<!-- api:end -->

#### Constraints and errors

These scorers use the modular stack's loaded MMDetection model. The RPN scorer returns a
disconnected scalar zero when no anchor lies in the target box.

#### See also

[Driving systems](../guides/driving-systems.md).

### avsectester.closed_loop.environment

<!-- api:begin module=avsectester.closed_loop.environment -->
World settings and deterministic sensor corruptions.

| Name and signature | Meaning |
|---|---|
| `normalize_environment(value: Mapping[str, Any] \| None = None) -> dict[str, Any]` | Copy and validate the public `{world, sensors}` structure. |
| `merge_environments(*values: Mapping[str, Any]) -> dict[str, Any]` | Merge world mappings recursively. An explicit later sensors list replaces the earlier list. |
| `validate_environment(value: Mapping[str, Any], simulator: Any) -> dict[str, Any]` | Reject unsupported world settings before opening any simulator resources. |
| `SensorEnvironment(environment: Mapping[str, Any], seed: int = 0)` | One run's corruption instances, with randomness indexed by seed and absolute dataset frame position. |
| `SensorEnvironment.apply(observation: Observation, position: int) -> Observation` | Apply configured corruptions to eligible sensors using the decision random generator. |
<!-- api:end -->

#### Constraints and errors

Environment support follows [Configuration](configuration.md#environment).
Sensor and hook randomization follow [Randomness](../guides/environments.md#randomness).

#### See also

[Experiment environments](../guides/environments.md).

### avsectester.closed_loop.hooks

<!-- api:begin module=avsectester.closed_loop.hooks -->
Ordered attack and defense callbacks with per-run lifecycle management.

| Name and signature | Meaning |
|---|---|
| `StageEmitter(Protocol)` | What an adapter is bound to during a run: normally the run's hook dispatcher. |
| `StageEmitter.emit(stage: str, value: Any, sensor: str \| None = None) -> Any` | Run registered callbacks for a supported stage and return the resulting value. |
| `UnsupportedStageError(ValueError)` | A requested hook stage is absent from the selected adapters. |
| `CallInputs(args: tuple = (), kwargs: dict[str, Any] = dict())` | Inputs to a component pre-hook, retaining positional and keyword arguments. |
| `CameraPoseRequest(pose: Any, camera: str)` | Camera name and rendering pose editable at a rendering simulator's `render.pre` stage. |
| `RunContext(simulator_runtime: SimulatorRuntime, avstack: AVStack, rng: np.random.Generator)` | Borrowed run resources. Model, native SDK and metadata access is unrestricted. |
| `RunContext.native: Mapping[str, Mapping[str, Any]]` | Access a simulator- or driving-system-native run resource by name. |
| `StageContext(run: RunContext, step: int, stage: str, physical_state: WorldSnapshot, sensor: str \| None = None, observation: Observation \| None = None, control: Control \| None = None)` | One callback invocation and an independent snapshot of the physical state at it. |
| `StageContext.record_output(name: str, value: Any) -> None` | Save a named, independent output in this run's RunLog with callback timing. |
| `StageContext.simulator_runtime: SimulatorRuntime` | Simulator runtime associated with the active run. |
| `StageContext.avstack: AVStack` | Driving system associated with the active run. |
| `StageContext.rng: np.random.Generator` | Random generator assigned to this callback invocation. |
| `StageContext.native: Mapping[str, Mapping[str, Any]]` | Access a native resource through the active run context. |
| `StageContext.sim_time: float` | Physical simulation time at the active stage. |
| `Plugin(ABC)()` | Optional lifecycle base for a stateful attack, defense or localization adapter. |
| `Plugin.reset(context: RunContext) -> None` | Clear per-run history. Actors may not exist until `simulator.setup`. |
| `Plugin.transform(value: Any, context: StageContext) -> Any` | Abstract. Return the value consumed by the next handler or downstream component. |
| `Plugin.after_step(context: StageContext) -> None` | Receive feedback after the command has affected the world. Optional. |
| `Plugin.close() -> None` | Release resources, including partially initialized resources after an error. |
| `Hook(stage: str, handler: Any)` | Bind a function or a lifecycle plugin to an adapter-supported execution stage. |
| `HookSet(hooks: Iterable[Hook] = (), *, seed: int \| None = 0)` | Reusable ordered hooks, independent of the driving system's ObservationPipeline. |
| `HookSet.with_hooks(hooks: Iterable[Hook], *, prepend: bool = False) -> Self` | Copy this configuration with additional ordered hooks. |
| `HookSet.validate_stages(simulator_runtime: SimulatorRuntime, avstack: AVStack) -> frozenset[str]` | Check adapter capabilities without binding a dispatcher or invoking handlers. |
| `HookSet.setup_ticks: int` | Required deployment ticks, shared by every treatment in a comparison group. |
| `HookSet.session(simulator_runtime: SimulatorRuntime, avstack: AVStack) -> HookDispatcher` | Create one execution session before resetting the world. |
| `HookDispatcher(hook_set: HookSet, simulator_runtime: SimulatorRuntime, avstack: AVStack)` | One bound run. Custom adapters publish supported stages through `emit`. |
| `HookDispatcher.CORE_STAGES = frozenset({"sensors.post", "localization.post", "observation", "command"})` | Framework stages available independently of native adapter stages. |
| `HookDispatcher.activate() -> Iterator[HookDispatcher]` | Context manager. Bind both adapters and reset participants, closing them when the scope exits. |
| `HookDispatcher.stage_context(stage: str, sensor: str \| None = None) -> StageContext` | Capture current physical state and model-visible values for a callback. |
| `HookDispatcher.record_output(context: StageContext, name: str, value: Any) -> None` | Snapshot a named output; reject recording outside an active dispatcher scope. |
| `HookDispatcher.drain_outputs() -> list[RecordedOutput]` | Transfer recorded outputs to a consumer without retaining the run's history. |
| `HookDispatcher.emit(stage: str, value: Any, sensor: str \| None = None) -> Any` | Transform a value through the supported stage's ordered callbacks. |
| `HookDispatcher.observe(observation: Observation) -> Observation` | Apply the driving system's ObservationPipeline with this run's ordered hooks. |
| `HookDispatcher.command(control: Control) -> Control` | Apply command hooks and validate the resulting command against the simulator. |
| `HookDispatcher.after_step(observation: Observation) -> None` | Notify participants with independent copies of the next raw measurement. |
<!-- api:end -->

#### Constraints and errors

`Plugin.transform` must be implemented. Hooks retain registration order and shared handlers have
one lifecycle per run. `record_output` requires an active callback.
`emit` checks that the requested stage is supported; it does not require a callback.
`HookSet.setup_ticks` sums setup ticks declared by `simulator.setup` handlers; those ticks count
toward the deployment budget. The upper bound against `Attack.deployment_ticks` applies to
`AttackRuntime.setup_ticks`, not to `HookSet`.

#### Extension contract

Implement `Plugin.transform(value, context)` to return the next stage value.
Optional lifecycle methods reset per-run state, receive step feedback and release resources.
A handler registered at `simulator.setup` can declare nonnegative integer `setup_ticks`;
it may call `context.simulator_runtime.setup_tick()` within that budget.

#### See also

[Runtime reference](runtime.md).

### avsectester.closed_loop.sensor_corruptions

<!-- api:begin module=avsectester.closed_loop.sensor_corruptions -->
Image-space sensor corruptions used by experiment environments.

| Name and signature | Meaning |
|---|---|
| `CORRUPTIONS` | Mapping from configuration names to the supplied sensor corruption classes. |
| `Corruption(ABC)(severity: float = 0.5)` | One image-space corruption. `apply(rgb, rng) -> rgb` maps an `(H,W,3)` uint8 RGB frame to a corrupted one; `severity` in `[0, 1]` scales the corruption strength (0 ≈ identity). |
| `Corruption.category: str = "sensor"` | Environment corruption category used to group related sensor effects. |
| `Corruption.name: str` | Configuration name identifying this corruption. |
| `Corruption.apply(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray` | Abstract. Transform one sensor payload using the supplied random generator. |
| `Fog(Corruption)(severity: float = 0.5)` | Atmospheric haze: blend toward a bright airlight and flatten contrast (uniform, depth-free). |
| `Fog.category = "weather"` | Environment corruption category used to group related sensor effects. |
| `Rain(Corruption)(severity: float = 0.5)` | Darken, then add motion-blurred bright streaks (procedural). |
| `Rain.category = "weather"` | Environment corruption category used to group related sensor effects. |
| `Snow(Corruption)(severity: float = 0.5)` | Bright speckles plus a slight brighten and blur (procedural). |
| `Snow.category = "weather"` | Environment corruption category used to group related sensor effects. |
| `Brightness(Corruption)(severity: float = 0.5)` | Multiplicative brightening (over-exposure / bright sun). |
| `Brightness.category = "lighting"` | Environment corruption category used to group related sensor effects. |
| `LowLight(Corruption)(severity: float = 0.5)` | Night / underexposure: darken and add read noise (a dark sensor is a noisy sensor). |
| `LowLight.category = "lighting"` | Environment corruption category used to group related sensor effects. |
| `Contrast(Corruption)(severity: float = 0.5)` | Reduce contrast toward the frame mean (flat, washed-out light). |
| `Contrast.category = "lighting"` | Environment corruption category used to group related sensor effects. |
| `Gamma(Corruption)(severity: float = 0.5)` | Non-linear tone shift (darker mid-tones), as from a different exposure curve. |
| `Gamma.category = "lighting"` | Environment corruption category used to group related sensor effects. |
| `ColorTemp(Corruption)(severity: float = 0.5, warm: bool = True)` | Color-cast shift — warm (default, sodium street-lights / sunset) or cool. |
| `ColorTemp.category = "lighting"` | Environment corruption category used to group related sensor effects. |
| `GaussianNoise(Corruption)(severity: float = 0.5)` | Additive Gaussian read noise. |
| `ShotNoise(Corruption)(severity: float = 0.5)` | Poisson (photon shot) noise — signal-dependent, as in low light. |
| `MotionBlur(Corruption)(severity: float = 0.5)` | Directional blur from ego/scene motion (random angle, length scales with severity). |
| `DefocusBlur(Corruption)(severity: float = 0.5)` | Out-of-focus blur (Gaussian, radius scales with severity). |
| `JPEGCompression(Corruption)(severity: float = 0.5)` | Lossy-compression block artefacts (encoder quality drops with severity). |
| `ChromaticAberration(Corruption)(severity: float = 0.5)` | Lateral color fringing — shift the red and blue channels apart. |
| `AlbumentationsCorruption(Corruption)(transform: str, severity: float = 0.5)` | Named Albumentations transform with deterministic, severity-scaled parameters. |
<!-- api:end -->

#### Constraints and errors

Corruptions transform camera RGB with an explicit random generator. Albumentations operators
require their optional package. Missing sensor IDs are errors rather than implicit
camera substitutions.

#### See also

[Experiment environments](../guides/environments.md).

### avsectester.closed_loop.simulators

<!-- api:begin module=avsectester.closed_loop.simulators -->
Simulator interfaces; import concrete implementations from their own modules.

| Name and signature | Meaning |
|---|---|
| `Simulator` | [Simulator](#avsectesterclosed_loopsimulatorsbase) |
| `SimulatorRuntime` | [SimulatorRuntime](#avsectesterclosed_loopsimulatorsbase) |
| `SensorView` | [SensorView](#avsectesterclosed_loopsimulatorsbase) |
| `SensorGeometry` | [SensorGeometry](#avsectesterclosed_loopsimulatorsbase) |
| `ClockSnapshot` | [ClockSnapshot](#avsectesterclosed_loopsimulatorsbase) |
| `DeploymentPlan` | [DeploymentPlan](#avsectesterclosed_loopsimulatorsbase) |
| `Initialization` | [Initialization](#avsectesterclosed_loopsimulatorsbase) |
| `FrameDataSequence` | [FrameDataSequence](#avsectesterclosed_loopsimulatorsbase) |
| `UnsupportedStartError` | [UnsupportedStartError](#avsectesterclosed_loopsimulatorsbase) |
<!-- api:end -->

#### See also

[Simulators](../guides/simulators.md).

### avsectester.closed_loop.simulators.base

<!-- api:begin module=avsectester.closed_loop.simulators.base -->
The simulator: a configuration (`Simulator`) and its live worlds (`SimulatorRuntime`).

| Name and signature | Meaning |
|---|---|
| `DeploymentPlan(ticks: int = 0)` | Shared tick budget for all treatments, including frame data generation. |
| `ClockSnapshot(t: float, frame: int)` | Physical simulator time and frame, never a model-visible timestamp. |
| `SensorGeometry(sensor: str, camera: Camera, cam_from_world: np.ndarray)` | Physical optical-camera geometry at one capture, without image or SDK handles. |
| `SensorView(sensor: str, data: Any, camera: Camera \| None, cam_from_world: np.ndarray \| None, actors: Mapping[str, ActorState], ego: ActorState, time: float, frame: int, sensor_from_world: np.ndarray \| None = None, depth: np.ndarray \| None = None, native: Mapping[str, Any] = dict(), objects: Mapping[str, PlacedObject] = dict(), composition: ObjectComposition \| None = None, object_occlusion: str = "geometry", occlusion_tolerance_m: float = 0.1)` | One sensor capture, including world objects composed by the framework. |
| `SensorView.composite_plan(objects: Sequence[PlacedObject]) -> CompositePlan` | Plan composing `objects` into this camera view exactly as simulator captures do. |
| `Initialization(expected: FrameData, resolved_scenario: Mapping[str, Any] = dict(), provenance: Mapping[str, Any] = dict())` | Source replay inputs only. No attack origin, score or placement enters the simulator. |
| `FrameDataSequence(frames: tuple[FrameData, ...], resolved_scenario: Mapping[str, Any], provenance: Mapping[str, Any])` | Detached generated frames and the resolved scenario and provenance that produced them. |
| `FrameDataSequence.at(frame: FrameReference) -> FrameData` | Return frame data at the requested recorded time. |
| `UnsupportedStartError(ValueError)` | A correctly configured simulator cannot start at this particular scenario start. |
| `Simulator(ABC)()` | A configured simulator: settings and declarations, without world state. |
| `Simulator.dataset_kinds: tuple[str, ...] = ()` | Dataset kinds supported by this simulator. |
| `Simulator.can_render: bool = False` | Whether arbitrary-view rendering is supported. |
| `Simulator.can_replay: bool = False` | Whether recorded-tick replay is supported. |
| `Simulator.can_observe_past: bool = False` | Whether observations before a selected start can be reconstructed. |
| `Simulator.can_insert_objects: bool = False` | Whether world objects can be inserted into sensor rendering. |
| `Simulator.supported_world_settings: frozenset[str] = frozenset()` | World-setting keys accepted by the simulator. |
| `Simulator.simulator_version = "1"` | Version identifier included in simulator compatibility summaries. |
| `Simulator.capture_ticks = 0` | Simulator ticks consumed by a sensor capture. |
| `Simulator.occlusion_tolerance_m: float = 0.1` | Depth tolerance in meters for deciding whether an inserted object is occluded. |
| `Simulator.create_runtime(seed: int \| None = None) -> SimulatorRuntime` | Abstract. Create a new live world with this configuration; `seed` seeds its random choices. |
| `Simulator.past_steps: int` | Recorded decision steps required before a selected start. |
| `Simulator.decision_time_step: float \| None` | Decision interval in seconds; None leaves the interval unrestricted. |
| `Simulator.command_modes: frozenset[str] \| None` | Command representations consumed by this simulator, or None if undeclared. |
| `Simulator.supported_stages: frozenset[str]` | Hook stages supported by the component. |
| `Simulator.identity: Mapping[str, Any]` | Configuration identity used to check saved-preparation compatibility. |
| `Simulator.validate_world_settings(settings: Mapping[str, Any]) -> None` | Check world-setting names and values without creating a live world. |
| `Simulator.validate_start(start: ScenarioStart \| None) -> None` | Reject a scenario start this simulator cannot reset to, before any world exists. |
| `SimulatorRuntime(ABC)(simulator: Simulator)` | One live world of a `Simulator`: world state, clock, actors, sensors and sensor views. |
| `SimulatorRuntime.dataset_kinds: tuple[str, ...]` | Dataset kinds supported by this simulator. |
| `SimulatorRuntime.can_render: bool` | Whether arbitrary-view rendering is supported. |
| `SimulatorRuntime.can_replay: bool` | Whether recorded-tick replay is supported. |
| `SimulatorRuntime.can_observe_past: bool` | Whether observations before a selected start can be reconstructed. |
| `SimulatorRuntime.can_insert_objects: bool` | Whether world objects can be inserted into sensor rendering. |
| `SimulatorRuntime.supported_world_settings: frozenset[str]` | World-setting keys accepted by the simulator. |
| `SimulatorRuntime.capture_ticks: int` | Simulator ticks consumed by a sensor capture. |
| `SimulatorRuntime.decision_time_step: float \| None` | Decision interval in seconds; None leaves the interval unrestricted. |
| `SimulatorRuntime.command_modes: frozenset[str] \| None` | Control command modes accepted or produced by this component. |
| `SimulatorRuntime.supported_stages: frozenset[str]` | Return the hook stages supported by this adapter. |
| `SimulatorRuntime.identity: Mapping[str, Any]` | Configuration identity used to check saved-preparation compatibility. |
| `SimulatorRuntime.termination_details: Mapping[str, Any]` | Simulator diagnostics saved with the run's termination event. |
| `SimulatorRuntime.validate_start(start: ScenarioStart \| None) -> None` | Check dataset and simulator constraints for a recorded scenario start. |
| `SimulatorRuntime.validate_command(command: Control) -> None` | Reject active command fields that this simulator would silently ignore. |
| `SimulatorRuntime.copy_observation(observation: Observation) -> Observation` | Detach the complete mutable data graph before model-visible processing. |
| `SimulatorRuntime.close() -> None` | Release the world and native connections. Optional. |
| `SimulatorRuntime.world_key: Hashable` | Identify the mutable world owned by this runtime. |
| `SimulatorRuntime.claim_world(owner: object) -> Iterator[None]` | Context manager. Claim the world for an owner until exit, allowing that owner's nested claims. |
| `SimulatorRuntime.exclusive() -> Iterator[None]` | Context manager. Claim the world and allow only one operation on this runtime at a time. |
| `SimulatorRuntime.clock: ClockSnapshot` | Return the current `ClockSnapshot` containing time and decision frame. |
| `SimulatorRuntime.state() -> WorldSnapshot` | Read detached physical state without a capture, callback or simulator tick. |
| `SimulatorRuntime.sensors: frozenset[str]` | All sensor IDs for which this runtime can provide a current SensorView. |
| `SimulatorRuntime.camera_sensors: frozenset[str]` | Explicit camera IDs whose current physical geometry can be recorded. |
| `SimulatorRuntime.sensor_geometry(sensor: str) -> SensorGeometry` | Read optical-camera geometry; override for a lightweight, image-free path. |
| `SimulatorRuntime.sensor_view(sensor: str) -> SensorView` | Return the view of the current capture of `sensor`; it never advances time. |
| `SimulatorRuntime.insert_object(object_id: str, asset: ObjectAsset, world_from_object: np.ndarray, *, harmonizer: Harmonizer \| None = None) -> None` | Register a physical world object, visible from the next sensor capture. |
| `SimulatorRuntime.remove_object(object_id: str) -> None` | Remove a world object; the previous capture remains unchanged. |
| `SimulatorRuntime.object_visibility(object_id: str) -> Mapping[str, ObjectVisibility]` | Visibility of an object in the latest capture, for every camera whose image contains it. |
| `SimulatorRuntime.control_actor(actor_id: str, control: Control) -> None` | Drive another actor with an actuator command from the next world step. |
| `SimulatorRuntime.set_actor_state(actor_id: str, state: ActorState) -> None` | Move another actor to a canonical world state; visible from the next capture. |
| `SimulatorRuntime.reset(start: ScenarioStart \| None = None, initialization: Any = None, *, deployment: DeploymentPlan \| None = None, world: Mapping[str, Any] \| None = None) -> None` | Start a run at `start` or the configured origin and capture the first frame. |
| `SimulatorRuntime.setup_tick() -> None` | Advance one setup tick without requesting a driving decision. |
| `SimulatorRuntime.step(control: Control) -> None` | Apply `control`, advance the world one decision and capture; read it with `observe`. |
| `SimulatorRuntime.observe() -> Observation` | Return an independent copy of the current capture; never advance or recapture. |
| `SimulatorRuntime.observe_past(count: int) -> tuple[Observation, ...]` | Read recorded observations before the selected start, from oldest to newest. |
| `SimulatorRuntime.start_replay(start: ScenarioStart, initialization: Initialization \| None = None, *, world: Mapping[str, Any] \| None = None, deployment: DeploymentPlan \| None = None) -> None` | Reset for reading prescribed motion, without hooks or a driving system. |
| `SimulatorRuntime.advance_replay(count: int) -> None` | Advance prescribed motion and capture each frame, without hooks or a driver. |
| `SimulatorRuntime.generate_frame_data(start: ScenarioStart, frames: Sequence[FrameReference], *, world: Mapping[str, Any] \| None = None, deployment: DeploymentPlan \| None = None) -> FrameDataSequence` | Record actual frame data by replaying `start`'s prescribed motion over `frames`. |
| `SimulatorRuntime.native: Mapping[str, Any]` | Expose native components available to callbacks, keyed by component name. |
| `SimulatorRuntime.bind_hooks(dispatcher: StageEmitter) -> Iterator[None]` | Context manager. Bind one dispatcher for the scope; reject an already active binding. |
<!-- api:end -->

#### Constraints and errors

`validate_start` raises `TypeError` for a non-`ScenarioStart` and `ValueError` for an
incompatible dataset kind or decision interval. A supported configuration whose particular start
lacks required support raises `UnsupportedStartError`. Past captures and world ownership follow
[Simulators](../guides/simulators.md)
and [Architecture](../concepts/architecture.md#world-ownership).

#### Extension contract

Implement `Simulator.create_runtime(seed)` and declare dataset kinds, capabilities, stages,
decision timing and command modes on the configuration class.
Runtime subclasses must implement
`_initialize_world(start: ScenarioStart | None, initialization: Any, environment: Mapping[str, Any]) -> None`,
`_advance_native(control)`, `_observe()` and `_state()`.
Optional native lifecycle methods are `_sensor_view`, `_replay_tick`, `_observe_recorded`,
`_recorded_sensor_geometry`, `_recorded_sensor_view`, `_capture_frame_data`, `_hold_tick`,
`_replay_position`, `_check_initialization`, `_before_selected_capture`, `_replayed_scenario`
and `_frame_data_provenance`. The default `_hold_tick` raises `ValueError` when deployment
ticks require holding an origin. Overriding
`reset`, `step`, `sensor_view`, `start_replay`, `advance_replay` or `observe_past` raises
`TypeError`. `MemorySimulator` in [tests/fakes.py](../../tests/fakes.py) is a
minimal implementation.

The default `SimulatorRuntime.world_key` is `("simulator", id(self))`. For optical camera views,
`sensor_from_world = BODY_FROM_OPTICAL @ cam_from_world`. `setup_tick` raises `RuntimeError`
outside `simulator.setup` or after the deployment tick budget is exhausted.

#### See also

[Simulators](../guides/simulators.md).

### avsectester.closed_loop.simulators.carla

<!-- api:begin module=avsectester.closed_loop.simulators.carla -->
CARLA simulator configuration and live-world runtime.

| Name and signature | Meaning |
|---|---|
| `CarlaSimulator` | [CarlaSimulator](#avsectesterclosed_loopsimulatorscarlasimulator) |
| `CarlaSimulatorRuntime` | [CarlaSimulatorRuntime](#avsectesterclosed_loopsimulatorscarlasimulator) |
<!-- api:end -->

#### See also

[Simulators](../guides/simulators.md).

### avsectester.closed_loop.simulators.carla.conversions

<!-- api:begin module=avsectester.closed_loop.simulators.carla.conversions -->
Convert native CARLA actors, signal states and rotations to canonical world records.

| Name and signature | Meaning |
|---|---|
| `rotation_to_carla(rotation: ArrayLike) -> tuple[float, float, float]` | Return native CARLA (pitch, yaw, roll) degrees from a native rotation matrix. |
<!-- api:end -->

#### Constraints and errors

`rotation_to_carla` converts a native CARLA rotation matrix to pitch/yaw/roll.
Callers apply canonical Y-axis reflection before calling it.

#### See also

[Simulators](../guides/simulators.md).

### avsectester.closed_loop.simulators.carla.sensors

<!-- api:begin module=avsectester.closed_loop.simulators.carla.sensors -->
CARLA sensor alignment, pinhole registration, depth decoding and owned SDK captures.

| Name and signature | Meaning |
|---|---|
| `SensorCapture(native_type: str, raw_bytes: bytes, frame: int, timestamp: float, metadata: dict[str, Any] = dict())` | Owned bytes from a CARLA sensor capture; contains no simulator handle. |
| `SensorCapture.raw_data: memoryview` | Native sensor payload before conversion to framework records. |
| `SensorAlignmentError(RuntimeError)` | A configured sensor has no capture for the current world frame. |
| `decode_carla_depth(image)` | Decode CARLA's 24-bit depth image to forward camera depth in meters. |
<!-- api:end -->

#### Constraints and errors

`decode_carla_depth` converts CARLA's encoded depth bytes to metric optical-axis depth.
The pinhole provider raises `ValueError` for nonzero distortion attributes.

#### See also

[Simulators](../guides/simulators.md).

### avsectester.closed_loop.simulators.carla.simulator

<!-- api:begin module=avsectester.closed_loop.simulators.carla.simulator -->
Configured CARLA simulations and their live native-world runtimes.

| Name and signature | Meaning |
|---|---|
| `CarlaSimulator(Simulator)(client: dict, settle_iters: int = 100, *, simulator_version="0.9.16", depth_camera: str \| None = None, occlusion_tolerance_m: float = 0.1)` | CARLA runtime settings, with scene content supplied by a dataset recipe. |
| `CarlaSimulator.dataset_kinds = ("CarlaDataset",)` | Dataset kinds supported by this simulator. |
| `CarlaSimulator.capture_ticks = 1` | Simulator ticks consumed by a sensor capture. |
| `CarlaSimulator.can_render = True` | Whether arbitrary-view rendering is supported. |
| `CarlaSimulator.can_replay = True` | Whether recorded-tick replay is supported. |
| `CarlaSimulator.can_insert_objects = True` | Whether world objects can be inserted into sensor rendering. |
| `CarlaSimulator.supported_world_settings = CARLA_WORLD_SETTINGS` | World-setting keys accepted by the simulator. |
| `CarlaSimulator.validate_world_settings(world: Mapping[str, Any]) -> None` | Validate keys, native enum names, types and ranges without connecting to CARLA. |
| `CarlaSimulator.create_runtime(seed: int \| None = None) -> CarlaSimulatorRuntime` | A new live world; an explicit run seed overrides the configured client seed. |
| `CarlaSimulatorRuntime(SimulatorRuntime)(simulator: CarlaSimulator, seed: int \| None = None)` | One live CARLA world of a `CarlaSimulator`. |
| `CarlaSimulatorRuntime.native: dict[str, Any]` | Expose live simulator resources for explicit world-level interventions. |
| `CarlaSimulatorRuntime.copy_observation(observation: Observation) -> Observation` | Detach mutable payloads, calibration and reference graphs in one copy operation. |
| `CarlaSimulatorRuntime.supported_stages: frozenset[str]` | Return the hook stages supported by this adapter. |
| `CarlaSimulatorRuntime.bind_hooks(dispatcher: StageEmitter) -> Iterator[None]` | Context manager. Bind one dispatcher for the scope; reject an already active binding. |
<!-- api:end -->

#### Constraints and errors

CARLA SDK/server dependencies belong to this implementation. World claims cover the full
operation; native actor handles are not persisted in `RunLog`.

#### See also

[Simulators](../guides/simulators.md).

### avsectester.closed_loop.simulators.ego_dynamics

<!-- api:begin module=avsectester.closed_loop.simulators.ego_dynamics -->
In-process ego motion from actuator or rig-frame trajectory commands.

| Name and signature | Meaning |
|---|---|
| `PlanarEgoState(x: float = 0.0, y: float = 0.0, yaw: float = 0.0, speed: float = 0.0, t: float = 0.0, lateral_speed: float = 0.0, yaw_rate: float = 0.0, acceleration: float = 0.0)` | Planar physical ego state in meters, radians and seconds. |
| `EgoDynamics(Protocol)` | A custom ego-motion implementation, with an optional `command_modes` declaration. |
| `EgoDynamics.reset(state: PlanarEgoState) -> None` | Optionally initialize integrator state at the recorded start. |
| `EgoDynamics.step(pose: PlanarEgoState, control: Control, dt: float) -> PlanarEgoState` | Advance the physical ego state by `dt` seconds using the final command. |
| `KinematicBicycle(wheelbase: float = 2.8, max_accel: float = 3.0, max_brake: float = 8.0, max_steer: float = 0.6)` | Integrate actuator commands with a deterministic kinematic bicycle model. |
| `KinematicBicycle.command_modes: ClassVar[frozenset[str]] = frozenset({"actuator"})` | Control command modes accepted or produced by this component. |
| `KinematicBicycle.reset(state: PlanarEgoState) -> None` | No warmup is needed: this integrator carries all state in its argument. |
| `KinematicBicycle.step(pose: PlanarEgoState, control: Control, dt: float) -> PlanarEgoState` | Integrate actuator commands for one time interval using bicycle dynamics. |
| `TrajectoryFollower(decel: float = 4.0)` | Realize a rig-frame planned trajectory by direct position interpolation. |
| `TrajectoryFollower.command_modes: ClassVar[frozenset[str]] = frozenset({"trajectory"})` | Control command modes accepted or produced by this component. |
| `TrajectoryFollower.reset(state: PlanarEgoState) -> None` | No warmup is needed: the provided state anchors each planned trajectory. |
| `TrajectoryFollower.step(pose: PlanarEgoState, control: Control, dt: float) -> PlanarEgoState` | Track the requested trajectory for one time interval. |
| `AlpaSimController(mpc_implementation: str = "linear", n_horizon: int = 20, dt_mpc: float = .1, gains: Mapping[str, float \| int] \| None = None)` | AlpaSim's in-process controller and dynamic bicycle model. |
| `AlpaSimController.command_modes: ClassVar[frozenset[str]] = frozenset({"trajectory"})` | Control command modes accepted or produced by this component. |
| `AlpaSimController.EPOCH_US: ClassVar[int] = 10_000_000` | Timestamp offset in microseconds used by the trajectory interface. |
| `AlpaSimController.reset(state: PlanarEgoState) -> None` | Create a fresh controller at the supplied recorded physical state. |
| `AlpaSimController.step(pose: PlanarEgoState, control: Control, dt: float) -> PlanarEgoState` | Track the t0-led rig-frame plan and return the controller's final physical state. |
<!-- api:end -->

#### Constraints and errors

`AlpaSimController` requires position/quaternion trajectory points with strictly increasing
integer-microsecond timestamps and a rig-frame origin at the current step time.
Optional MPC dependencies are imported lazily.

`AlpaSimController.gains` must be a mapping whose keys name `MPCGains` fields. A step whose
state time differs from the latest propagated time raises `ValueError`. `PlanarEgoState.x`, `y`
and `yaw` locate the ego origin in world coordinates; `speed` and `lateral_speed` are ego-axis
velocities, `yaw_rate` is radians per second, `acceleration` is longitudinal meters per second
squared, and `t` is seconds.

#### See also

[Simulators](../guides/simulators.md).

### avsectester.closed_loop.simulators.ground_contact

<!-- api:begin module=avsectester.closed_loop.simulators.ground_contact -->
AlpaSim-style ego ground contact using CPU mesh-ray intersections.

| Name and signature | Meaning |
|---|---|
| `constrain_to_ground(raycaster: MeshRaycaster, world_from_box: np.ndarray, extent: tuple[float, float, float]) -> tuple[np.ndarray, str]` | Fit a vehicle's bottom to nearby ground, preserving its predicted center's XY. |
<!-- api:end -->

#### Constraints and errors

`world_from_box` must be a finite rigid transform and `extent` must contain three positive
finite box dimensions. A 4×4 bottom grid is queried vertically in both directions within 100
meters. At least six intersections are required. Corrections greater than 1.5 meters or 10
degrees are rejected. Status is `successful_update`, `insufficient_intersections`,
`high_translation` or `high_rotation`; rejected updates retain the input pose.

#### See also

[Simulators](../guides/simulators.md).

### avsectester.closed_loop.simulators.navigation

<!-- api:begin module=avsectester.closed_loop.simulators.navigation -->
Navigation intentions derived from a scenario's map or recorded ego trajectory.

| Name and signature | Meaning |
|---|---|
| `RouteGenerator(kind: str, recorded_points: Sequence[Sequence[float]] \| np.ndarray, scenario_map: ScenarioMap \| Unavailable \| None)` | Generate 20 ego-frame navigation points spaced `80 / 19` meters apart. |
| `RouteGenerator.route(world_from_ego: np.ndarray) -> np.ndarray \| None` | Return forward navigation from the actual ego projection, in ego axes, with z=0. |
<!-- api:end -->

#### Constraints and errors

Routes require at least two finite recorded world positions. A reversal exceeding 150 degrees
raises `ValueError`. Map navigation requires an available map containing driving lanes and
rejects recorded waypoints more than 10 meters from the derived map route. Use `recorded`
navigation when only recorded motion is available, or `none` to disable routes.

#### See also

[Simulators](../guides/simulators.md).

### avsectester.closed_loop.simulators.nurec

<!-- api:begin module=avsectester.closed_loop.simulators.nurec -->
NuRec simulator configuration, live runtime and RPC renderer.

| Name and signature | Meaning |
|---|---|
| `AlpaSimTraffic` | [AlpaSimTraffic](#avsectesterclosed_loopsimulatorsnurectraffic) |
| `TrafficSession` | [TrafficSession](#avsectesterclosed_loopsimulatorsnurectraffic) |
| `TrafficTrack` | [TrafficTrack](#avsectesterclosed_loopsimulatorsnurectraffic) |
| `NuRecSimulator` | [NuRecSimulator](#avsectesterclosed_loopsimulatorsnurecsimulator) |
| `NuRecSimulatorRuntime` | [NuRecSimulatorRuntime](#avsectesterclosed_loopsimulatorsnurecsimulator) |
| `NuRecRenderer` | [NuRecRenderer](#avsectesterclosed_loopsimulatorsnurecrenderer) |
<!-- api:end -->

#### See also

[Simulators](../guides/simulators.md).

### avsectester.closed_loop.simulators.nurec.renderer

<!-- api:begin module=avsectester.closed_loop.simulators.nurec.renderer -->
NuRec RGB rendering and native camera/pose conversion at the RPC boundary.

| Name and signature | Meaning |
|---|---|
| `Renderer(ABC)()` | Renders sensor observations from the reconstructed scene at a given ego pose. |
| `Renderer.cameras: ClassVar[list[str]] = ["camera_front"]` | Camera names rendered by this renderer. |
| `Renderer.calibration: dict[str, CameraCalibration]` | Camera calibrations supplied by a loaded renderer; empty by default. |
| `Renderer.copy() -> Renderer` | A renderer with this configuration and no connection, loaded scene or run origin. |
| `Renderer.reset_origin(start=None)` | Set the run origin from the dataset start; configured renderers have no origin. |
| `Renderer.origin_transform` | World transform locating the loaded scene origin. |
| `Renderer.origin_timestamp_us` | Recorded timestamp in microseconds at the scene origin. |
| `Renderer.origin_bounds` | Spatial bounds of the loaded scene origin. |
| `Renderer.load_scene(scene: Any) -> None` | Load a reconstruction once; renderers without a service may leave this empty. |
| `Renderer.render(world_from_rig: np.ndarray, timestamp_us: int, camera: str, dynamic_objects=()) -> Any` | Abstract. Render one camera at the explicit three-dimensional rig pose and scene timestamp. |
| `Renderer.render_scene_visibility(world_from_rig: np.ndarray, timestamp_us: int, camera: str, objects: Sequence[PlacedObject], dynamic_objects=(), *, occlusion_tolerance_m: float = 0.) -> np.ndarray` | Pixels where the reconstructed scene does not hide `objects` in this capture. |
| `Renderer.camera_model(camera: str \| None = None) -> Camera \| None` | Optical camera model, or None when this renderer supplies no geometry. |
| `Renderer.cam_from_world(world_from_rig: np.ndarray, camera: str \| None = None) -> np.ndarray` | World-to-optical-camera transform for the rendered pose. |
| `Renderer.close() -> None` | Release renderer resources; the default implementation has no effect. |
| `NuRecRenderer(Renderer)(endpoint: str = "127.0.0.1:50051", scene_id: str \| None = None, cameras: list[str] = ["camera_front_wide_120fov"], image_quality: float = 95.0, resolution_hw: Mapping[str, tuple[int, int]] = dict(), probe_directory: str \| Path \| None = None, server_probe_directory: str \| None = None, track_padding_m: tuple[float, float, float] = (1.0, 1.0, 0.25))` | Real NuRec renderer: one stateless `SensorsimService.render_rgb` per frame to an nre server. |
| `NuRecRenderer.copy() -> NuRecRenderer` | The same configuration without a connection, loaded scene or run origin. |
| `NuRecRenderer.served_scenes(timeout_s: float = 2.0) -> tuple[str, ...] \| None` | Query supported scene IDs without changing this renderer's loaded scene. |
| `NuRecRenderer.load_scene(scene: Any) -> None` | Connect to the nre-ga renderer and read the scene id, camera spec, and start pose/time. |
| `NuRecRenderer.calibration: dict[str, CameraCalibration]` | Loaded camera lenses and optical-camera-to-rig poses; empty before loading. |
| `NuRecRenderer.service` | Native SensorsimService client, including metadata APIs not normalized by this adapter. |
| `NuRecRenderer.camera_model(camera: str \| None = None) -> Camera` | The lens scaled to the image dimensions actually returned by NRE during load_scene. |
| `NuRecRenderer.cam_from_world(world_from_rig: np.ndarray, camera: str \| None = None) -> np.ndarray` | World-to-optical-camera transform at the explicit three-dimensional rig pose. |
| `NuRecRenderer.render(world_from_rig: np.ndarray, timestamp_us: int, camera: str, dynamic_objects=())` | Render one RGB frame at an explicit three-dimensional rig pose and scene timestamp. |
| `NuRecRenderer.render_at(world_from_rig, timestamp_us: int, camera: str, dynamic_objects=())` | Render an absolute rig pose/time and all present track-ID/world-pose pairs. |
| `NuRecRenderer.render_scene_visibility(world_from_rig: np.ndarray, timestamp_us: int, camera: str, objects: Sequence[PlacedObject], dynamic_objects=(), *, occlusion_tolerance_m: float = 0.) -> np.ndarray` | Pixels where the reconstructed scene does not hide `objects` in an NRE render. |
<!-- api:end -->

#### Constraints and errors

`load_scene` establishes calibration from the same native extrinsics as `cam_from_world`.
`served_scenes` returns None only for UNAVAILABLE or `DEADLINE_EXCEEDED`; other RPC errors
propagate. Probe paths follow client/server configuration rules.

#### See also

[Simulators](../guides/simulators.md).

### avsectester.closed_loop.simulators.nurec.server

<!-- api:begin module=avsectester.closed_loop.simulators.nurec.server -->
Build, start, reuse and stop local NuRec renderer and traffic-service containers.

| Name and signature | Meaning |
|---|---|
| `local_services(dataset: Dataset, simulator: Simulator, *, nre_gpu: str \| int \| None = None, nre_name: str \| None = None, trafficsim_gpu: str \| int \| None = None, trafficsim_models: str \| Path \| None = None, trafficsim_name: str \| None = None, timeout_s: float = 600., log: Callable[[str], Any] = print) -> Iterator[None]` | Context manager. Reuse compatible services or own local NRE/CATK containers for one experiment. |
<!-- api:end -->

#### Constraints and errors

Commands mount client paths as host paths. Automatic startup requires a local endpoint and an
unambiguous GPU; an existing compatible service is reused. `local_services` stops only
containers it starts, in reverse order. Startup failures include recent logs and clean up
owned containers.

#### See also

[Simulators](../guides/simulators.md).

### avsectester.closed_loop.simulators.nurec.simulator

<!-- api:begin module=avsectester.closed_loop.simulators.nurec.simulator -->
NuRec live worlds with independent renderer, actor state and ego dynamics.

| Name and signature | Meaning |
|---|---|
| `NuRecSimulator(Simulator)(renderer: Renderer, dynamics: EgoDynamics \| None = None, *, time_step: float \| None = None, camera_aliases: Mapping[str, str] \| None = None, depth: str = "mesh", route_generator: str = "map", physics_update_mode: str = "ego_only", ground_mesh: str = "mesh", object_occlusion: str = "rendered_probes", occlusion_tolerance_m: float = 0.15, traffic: AlpaSimTraffic \| None = None, min_track_duration_s: float = 3.)` | A NuRec neural reconstruction with in-process ego dynamics and a pluggable renderer. |
| `NuRecSimulator.dataset_kinds = ("NuRecDataset",)` | Dataset kinds supported by this simulator. |
| `NuRecSimulator.can_render = True` | Whether arbitrary-view rendering is supported. |
| `NuRecSimulator.can_observe_past = True` | Whether observations before a selected start can be reconstructed. |
| `NuRecSimulator.can_insert_objects = True` | Whether world objects can be inserted into sensor rendering. |
| `NuRecSimulator.create_runtime(seed: int \| None = None) -> NuRecSimulatorRuntime` | A new world; the optional traffic session receives seed for its log. |
| `NuRecSimulator.past_steps: int` | Recorded decision history needed by the optional traffic model. |
| `NuRecSimulator.command_modes: frozenset[str] \| None` | Delegate command capabilities to the selected ego dynamics implementation. |
| `NuRecSimulatorRuntime(SimulatorRuntime)(simulator: NuRecSimulator, seed: int \| None = None)` | One live NuRec world: the ego pose and dynamics loop, and its own renderer connection. |
| `NuRecSimulatorRuntime.native: dict[str, Any]` | Native resources for explicit world changes or custom renderer operations. |
| `NuRecSimulatorRuntime.render_request(camera: str) -> CameraPoseRequest` | Return the effective request that produced this camera's current observation. |
| `NuRecSimulatorRuntime.set_actor_state(actor_id: str, state: ActorState) -> None` | Override a current reconstructed actor's pose and world velocity until reset. |
| `NuRecSimulatorRuntime.checkpoint() -> dict` | Capture a replay checkpoint; active reactive-traffic sessions cannot be checkpointed. |
| `NuRecSimulatorRuntime.restore(ckpt: dict) -> None` | Restore a saved replay checkpoint into this runtime. |
| `NuRecSimulatorRuntime.supported_stages: frozenset[str]` | Return the hook stages supported by this adapter. |
| `NuRecSimulatorRuntime.bind_hooks(dispatcher: StageEmitter) -> Iterator[None]` | Context manager. Bind one dispatcher for the scope; reject an already active binding. |
<!-- api:end -->

#### Constraints and errors

NuRec requires `ScenarioStart` and a decision `time_step`. In `rendered_probes` mode, capture
with the objects inserted, then use `view.composition.plan`; `SensorView.composite_plan` raises
`ValueError` for these views. Ground contact requires the selected mesh unless
`physics_update_mode` is none.

#### See also

[Simulators](../guides/simulators.md).

### avsectester.closed_loop.simulators.nurec.traffic

<!-- api:begin module=avsectester.closed_loop.simulators.nurec.traffic -->
AlpaSim CATK configuration, stateless RPC and owned live traffic sessions.

| Name and signature | Meaning |
|---|---|
| `TrafficTrack(track_id: str, trajectory: tuple[tuple[int, np.ndarray], ...], extent: tuple[float, float, float], label_class: str, is_static: bool)` | A recorded box-center trajectory and its native dimensions and class. |
| `AlpaSimTraffic(endpoint: str = "127.0.0.1:6200", history_steps: int = 16, rpc_timeout_s: float = 300.)` | Configure stateless RPC and open owned CATK sessions; seed is only logged. |
| `AlpaSimTraffic.get_metadata(*, timeout_s: float \| None = None) -> Any` | Fetch TrafficModuleMetadata over a short-lived connection. |
| `AlpaSimTraffic.get_available_scenes(*, timeout_s: float \| None = None) -> tuple[str, ...]` | Return the scene IDs served by this endpoint. |
| `AlpaSimTraffic.served_scenes(timeout_s: float = 2.0) -> tuple[str, ...] \| None` | Query scene support; unavailable endpoints and deadline expiry return None. |
| `AlpaSimTraffic.open_session(*, scene_id: str, random_seed: int, start_us: int, end_us: int, ego_extent: tuple[float, float, float], ego_trajectory: Sequence[tuple[int, np.ndarray]], tracks: Sequence[TrafficTrack]) -> TrafficSession` | Validate support and open a session with EGO first and clipped native tracks. |
| `TrafficSession(channel: Any, session_uuid: str, handover_time_us: int, tracks: Sequence[TrafficTrack], *, history_steps: int, rpc_timeout_s: float)` | One live CATK session, its channel and complete constant-velocity override history. |
| `TrafficSession.record_override(track_id: str, time_us: int, world_from_box: np.ndarray, velocity: Sequence[float]) -> None` | Record an override, backfilling only its first constant-velocity history. |
| `TrafficSession.simulate(time_query_us: int, ego_world_from_box: np.ndarray, override_world_from_box: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]` | Send current EGO and full override histories; return current movable poses. |
| `TrafficSession.close() -> None` | Close once and release the channel even when the close RPC fails. |
<!-- api:end -->

#### Constraints and errors

A session checks scene availability and the configured history length before starting. The first
trajectory is `EGO`; track trajectories are clipped to the history-supported recording window
and empty tracks are omitted. Handover is one traffic step after `start_us`. `record_override`
backfills the initial constant-velocity history; `simulate` requires override keys to match
recorded override tracks exactly. Returned poses exclude static, overridden and unknown tracks.
Missing requested timestamps or RPC failures close the session before propagating the error.
RPC failures raise `RuntimeError` containing the RPC status code and details.

#### See also

[Simulators](../guides/simulators.md).

### avsectester.closed_loop.simulators.recorded

<!-- api:begin module=avsectester.closed_loop.simulators.recorded -->
Bounded open-loop replay of recorded frames.

| Name and signature | Meaning |
|---|---|
| `RecordedReplaySimulator(Simulator)(start: ScenarioStart \| None = None, *, camera_aliases: Mapping[str, str] = {}, occlusion_tolerance_m: float = 0.1)` | Open-loop replay with native camera names or explicitly configured output aliases. |
| `RecordedReplaySimulator.dataset_kinds = ("NuScenesDataset",)` | Dataset kinds supported by this simulator. |
| `RecordedReplaySimulator.can_render = True` | Whether arbitrary-view rendering is supported. |
| `RecordedReplaySimulator.can_replay = True` | Whether recorded-tick replay is supported. |
| `RecordedReplaySimulator.can_insert_objects = True` | Whether world objects can be inserted into sensor rendering. |
| `RecordedReplaySimulator.create_runtime(seed: int \| None = None) -> RecordedReplaySimulatorRuntime` | A new replay; recorded frames are fixed, so `seed` is unused. |
| `RecordedReplaySimulatorRuntime(SimulatorRuntime)(simulator: RecordedReplaySimulator)` | One replay of recorded frames. |
| `RecordedReplaySimulatorRuntime.supported_stages: frozenset[str]` | Return the hook stages supported by this adapter. |
| `RecordedReplaySimulatorRuntime.native: Mapping[str, Any]` | Expose native components available to callbacks, keyed by component name. |
| `RecordedReplaySimulatorRuntime.bind_hooks(dispatcher: StageEmitter) -> Iterator[None]` | Context manager. Bind one dispatcher for the scope; reject an already active binding. |
<!-- api:end -->

#### Constraints and errors

Reset requires a nuScenes `ScenarioStart`, no environment overrides and no CARLA replay
initialization. Frames must begin at the requested start, be contiguous and contain ego state at
every frame. Duplicate aliased sensor names, unknown or uncalibrated cameras, absent RGB
captures and steps beyond terminal support raise `ValueError`. State or geometry access before
reset raises `RuntimeError`.

#### See also

[Simulators](../guides/simulators.md).

### avsectester.attack

<!-- api:begin module=avsectester.attack -->
Attack contracts, runtime actions and structured offline reports.

| Name and signature | Meaning |
|---|---|
| `AVStackAttackAction` | [AVStackAttackAction](#avsectesterattackruntime) |
| `Attack` | [Attack](#avsectesterattackattack) |
| `AttackAction` | [AttackAction](#avsectesterattackruntime) |
| `AttackRuntime` | [AttackRuntime](#avsectesterattackruntime) |
| `AttackSegment` | [AttackSegment](#avsectesterattackattack) |
| `OfflineContext` | [OfflineContext](#avsectesterattackattack) |
| `OfflineReport` | [OfflineReport](#avsectesterattackoffline_report) |
| `ReportValue` | [ReportValue](#avsectesterattackoffline_report) |
| `ReportSeries` | [ReportSeries](#avsectesterattackoffline_report) |
| `ReportTable` | [ReportTable](#avsectesterattackoffline_report) |
| `ReportImage` | [ReportImage](#avsectesterattackoffline_report) |
| `OnlineContext` | [OnlineContext](#avsectesterattackruntime) |
| `ResultSubscription` | [ResultSubscription](#avsectesterattackruntime) |
| `SensorAttackAction` | [SensorAttackAction](#avsectesterattackruntime) |
| `WorldAttackAction` | [WorldAttackAction](#avsectesterattackruntime) |
| `split_segments` | [split_segments](#avsectesterattacksegments) |
<!-- api:end -->

#### See also

[Implement an attack](../guides/attacks.md).

### avsectester.attack.attack

<!-- api:begin module=avsectester.attack.attack -->
The user's attack: the metabuilder, the offline phase and the online phase.

| Name and signature | Meaning |
|---|---|
| `AttackSegment(scenario_id: str, start: FrameReference, stop: FrameReference, user_data: Mapping[str, Any] = dict())` | One selected scenario and decision interval with attack-owned data. |
| `AttackSegment.decision_steps: int` | Number of decisions between the inclusive start and exclusive stop references. |
| `OfflineContext(services: ClosedLoopServices, metadata: tuple[AttackSegment, ...], seed: int = 0, environment: Mapping[str, Any] = dict(), deployment: DeploymentPlan \| None = None, validation_metadata: tuple[AttackSegment, ...] = (), report: OfflineReport = OfflineReport(None))` | What `run_offline` receives: the closed-loop services and the attack metadata. |
| `Attack(ABC)()` | One attack with its own parameters and optional offline preparation. |
| `Attack.environment: Mapping[str, Any] = MappingProxyType({})` | Environment settings merged with experiment settings during selection and the offline phase. |
| `Attack.build_metadata(scenario: ScenarioView) -> Sequence[AttackSegment]` | Select segments, or leave selection to a shared metadata.load file. |
| `Attack.validate_segment(segment: AttackSegment) -> None` | Accept shared metadata, or raise ValueError explaining why it is unsuitable. |
| `Attack.run_offline(context: OfflineContext) -> Any` | Return the artifacts used by the online phase, or None when no preparation is needed. |
| `Attack.deployment_ticks(segment: AttackSegment) -> int` | Declare native setup ticks before preparing artifacts; default has no world deployment. |
| `Attack.create_runtime(segment: AttackSegment, artifacts: Any) -> AttackRuntime` | Create the runtime for one run. By default, a runtime without attack actions. |
| `Attack.configuration_parameters() -> Mapping[str, Any]` | Describe the constructor parameters used to validate saved preparations. |
| `Attack.save_artifacts(artifacts: Any, directory: str \| Path) -> None` | Save the explicit `artifacts` value as pickle plus its content hash. |
| `Attack.load_artifacts(directory: str \| Path) -> Any` | Return saved artifacts without changing this attack's configuration or state. |
<!-- api:end -->

#### Constraints and errors

`AttackSegment` requires matching scenario IDs on both frame references, an exclusive stop
after the start, and mapping-valued `user_data`. Invalid intervals raise `ValueError`;
non-mapping `user_data` raises `TypeError` at construction. The default `Attack.build_metadata` raises
`NotImplementedError` and requires `metadata.load` for experiments.

#### See also

[Implement an attack](../guides/attacks.md).

### avsectester.attack.insertion

<!-- api:begin module=avsectester.attack.insertion -->
Public assets, calibrated compositing, visibility and actions for user attacks.

| Name and signature | Meaning |
|---|---|
| `Insertion` | [Insertion](#avsectesterattackinsertionassets) |
| `PlaneAsset` | [PlaneAsset](#avsectesterattackinsertionassets) |
| `PlanarAsset` | [PlanarAsset](#avsectesterattackinsertionassets) |
| `ResolvedInsertion` | [ResolvedInsertion](#avsectesterattackinsertionassets) |
| `resolve_insertion` | [resolve_insertion](#avsectesterattackinsertionassets) |
| `PatchCompositor` | [PatchCompositor](#avsectesterattackinsertioncompositor) |
| `render_resolved` | [render_resolved](#avsectesterattackinsertioncompositor) |
| `InsertObject` | [InsertObject](#avsectesterattackinsertionaction) |
| `InsertionAttackAction` | [InsertionAttackAction](#avsectesterattackinsertionaction) |
| `Visibility` | [Visibility](#avsectesterattackinsertiontypes) |
| `VisibilityEvidence` | [VisibilityEvidence](#avsectesterattackinsertiontypes) |
| `CuboidVisibilityEstimator` | [CuboidVisibilityEstimator](#avsectesterattackinsertionvisibility) |
| `DepthVisibilityEstimator` | [DepthVisibilityEstimator](#avsectesterattackinsertionvisibility) |
| `visibility_from_depth` | [visibility_from_depth](#avsectesterattackinsertionvisibility) |
| `visibility_from_masks` | [visibility_from_masks](#avsectesterattackinsertionvisibility) |
<!-- api:end -->

#### See also

[Implement an attack](../guides/attacks.md).

### avsectester.attack.insertion.action

<!-- api:begin module=avsectester.attack.insertion.action -->
World-object and captured-image insertion actions for user attacks.

| Name and signature | Meaning |
|---|---|
| `InsertObject(WorldAttackAction)(object_id: str, asset: ObjectAsset, world_from_object: np.ndarray, harmonizer: Harmonizer \| None = None)` | Insert arbitrary textured geometry into all cameras through the simulator. |
| `InsertionAttackAction(SensorAttackAction)(sensor: str, insertion: Insertion, compositor: PatchCompositor \| None = None)` | Render explicit planes with aligned depth when present, otherwise actor cuboids. |
| `InsertionAttackAction.resolve(view)` | Resolve the insertion asset, pose and placement for the current sensor geometry. |
| `InsertionAttackAction.evidence(view, resolved=None)` | Return insertion placement and visibility evidence for the action record. |
<!-- api:end -->

#### Constraints and errors

`Insertion` is a specification. `InsertObject` is a `WorldAttackAction` that registers cleanup;
`InsertionAttackAction` is a `SensorAttackAction`. `InsertObject` raises `NotImplementedError`
when the simulator does not declare `can_insert_objects`.

#### See also

[Implement an attack](../guides/attacks.md).

### avsectester.attack.insertion.assets

<!-- api:begin module=avsectester.attack.insertion.assets -->
Explicit insertion geometry, independent of a simulator or attack implementation.

| Name and signature | Meaning |
|---|---|
| `PlanarAsset(Protocol)` | Extension point for assets composed of local textured rectangular surfaces. |
| `PlanarAsset.planes() -> Sequence[PlaneSurface]` | Resolve the asset into textured planes in its local coordinate frame. |
| `PlaneAsset(texture: np.ndarray \| str \| Path, width_m: float, height_m: float)` | RGBA patch or sign, centered at the origin in the local YZ plane. |
| `PlaneAsset.planes() -> tuple[PlaneSurface, ...]` | Resolve the asset into textured planes in its local coordinate frame. |
| `Insertion(id: str, asset: PlanarAsset, placement: AttachedPlacement \| WorldPlacement, orientation: Orientation = Orientation())` | An independently configured object. No placement coordinates are searched or adjusted. |
| `ResolvedInsertion(id: str, pose: np.ndarray, asset: PlanarAsset, host_id: str \| None = None)` | An insertion's world pose at one instant, shared by rendering and selection. |
| `ResolvedInsertion.planes() -> tuple[PlaneSurface, ...]` | Resolve the asset into textured planes in its local coordinate frame. |
| `resolve_insertion(insertion: Insertion, actors: Mapping[str, ActorState], ego: ActorState) -> ResolvedInsertion` | Resolve a user-specified installation against current actor poses without mutation. |
<!-- api:end -->

#### Constraints and errors

Attached placement requires known actor dimensions. Unknown extent is not treated as zero.
Placement resolves against the current actor pose.

#### See also

[Implement an attack](../guides/attacks.md).

### avsectester.attack.insertion.compositor

<!-- api:begin module=avsectester.attack.insertion.compositor -->
Shared geometry-based insertion rendering and compositing.

| Name and signature | Meaning |
|---|---|
| `PatchCompositor(harmonizer: Harmonizer \| None = None, *, soften: float = 0.0)` | Appearance settings for geometry-based insertion compositing. |
| `render_resolved(frame: np.ndarray, camera: Camera, cam_from_world: np.ndarray, resolved: ResolvedInsertion, evidence: VisibilityEvidence \| None = None, *, compositor: PatchCompositor \| None = None) -> tuple[np.ndarray, np.ndarray]` | Composite a resolved insertion with per-pixel surface depth and visibility. |
<!-- api:end -->

#### Constraints and errors

Composition requires calibrated camera geometry and RGB/RGBA input. Harmonizers change only the
resolved visible region.

#### See also

[Implement an attack](../guides/attacks.md).

### avsectester.attack.insertion.placement

<!-- api:begin module=avsectester.attack.insertion.placement -->
Insertion positions and orientations shared by attack metadata.

| Name and signature | Meaning |
|---|---|
| `validate_attached_mount(anchor: str, offset_m: tuple[float, float, float]) -> tuple[float, float, float]` | Validate host-independent mounting options and return a finite metric offset. |
| `AttachedPlacement(host: str, anchor: str = "center", offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0))` | A user-specified host-local point. Named anchors refer to bounding-box faces. |
| `AttachedPlacement.local_position(extent: tuple[float, float, float]) -> np.ndarray` | Return this mounting point in host-local axes for full bounding-box dimensions. |
| `WorldPlacement(position_m: tuple[float, float, float])` | A fixed position in the canonical world frame, in meters. |
| `Orientation(mode: str = "fixed_world", rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0), target_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0))` | Rotation relative to a host, absolute in world, or directed at the ego. |
| `placement_from_spec(value: Any) -> AttachedPlacement \| WorldPlacement` | Read an explicit placement object or a `{type, options}` specification. |
| `orientation_from_spec(value: Any) -> Orientation` | Read an explicit orientation; default image insertions face the ego. |
<!-- api:end -->

#### Constraints and errors

World placement uses canonical world coordinates. Attached placement always uses the host's
bounding-box frame.

#### See also

[Implement an attack](../guides/attacks.md).

### avsectester.attack.insertion.types

<!-- api:begin module=avsectester.attack.insertion.types -->
Data shared by visibility providers and insertion renderers.

| Name and signature | Meaning |
|---|---|
| `Visibility(fraction: float, source: str, camera: str \| None = None, label: str \| None = None)` | Measured fraction or a coarse dataset representative. None camera means source-wide scope. |
| `VisibilityEvidence(visibility: Visibility \| None, reference_mask: np.ndarray, visible_mask: np.ndarray, target_depth: np.ndarray, reference_pixels: int, reason: str = "", sampled_rgba: np.ndarray \| None = None, ignored_occluders: tuple[str, ...] = ())` | One insertion's visibility and image-sized sampling results for one camera/instant. |
<!-- api:end -->

#### Constraints and errors

`Visibility` fractions describe visible projected coverage, not a driving outcome. Evidence
keeps the camera and occlusion source needed to interpret that measurement.

#### See also

[Implement an attack](../guides/attacks.md).

### avsectester.attack.insertion.visibility

<!-- api:begin module=avsectester.attack.insertion.visibility -->
Depth-based visibility and explicitly approximate cuboid visibility for insertions.

| Name and signature | Meaning |
|---|---|
| `CuboidVisibilityEstimator(*, tolerance_m: float = 0.002, max_reference_pixels: int = 4_000_000)` | Estimate opacity occlusion by known 3D boxes and other proposed planar insertions. |
| `CuboidVisibilityEstimator.estimate(subject: ResolvedInsertion, camera: Camera, cam_from_world: np.ndarray, *, occluders: Mapping[str, ActorState], camera_name: str, other_insertions: Sequence[ResolvedInsertion] = (), world_objects: Sequence[PlacedObject] = ()) -> VisibilityEvidence` | Measure target visibility using the configured camera and occlusion evidence. |
| `DepthVisibilityEstimator(*, tolerance_m: float = 0.0002, max_reference_pixels: int = 4_000_000)` | Compare known insertion surfaces with calibrated, aligned scene depth. |
| `DepthVisibilityEstimator.estimate(subject: ResolvedInsertion, camera: Camera, cam_from_world: np.ndarray, *, scene_depth: np.ndarray, depth_convention: str, camera_name: str, other_insertions: Sequence[ResolvedInsertion] = ()) -> VisibilityEvidence` | Measure target visibility using the configured camera and occlusion evidence. |
| `visibility_from_masks(reference, visible, *, camera: str, image_bounds=None) -> Visibility \| None` | Divide visible in-image pixels by the FULL unoccluded reference silhouette. |
| `visibility_from_depth(footprint, target_depth, scene_depth, *, camera: str, tolerance_m: float = 0.0002, image_bounds=None) -> Visibility \| None` | Compare target and scene depth in the same convention and pixel projection. |
<!-- api:end -->

#### Constraints and errors

Visibility uses calibrated camera geometry and explicit depth or cuboid occluders.
A target fully behind the camera has zero visible coverage.

The cuboid estimator reports `Visibility.source="actor_cuboids"`, the depth estimator reports
`"depth"`, and rendered-mask evidence reports `"rendered_masks"`.

#### See also

[Implement an attack](../guides/attacks.md).

### avsectester.attack.offline_report

<!-- api:begin module=avsectester.attack.offline_report -->
Structured, portable outputs produced by an attack's offline phase.

| Name and signature | Meaning |
|---|---|
| `ReportValue(value: Any, unit: str \| None = None, description: str \| None = None)` | One scalar measurement with its unit and explanation. |
| `ReportSeries(x: list[float], y: list[float], x_label: str = "iteration", y_label: str \| None = None, x_unit: str \| None = None, y_unit: str \| None = None, description: str \| None = None)` | A finite numeric series and its axis labels, units and explanation. |
| `ReportTable(rows: list[dict[str, Any]], description: str \| None = None, columns: dict[str, str] \| None = None)` | Flat scalar rows and explanations keyed by existing column names. |
| `ReportImage(pixels: np.ndarray, description: str \| None = None)` | Copied RGB/RGBA pixels and their caption. |
| `OfflineReport(directory: str \| Path \| None = None)` | Collect offline measurements and files independently of experiment execution. |
| `OfflineReport.set_value(name: str, value: Any, *, unit: str \| None = None, description: str \| None = None) -> None` | Set a scalar with its optional unit and human-readable explanation. |
| `OfflineReport.set_series(name: str, y: Sequence[float] \| np.ndarray, *, x: Sequence[float] \| np.ndarray \| None = None, x_label: str = "iteration", y_label: str \| None = None, x_unit: str \| None = None, y_unit: str \| None = None, description: str \| None = None) -> None` | Set a finite one-dimensional series and optional matching x coordinates. |
| `OfflineReport.set_table(name: str, rows: Sequence[Mapping[str, Any]], *, description: str \| None = None, columns: Mapping[str, str] \| None = None) -> None` | Set flat scalar rows and descriptions of existing columns. |
| `OfflineReport.set_image(name: str, pixels: np.ndarray, *, description: str \| None = None) -> None` | Set a copied uint8 RGB/RGBA image and optional caption. |
| `OfflineReport.save(directory: str \| Path) -> None` | Write structured entries, CSV/PNG files and the optional workspace. |
| `OfflineReport.load(directory: str \| Path) -> OfflineReport` | Read a saved report without constructing an attack, simulator or model. |
<!-- api:end -->

#### Constraints and errors

Names are shared across entry kinds and match ^[A-Za-z0-9][A-Za-z0-9_.-]*$. Tables contain flat
JSON scalar rows; series contain finite numbers; images are uint8 RGB/RGBA. Reusing a name
within its kind replaces its value.

`set_value` accepts booleans, strings, integers and finite real numbers; `None` and non-finite
numbers are rejected. Saving or loading rejects a status other than `"completed"` or `"failed"`.

#### See also

[Optimize attack artifacts](../guides/optimization.md).

### avsectester.attack.optimization

<!-- api:begin module=avsectester.attack.optimization -->
Public contracts, algorithms, parameterizations and session scoring for user attacks.

| Name and signature | Meaning |
|---|---|
| `AdversarialSample` | [AdversarialSample](#avsectesterattackoptimizationinterface) |
| `DataSource` | [DataSource](#avsectesterattackoptimizationinterface) |
| `GradientOptimizer` | [GradientOptimizer](#avsectesterattackoptimizationinterface) |
| `HideObject` | [HideObject](#avsectesterattackoptimizationinterface) |
| `OptimizationObjective` | [OptimizationObjective](#avsectesterattackoptimizationinterface) |
| `OptimizationResult` | [OptimizationResult](#avsectesterattackoptimizationinterface) |
| `Perturbation` | [Perturbation](#avsectesterattackoptimizationinterface) |
| `Scorer` | [Scorer](#avsectesterattackoptimizationinterface) |
| `SpoofObject` | [SpoofObject](#avsectesterattackoptimizationinterface) |
| `NES` | [NES](#avsectesterattackoptimizationblackbox) |
| `PGD(GradientOptimizer)(steps: int = 200, step_size: float = 4.0 / 255.0, batch: int = 4, tv_weight: float = 0.0, seed: int = 0, verbose: bool = False, log_every: int = 25)` | Projected Gradient Descent (sign-gradient descent on the loss, minimizing it). |
| `LinfImage(Perturbation)(epsilon=8 / 255, image_hw=(600, 800), device="cuda:0")` | A whole-image additive L-inf perturbation without an exported artifact. |
| `PatchPerturbation(Perturbation)(size=(128, 128), device="cuda:0", init="random", seed=0, alpha=None)` | A rectangular texture rendered into each sample's background, with photometric jitter. |
| `SessionScorer` | [SessionScorer](#avsectesterattackoptimizationscorers) |
<!-- api:end -->

#### See also

[Optimize attack artifacts](../guides/optimization.md).

### avsectester.attack.optimization.blackbox

<!-- api:begin module=avsectester.attack.optimization.blackbox -->
Natural Evolution Strategies over arbitrary user-defined samples and perturbations.

| Name and signature | Meaning |
|---|---|
| `NES(GradientOptimizer)(steps: int = 40, population_size: int = 20, sigma: float = 0.08, learning_rate: float = 0.03, seed: int = 0, verbose: bool = False, log_every: int = 5)` | Natural Evolution Strategies — gradient-free; estimates a gradient by Gaussian sampling. |
<!-- api:end -->

#### See also

[Optimize attack artifacts](../guides/optimization.md).

### avsectester.attack.optimization.interface

<!-- api:begin module=avsectester.attack.optimization.interface -->
Contracts for gradient-based and sampling-based optimization over perturbation models.

| Name and signature | Meaning |
|---|---|
| `AdversarialSample(x: Tensor, target: data_types.TargetSpec, context: dict = dict())` | One clean input to attack. `context` carries threat-model specifics — e.g. a patch's calibrated sampling grid and visibility mask — and is empty for a whole-image perturbation. |
| `DataSource(ABC)()` | Yields batches of clean inputs. A single fixed image for an L-inf attack; sampled backgrounds + placements + photometric jitter (Expectation-over-Transformation) for a physical patch. |
| `DataSource.sample(batch: int) -> list[AdversarialSample]` | Abstract. Return `batch` samples (with any per-sample randomness freshly drawn). |
| `Perturbation(ABC)()` | A threat model: the adversarial variable δ, how it enters the input, and its feasible set. |
| `Perturbation.init() -> Tensor` | Abstract. Return an initial δ (zeros / random) of this threat model's shape. |
| `Perturbation.apply(sample: AdversarialSample, delta: Tensor) -> Tensor` | Abstract. Form the adversarial input from a clean `sample` and δ (differentiable in δ). |
| `Perturbation.project(delta: Tensor) -> Tensor` | Abstract. Project δ back onto the feasible set (in place or returned). |
| `Perturbation.export(delta: Tensor) -> np.ndarray \| None` | Physical realization of δ, if any — a patch returns `(Hp,Wp,4)` uint8; default None. |
| `Scorer(ABC)()` | Access to the perception under attack: image -> target detection confidence in [0, 1]. |
| `Scorer.differentiable: bool = True` | Whether input-image gradients are supported by this component. |
| `Scorer.target_score(image: Tensor, target: data_types.TargetSpec) -> Tensor` | Abstract. Return a scalar tensor in [0, 1] for detecting `target` in `image`. |
| `OptimizationObjective(ABC)()` | Shape the target confidence into a scalar loss to MINIMIZE (lower == attack succeeds). |
| `OptimizationObjective.loss(score: Tensor) -> Tensor` | Abstract. Map a target confidence in [0, 1] to the scalar loss to minimize. |
| `HideObject(OptimizationObjective)()` | Suppress the target's detection by minimizing `loss = score` toward zero. |
| `SpoofObject(OptimizationObjective)()` | Fabricate / keep a detection where the target box is: `loss = 1 - score`. |
| `OptimizationResult(delta: Tensor, export: np.ndarray \| None, history: list[float], final_score: float)` | Output of an optimization run. |
| `GradientOptimizer(ABC)()` | Optimization algorithm over a data source, perturbation, scorer and objective. |
| `GradientOptimizer.run(data: DataSource, perturbation: Perturbation, scorer: Scorer, objective: OptimizationObjective) -> OptimizationResult` | Abstract. Optimize δ with exact input gradients in PGD or sampled gradient estimates in NES. |
<!-- api:end -->

#### Constraints and errors

`DataSource` yields `AdversarialSample` values. `Perturbation` owns parameters and export.
Differentiable scorers return an input-connected scalar; objective direction is explicit.

#### See also

[Optimize attack artifacts](../guides/optimization.md).

### avsectester.attack.optimization.perturbations

<!-- api:begin module=avsectester.attack.optimization.perturbations -->
Concrete threat models (`Perturbation`): a physical patch and a whole-image L-inf perturbation.

| Name and signature | Meaning |
|---|---|
| `PatchPerturbation(Perturbation)(size=(128, 128), device="cuda:0", init="random", seed=0, alpha=None)` | A rectangular texture rendered into each sample's background, with photometric jitter. |
| `LinfImage(Perturbation)(epsilon=8 / 255, image_hw=(600, 800), device="cuda:0")` | A whole-image additive L-inf perturbation without an exported artifact. |
<!-- api:end -->

#### Constraints and errors

`PatchPerturbation` requires two positive integer dimensions, `init="random"` or `"gray"`, and
an optional uint8 alpha array matching that size. Its RGB variable is CHW in [0, 1]; projection
clamps each value to this interval. Applying it requires a `(1, H, W)` visible mask and either
in-image `PlaneSampling` bounds or an `(H, W, 2)` sampling grid. Export produces HWC uint8 RGBA
with the supplied alpha unchanged. `LinfImage` initializes a zero additive CHW variable,
projects each element into [−epsilon, epsilon], and clamps `sample.x + delta` into [0, 1]; it
uses the base perturbation export behavior.

#### See also

[Optimize attack artifacts](../guides/optimization.md).

### avsectester.attack.optimization.scorers

<!-- api:begin module=avsectester.attack.optimization.scorers -->
Score arbitrary images using a closed-loop session's driving-system binding.

| Name and signature | Meaning |
|---|---|
| `SessionScorer(Scorer)(session, scorer_id, *, require_grad, input_format=None, device="cpu")` | Use a session scorer for user-generated images, preserving requested gradients. |
<!-- api:end -->

#### Constraints and errors

`SessionScorer` resolves a declared scorer. Gradient requests require a differentiable input and
connected result. `no_grad` evaluation uses the same scorer without requiring gradients.

#### See also

[Optimize attack artifacts](../guides/optimization.md).

### avsectester.attack.optimization.whitebox

<!-- api:begin module=avsectester.attack.optimization.whitebox -->
Gradient-based optimization algorithms (torch).

| Name and signature | Meaning |
|---|---|
| `PGD(GradientOptimizer)(steps: int = 200, step_size: float = 4.0 / 255.0, batch: int = 4, tv_weight: float = 0.0, seed: int = 0, verbose: bool = False, log_every: int = 25)` | Projected Gradient Descent (sign-gradient descent on the loss, minimizing it). |
<!-- api:end -->

#### Constraints and errors

`PGD` raises `ValueError` when the scorer declares no differentiable input.
Perturbation projection applies that perturbation's feasible bounds.

#### See also

[Optimize attack artifacts](../guides/optimization.md).

### avsectester.attack.runtime

<!-- api:begin module=avsectester.attack.runtime -->
The online phase: one AttackRuntime per run and the attack actions it returns.

| Name and signature | Meaning |
|---|---|
| `ResultSubscription(kind: Literal["observation", "physical_state"])` | Declare a closed-loop result read by an attack runtime. |
| `OnlineContext(step: int, stage: str, sim_time: float, frame: int, observation: Observation \| None = None, physical_state: WorldSnapshot \| None = None)` | Independent snapshots of subscribed results, with a read-only physical clock. |
| `OnlineContext.record_output(name: str, value: Any) -> None` | Save a named copy of `value`, such as an attack decision, in this run's log. |
| `AttackAction()` | Base of everything the online phase applies. Subclasses state where it acts. |
| `AttackAction.receipt_details() -> Mapping[str, Any]` | Data-only evidence from the most recent application, included in its receipt. |
| `WorldAttackAction(AttackAction, ABC)()` | Change the simulated world. The simulator applies it; it is visible from the next capture. |
| `WorldAttackAction.setup_ticks: int = 0` | Simulator setup ticks requested by this participant before decisions begin. |
| `WorldAttackAction.apply(simulator_runtime: SimulatorRuntime, resources: ExitStack) -> None` | Abstract. Change the world through the simulator's native access. |
| `SensorAttackAction(AttackAction, ABC)()` | Change one captured sensor's data before ObservationPipeline, in the current decision. |
| `SensorAttackAction.sensor: str` | Name of the sensor whose captured data this action modifies. |
| `SensorAttackAction.apply(data: Any, view: SensorView) -> Any` | Abstract. Return the changed sensor data. |
| `AVStackAttackAction(AttackAction)(stage: str, apply: Callable[[Any, OnlineContext], Any])` | Change the value at a driving-system stage in the current decision. |
| `AttackRuntime()` | One online-phase instance per run; every callback shares this object's state. |
| `AttackRuntime.subscriptions: Sequence[ResultSubscription] = ()` | Result subscriptions delivered to the runtime during execution. |
| `AttackRuntime.setup_ticks: int = 0` | Simulator setup ticks requested by this participant before decisions begin. |
| `AttackRuntime.setup(context: OnlineContext) -> Sequence[WorldAttackAction]` | Initialize online-phase state and return actions applied before the first decision. |
| `AttackRuntime.step(context: OnlineContext) -> Sequence[AttackAction]` | Return attack actions for the current decision. |
| `AttackRuntime.stop(context: OnlineContext) -> None` | Release online-phase resources; the default implementation has no effect. |
<!-- api:end -->

#### Constraints and errors

`setup` emits world actions; `step` emits typed actions. Sensor actions operate on captured-data
copies. `AVStackAttackAction` targets `localization.pre` through `command`,
excluding `sensors.post`.
Receipts describe application status and always include details; they do not measure
driving impact.

#### See also

[Implement an attack](../guides/attacks.md).

### avsectester.attack.segments

<!-- api:begin module=avsectester.attack.segments -->
Split a scenario timeline into attack segments for an attack's own `build_metadata`.

| Name and signature | Meaning |
|---|---|
| `split_segments(frames: Sequence[FrameReference], eligible: Callable[[FrameReference], bool], *, decision_steps: int, keep_tail: bool = False) -> tuple[AttackSegment, ...]` | Cut each run of consecutive eligible decision frames into segments of `decision_steps`. |
<!-- api:end -->

#### Constraints and errors

Eligible windows retain canonical frame references and a terminal stop boundary. Selection only
splits supported decision windows; it does not advance a simulator.

#### See also

[Implement an attack](../guides/attacks.md).

### avsectester.runlog

<!-- api:begin module=avsectester.runlog -->
The run log: one record of everything captured during a run, with Parquet persistence.

| Name and signature | Meaning |
|---|---|
| `EventRecord` | [EventRecord](#avsectesterrunloglog) |
| `Recorder` | [Recorder](#avsectesterrunloglog) |
| `RunLog` | [RunLog](#avsectesterrunloglog) |
| `StepRecord` | [StepRecord](#avsectesterrunloglog) |
| `read_table` | [read_table](#avsectesterrunloglog) |
<!-- api:end -->

#### See also

[Runtime reference](runtime.md).

### avsectester.runlog.log

<!-- api:begin module=avsectester.runlog.log -->
Independent run snapshots and incremental Parquet persistence.

| Name and signature | Meaning |
|---|---|
| `StepRecord(step: int, raw_observation: Any = None, model_observation: Any = None, physical_pre: Any = None, physical_post: Any = None, produced_command: Any = None, executed_command: Any = None, components: dict[str, Any] = dict(), detections: Sequence[Detection] \| None = None, sensor_geometry: Mapping[str, SensorGeometry] = dict(), object_visibility: Mapping[str, Mapping[str, ObjectVisibility]] = dict(), actions: list[Any] = list(), outputs: list[Any] = list(), telemetry: dict[str, Any] = dict())` | One decision, with independently captured physical and model-visible data. |
| `EventRecord(kind: str, step: int \| None = None, data: dict[str, Any] = dict())` | An initialization, termination or other non-step event. |
| `RunLog(metadata: dict[str, Any] = dict(), steps: list[StepRecord] = list(), events: list[EventRecord] = list(), status: str = "incomplete", path: Path \| None = None)` | Captured records, optionally backed by an incrementally written directory. |
| `RunLog.read(path: str \| Path, *, registry: SerializationRegistry = DEFAULT_REGISTRY, load_payloads: bool = True) -> RunLog` | Read completed parts, including those left by an interrupted writer. |
| `Recorder(path: str \| Path \| None = None, *, metadata: dict[str, Any] \| None = None, registry: SerializationRegistry = DEFAULT_REGISTRY, chunk_steps: int = 128, chunk_bytes: int = 8 * 1024 * 1024, payload_shard_bytes: int = 64 * 1024 * 1024, retain_steps: bool = True)` | Capture a run in memory and optionally stream it to Parquet. |
| `Recorder.snapshot(value: Any) -> Any` | Copy mutable data now; tensors are detached without importing torch. |
| `Recorder.record_step(record: StepRecord) -> StepRecord` | Append a decision record and flush a complete row group when due. |
| `Recorder.record_event(kind: str, *, step: int \| None = None, **data: Any) -> EventRecord` | Append a timestamped runtime event to the recording. |
| `Recorder.flush() -> None` | Publish the current chunks without marking the run complete. |
| `Recorder.finish(status: str = "completed", **details: Any) -> RunLog` | Finish recording with a valid terminal status and persist its manifest. |
| `Recorder.abort() -> None` | Discard unpublished records and leave the persisted run incomplete. |
| `read_table(path: str \| Path, columns: Sequence[str] \| None = None, *, stream: str = "steps") -> Any` | Read selected top-level columns without loading arrays or other columns. |
<!-- api:end -->

#### Constraints and errors

`RunLog` contains metadata, steps, events, status and an optional storage path. read restores
typed records through its explicit/default registry; `load_payloads=False` leaves payload
references unresolved.

#### See also

[Runtime reference](runtime.md).

### avsectester.execution

<!-- api:begin module=avsectester.execution -->
Closed-loop execution, isolated services, attack dispatch and frame-data caching.

No exports; import from submodules.
<!-- api:end -->

#### See also

[Runtime reference](runtime.md).

### avsectester.execution.frame_data_cache

<!-- api:begin module=avsectester.execution.frame_data_cache -->
Preparation-owned caching of simulator-generated frame data.

| Name and signature | Meaning |
|---|---|
| `FrameDataCache()` | Cache simulator-generated frame sequences independently of the dataset. |
| `FrameDataCache.key(start: ScenarioStart, simulator_identity: Mapping[str, Any], seed: int) -> str` | Hash start kind, scenario, payload, simulator identity and seed. |
| `FrameDataCache.get_or_generate(key: str, generate: Callable[[], FrameDataSequence]) -> FrameDataSequence` | Return cached frame data or generate and retain it once for this key. |
| `FrameDataCache.get(key: str) -> FrameDataSequence` | Return a copy of the cached sequence; raise `KeyError` when the key is absent. |
<!-- api:end -->

#### Constraints and errors

Cache identity uses start kind, scenario and payload, simulator identity and seed.
Services include environment and deployment ticks in the simulator identity.

#### See also

[Implement an attack](../guides/attacks.md).

### avsectester.execution.runner

<!-- api:begin module=avsectester.execution.runner -->
One closed-loop scheduler with independent RunLog captures.

| Name and signature | Meaning |
|---|---|
| `RecordingPolicy(sensor_ids: tuple[str, ...] = (), representations: tuple[str, ...] = ("model_input",), every_n_steps: int = 1)` | Choose sensor payloads without sampling commands or physical state. |
| `RecordingPolicy.observation(value: Observation, step: int, representation: str) -> Observation` | Apply the sensor recording policy while preserving observation metadata. |
| `RunRequest(decision_steps: int, hooks: HookSet \| None = None, attacks: Sequence[Any] = (), recording: RecordingPolicy = RecordingPolicy(), output_path: str \| Path \| None = None, retain_steps: bool = True, metadata: Mapping[str, Any] = dict(), start: Any = None, initialization: Any = None, deployment: Any = None, environment: Mapping[str, Any] = dict())` | One caller-owned simulator/driving-system run, independent of experiment orchestration. |
| `ClosedLoopRunner()` | Schedule one driving loop; never close caller-owned simulator/driving objects. |
| `ClosedLoopRunner.run(simulator_runtime: SimulatorRuntime, avstack: AVStack, request: RunRequest, *, recorder: Recorder \| None = None, release_owned_resources: Callable[[], None] \| None = None) -> RunLog` | Execute the requested treatment, record its decisions and close all run resources. |
<!-- api:end -->

#### Constraints and errors

Raw/model observations and produced/executed commands are separate evidence. `RecordingPolicy`
controls retained payloads.

#### See also

[Runtime reference](runtime.md).

### avsectester.execution.services

<!-- api:begin module=avsectester.execution.services -->
The metabuilder and the offline phase share one isolated closed-loop service boundary.

| Name and signature | Meaning |
|---|---|
| `CandidateRequest(segment: AttackSegment, seed: int, decision_steps: int, environment: Mapping[str, Any], recording: RecordingPolicy, attack: Any = None, artifacts: Any = None, output_path: str \| Path \| None = None)` | One candidate rollout. `attack` creates the deployed runtime with its create_runtime. |
| `SensorRequest(scenario_id: str, frame: FrameReference, sensor_ids: tuple[str, ...], environment: Mapping[str, Any] = dict(), views: bool = False)` | Recorded sensor request with an optional camera selection. |
| `RenderRequest(scenario_id: str, frame: FrameReference, sensor_id: str, actions: Sequence[AttackAction] = (), environment: Mapping[str, Any] = dict())` | Capture `sensor_id` at `frame` with attack actions applied. |
| `SensorSample(frame: FrameReference, observation: Observation, provenance: Mapping[str, Any], views: Mapping[str, SensorView] = dict())` | An environment-processed observation and optional detached raw sensor views. |
| `RenderResult(frame: FrameReference, observation: Observation, view: SensorView, provenance: Mapping[str, Any], object_visibility: Mapping[str, Mapping[str, ObjectVisibility]] = dict())` | The changed capture and the requested sensor's view before its sensor actions. |
| `ComponentResult(value: Any, descriptor: driving_adapter.ComponentDescriptor, provenance: Mapping[str, Any])` | Outputs returned by an explicitly evaluated driving-system component. |
| `ScoreResult(value: Any, descriptor: driving_adapter.ScorerDescriptor, provenance: Mapping[str, Any])` | Score returned by a named scorer, including its serialized representation. |
| `ClosedLoopCapabilities(dataset_kinds: tuple[str, ...], sensor_data: bool, render: bool, insert_objects: bool, rollout: bool, components: tuple[driving_adapter.ComponentDescriptor, ...], scorers: tuple[driving_adapter.ScorerDescriptor, ...])` | Capabilities exposed by the dataset, simulator and driving adapter combination. |
| `ClosedLoopServices(dataset: Dataset, simulator: Simulator, driving_adapter: driving_adapter.DrivingAdapter \| None = None, frame_data_cache: FrameDataCache \| None = None, *, deployment: DeploymentPlan = DeploymentPlan())` | Factory for isolated scenario and closed-loop sessions using shared frame-data caching. |
| `ClosedLoopServices.close() -> None` | Close all sessions, including sessions a metabuilder left open on error. |
| `ClosedLoopServices.capabilities: ClosedLoopCapabilities` | Capabilities available without constructing a simulator runtime. |
| `ClosedLoopServices.session(seed: int) -> ClosedLoopSession` | Create an isolated session with fresh world and driving history for `seed`. |
| `ClosedLoopServices.scenario(scenario_id: str, *, seed: int) -> ScenarioView` | Create a selection view for the requested scenario and seed. |
| `ClosedLoopSession(services: ClosedLoopServices, seed: int)` | Owned simulator and driving-system services for selection, rendering, scoring and rollouts. |
| `ClosedLoopSession.close() -> None` | Close session resources and release world ownership. |
| `ClosedLoopSession.can_start_at(position: int) -> bool` | Check deployment and recorded-history requirements without starting a runtime. |
| `ClosedLoopSession.frame_data(scenario: str, frame: FrameReference, environment: Mapping[str, Any] \| None = None) -> FrameData` | Return recorded frame data through the shared cache. |
| `ClosedLoopSession.initialization(start: ScenarioStart, environment: Mapping[str, Any], deployment: DeploymentPlan \| None = None) -> Initialization \| None` | Resolve replay inputs for a selected start, using the preparation-owned cache. |
| `ClosedLoopSession.read_sensors(request: SensorRequest) -> SensorSample` | Read environment-processed sensor observations at the requested recorded frame. |
| `ClosedLoopSession.render(request: RenderRequest) -> RenderResult` | Render the requested frame or viewpoint using this session. |
| `ClosedLoopSession.components() -> Sequence[driving_adapter.ComponentDescriptor]` | Return the available driving-system component descriptors. |
| `ClosedLoopSession.scorers() -> Sequence[driving_adapter.ScorerDescriptor]` | Return the available driving-system scorer descriptors. |
| `ClosedLoopSession.evaluate(component_id: str, inputs: Any) -> ComponentResult` | Evaluate an explicitly bound driving-system component. |
| `ClosedLoopSession.score(scorer_id: str, inputs: Any, target: Any, require_grad: bool = False) -> ScoreResult` | Evaluate a bound scorer with the requested target and codec. |
| `ClosedLoopSession.rollout(request: CandidateRequest) -> RunLog` | Execute one candidate rollout from its recorded start. |
| `ScenarioView(scenario_id: str, services: ClosedLoopServices, seed: int)` | Recorded scenario and selection services, with candidate-rejection reasons. |
| `ScenarioView.record_reason(text: str) -> None` | Attach a user-supplied selection explanation to this scenario's preparation record. |
| `ScenarioView.reasons: tuple[str, ...]` | Bounded, insertion-ordered reason counts; repeated candidates add no entries. |
| `ScenarioView.frames: Sequence[FrameReference]` | Ordered decision-frame references available to this selection view. |
| `ScenarioView.frame_data(frame: FrameReference) -> FrameData` | Resolve recorded data or a preparation-owned cache, without mutating Dataset. |
| `ScenarioView.scenario_map() -> ScenarioMap \| Unavailable` | Read this scenario's semantic map without starting a simulator. |
<!-- api:end -->

#### Constraints and errors

Sessions reject overlapping operations and reused component instances. Reading sensor views or
rendering raises `NotImplementedError` when the simulator does not support rendering.
`CandidateRequest.decision_steps` must equal the segment interval, and its seed must match the
session. The terminal frame must match `dataset.decision_window`; deployment actions must fit
the shared budget.
`score(require_grad=True)` rejects unsupported or detached scores. Closing an active operation
raises `RuntimeError`.
Each `evaluate` call constructs a fresh `AVStack`.

#### See also

[Implement an attack](../guides/attacks.md).

### avsectester.evaluation

<!-- api:begin module=avsectester.evaluation -->
Metrics, reports and visualization derived solely from recorded experiment logs.

| Name and signature | Meaning |
|---|---|
| `ComparisonGroup` | [ComparisonGroup](#avsectesterevaluationmetric) |
| `DetectionMetrics` | [DetectionMetrics](#avsectesterevaluationmetric) |
| `DrivingMetrics` | [DrivingMetrics](#avsectesterevaluationmetric) |
| `FieldMeasurements` | [FieldMeasurements](#avsectesterevaluationmetric) |
| `Metric` | [Metric](#avsectesterevaluationmetric) |
| `MetricDescription` | [MetricDescription](#avsectesterevaluationmetric) |
| `MetricField` | [MetricField](#avsectesterevaluationmetric) |
| `MetricScalar` | [MetricScalar](#avsectesterevaluationmetric) |
| `PairMetric` | [PairMetric](#avsectesterevaluationmetric) |
| `ReasoningPhraseRate` | [ReasoningPhraseRate](#avsectesterevaluationmetric) |
| `StopImpact` | [StopImpact](#avsectesterevaluationmetric) |
| `StopLineMetrics` | [StopLineMetrics](#avsectesterevaluationmetric) |
| `TreatmentMeasurements` | [TreatmentMeasurements](#avsectesterevaluationmetric) |
| `signed_stop_line_distances` | [signed_stop_line_distances](#avsectesterevaluationmetric) |
| `TrajectoryDeviation` | [TrajectoryDeviation](#avsectesterevaluationmetric) |
| `TreatmentPair` | [TreatmentPair](#avsectesterevaluationmetric) |
| `UnavailableMetricValue` | [UnavailableMetricValue](#avsectesterevaluationmetric) |
| `EvaluationConfiguration` | [EvaluationConfiguration](#avsectesterevaluationreport) |
| `EvaluationReport` | [EvaluationReport](#avsectesterevaluationreport) |
| `RunGroup` | [RunGroup](#avsectesterevaluationevaluator) |
| `evaluate_groups` | [evaluate_groups](#avsectesterevaluationevaluator) |
<!-- api:end -->

#### See also

[Evaluate experiments](../guides/evaluation.md).

### avsectester.evaluation.evaluator

<!-- api:begin module=avsectester.evaluation.evaluator -->
Evaluate ordered recorded run groups independently of experiment configurations.

| Name and signature | Meaning |
|---|---|
| `RunGroup(segment_index: int, segment: AttackSegment, seed: int, logs: Mapping[str, RunLog], errors: Mapping[str, str] = dict(), scenario_map: ScenarioMap \| Unavailable = Unavailable("No saved scenario map is available"), treatment_attacks: Mapping[str, str \| None] = dict(), part: str \| None = None)` | Ordered treatment logs sharing a segment and seed, including unsuccessful runs. |
| `evaluate_groups(groups: Sequence[RunGroup], metrics: Sequence[Metric], *, notes: Sequence[str] = ()) -> EvaluationReport` | Evaluate run metrics and each treatment pair in the supplied group and log order. |
<!-- api:end -->

#### Constraints and errors

Pairs containing a non-completed run are marked unavailable. Metric errors are recorded
without discarding other results.

`evaluate_groups` leaves `EvaluationReport.experiment` empty; experiment-level evaluation fills
that context.

#### See also

[Evaluate experiments](../guides/evaluation.md).

### avsectester.evaluation.metric

<!-- api:begin module=avsectester.evaluation.metric -->
Metrics computed from recorded logs, without live simulators or driving models.

| Name and signature | Meaning |
|---|---|
| `MetricScalar` | Scalar measurement value or an unavailable value with its reason. |
| `FieldMeasurements` | Mapping from unprefixed metric fields to scalar measurements. |
| `TreatmentMeasurements` | Mapping from treatment names to their field measurements. |
| `UnavailableMetricValue(reason: str, value_type: Literal["number", "boolean"] = "number")` | A missing scalar and its reason; value_type preserves unavailable boolean fields. |
| `ComparisonGroup(segment: AttackSegment, seed: int, runs: Mapping[str, RunLog], scenario_map: ScenarioMap \| Unavailable = Unavailable("No saved scenario map is available"), treatment_attacks: Mapping[str, str \| None] = dict())` | Completed treatments sharing one segment unit and seed, plus its saved scenario map. |
| `TreatmentPair(treatment_a: str, treatment_b: str, log_a: RunLog, log_b: RunLog)` | A completed pair in configuration order; differences are defined as B minus A. |
| `MetricField(description: str, unit: str \| None = None)` | Meaning, calculation and missing-evidence conditions of one scalar field. |
| `MetricDescription(summary: str, fields: Mapping[str, MetricField])` | A metric summary and fields keyed exactly as returned by evaluate. |
| `Metric(ABC)()` | Evaluate recorded data; return scalars or UnavailableMetricValue values carrying their reasons. |
| `Metric.describe() -> MetricDescription` | Describe this instance; custom metrics may provide field explanations and units. |
| `Metric.evaluate(data: Any) -> Mapping[str, Any]` | Abstract. Return treatment-to-field mappings for the completed runs in a comparison group. |
| `PairMetric(Metric)()` | A metric evaluated once for each completed pair, rather than for individual runs. |
| `PairMetric.evaluate(pair: TreatmentPair) -> FieldMeasurements` | Abstract. Return scalar fields comparing the two completed runs in `pair`. |
| `DrivingMetrics(Metric)(stop_speed: float = DEFAULT_STOP_SPEED_MPS)` | Physical speed, actuator braking, actor distances and top-down box collisions. |
| `TrajectoryDeviation(PairMetric)()` | Symmetric distance between true ego positions at matching decision indices (meters). |
| `DetectionMetrics(Metric)(match_distance: float = 2.0, max_range: float \| None = 50.0, category_map: Mapping[str, str] \| None = None)` | Greedy nearest-center BEV matching in the true ego frame at capture time. |
| `StopImpact(PairMetric)(stop_speed: float = DEFAULT_STOP_SPEED_MPS, reference_speed: float = 1.0, min_overdrive_steps: int = 3)` | Directional stopping/braking comparisons, evaluated in both A→B and B→A directions. |
| `signed_stop_line_distances(positions: np.ndarray, line_points: np.ndarray, *, endpoint_tolerance_m: float = 3.) -> tuple[np.ndarray, tuple[int, float] \| None]` | Horizontal distances to an endpoint-extended stop polyline and first crossing. |
| `StopLineMetrics(Metric)(stop_speed: float = DEFAULT_STOP_SPEED_MPS, endpoint_tolerance_m: float = 3., brake_deceleration_mps2: float = 1.)` | Physical stopping and crossing relative to the selected saved map stop line. |
| `ReasoningPhraseRate(Metric)(phrase: str, component: str = "policy", field: str = "reasoning_text")` | Case-insensitive phrase counts and rate over all recorded decision texts. |
<!-- api:end -->

#### Constraints and errors

Metrics consume saved evidence without inference. `UnavailableMetricValue` retains type and
reason. Stop-line distances are horizontal, positive before crossing and negative afterward; a
run is assumed to start before its selected line.

`signed_stop_line_distances` returns one signed distance per input position and either `None` or
`(segment_index, fraction)` for the first crossing. Position arrays must be finite N×3; stop
lines must have at least two finite 3D points and a nonzero horizontal segment.

#### Extension contract

Implement `PairMetric.evaluate(pair)` to return one scalar field mapping for the completed
`TreatmentPair`. Missing evidence must produce `UnavailableMetricValue` with a reason rather
than fabricated measurements. Describe those fields through `describe`.

#### See also

[Evaluate experiments](../guides/evaluation.md).

### avsectester.evaluation.report

<!-- api:begin module=avsectester.evaluation.report -->
Offline report records, formatting and statistics for recorded measurements.

| Name and signature | Meaning |
|---|---|
| `EvaluationConfiguration(metrics: tuple[Metric, ...] \| None = None)` | Metrics for the whole experiment; None chooses one data-aware default metric set. |
| `EvaluationReport(groups: list[dict[str, Any]] = list(), treatment_summaries: list[dict[str, Any]] = list(), pair_summaries: list[dict[str, Any]] = list(), selections: list[dict[str, Any]] = list(), metrics: list[dict[str, Any]] = list(), notes: list[str] = list(), metric_errors: list[dict[str, Any]] = list(), path: Path \| None = None, experiment: dict[str, Any] = dict())` | Per-run and per-pair measurements, missing reasons and population statistics. |
| `EvaluationReport.summarize() -> None` | Rebuild treatment and pair population statistics from the recorded group rows. |
| `EvaluationReport.record_error(source: str, error: Exception, **context: Any) -> None` | Record an evaluation failure and its group or pair context without losing results. |
| `EvaluationReport.as_dict() -> dict[str, Any]` | Return JSON-compatible report data, independent of live experiment objects. |
| `EvaluationReport.format_summary() -> str` | Format every metric class by part and treatment, followed by report explanations. |
| `EvaluationReport.save(directory: str \| Path) -> Path` | Write report.json and runs.csv; metric names prefix columns to protect identities. |
<!-- api:end -->

#### Constraints and errors

`format_summary` returns a string. Summaries use population std (ddof=0). Pair differences are
B−A per group before aggregation. JSON/CSV keep full precision.

#### See also

[Evaluate experiments](../guides/evaluation.md).

### avsectester.evaluation.visualization

<!-- api:begin module=avsectester.evaluation.visualization -->
Images, overlays and animations exported from RunLogs after simulators have closed.

| Name and signature | Meaning |
|---|---|
| `View` | A callable that renders one recorded step into an image-compatible value. |
| `as_rgb(frame: Any) -> np.ndarray \| None` | Read canonical pixels or a logged native camera payload without importing its SDK. |
| `camera_view(record: StepRecord \| Observation, camera: str \| None = None) -> np.ndarray \| None` | Read a recorded model-visible camera image (or an explicit Observation). |
| `lidar_bev(record: StepRecord \| Observation, size: int = 800, meters: float = 60.0) -> np.ndarray \| None` | Read a saved lidar cloud and draw x-forward up, y-left left. Non-lidar payloads skip. |
| `annotate(image: Any, detections: Sequence[Any], threshold: float = 0.3) -> np.ndarray` | Draw explicit 2-D labels `(xyxy, score, text[, color])` on saved image pixels. |
| `box3d_corners(center: Any, extent: Any, yaw: float) -> np.ndarray` | Eight cuboid corners from full x/y/z extents and right-handed yaw, in radians. |
| `draw_boxes3d(image: Any, view: SensorView, *, actors: Mapping[str, Any] \| None = None, target: str \| None = None, subdiv: int = 8, max_distance: float \| None = None, color: tuple[int, int, int] = (40, 200, 40), target_color: tuple[int, int, int] = (235, 64, 52)) -> np.ndarray` | Project ActorState cuboids through a SensorView, without SceneGT. |
| `sensor_view(record: StepRecord, camera: str \| None = None) -> SensorView \| None` | Rebuild projection geometry from recorded canonical sensor-view metadata/calibration. |
| `labels_view(record: StepRecord, camera: str \| None = None, *, target: str \| None = None, max_distance: float \| None = None) -> np.ndarray \| None` | Overlay the saved physical_pre actors, using its real ego pose and recorded camera. |
| `detections_view(record: StepRecord, camera: str \| None = None, *, threshold: float = 0.3) -> np.ndarray \| None` | Overlay StepRecord.detections; this never runs a detector or reads components. |
| `save_image(image: Any, path: str \| Path) -> None` | Write an RGB frame without a plotting backend. |
| `record_run(log: RunLog, out_dir: str \| Path, *, visualize: View = camera_view, prefix: str = "frame") -> tuple[Path, ...]` | Export a frame per recorded decision. Does not run an experiment or simulator. |
| `filmstrip(source: RunLog \| Sequence[Any], cols: int = 4, pad: int = 6, bg: tuple[int, int, int] = (20, 20, 20), *, visualize: View = camera_view) -> np.ndarray` | Assemble recorded camera frames into one contact sheet. |
| `save_gif(source: RunLog \| Sequence[Any], path: str \| Path, fps: int = 5, *, visualize: View = camera_view) -> None` | Write an animation from recorded frames, without model or simulator imports. |
| `save_sequence(log: RunLog, out_dir: str \| Path, name: str = "sequence", cols: int = 4, fps: int = 4, *, visualize: View = camera_view) -> tuple[str, str] \| None` | Export a contact sheet and GIF from one recorded run; no frames yields None. |
| `plot_group(runs: Mapping[str, Any], path: str \| Path, *, title: str \| None = None) -> Path` | Plot physical speeds; add an actuator-brake panel only when some run records that evidence. |
<!-- api:end -->

#### Constraints and errors

These helpers do not run models. `plot_group` omits the actuator-brake panel when actuator
commands are unavailable.

#### See also

[Evaluate experiments](../guides/evaluation.md).

### avsectester.implemented_attacks

<!-- api:begin module=avsectester.implemented_attacks -->
Attack implementations, discovered by class name.

No exports; import from submodules.
<!-- api:end -->

#### See also

[Implement an attack](../guides/attacks.md).

### avsectester.implemented_attacks.object_insertion_attack

<!-- api:begin module=avsectester.implemented_attacks.object_insertion_attack -->
Public ObjectInsertionAttack implementation.

| Name and signature | Meaning |
|---|---|
| `ObjectInsertionAttack` | [ObjectInsertionAttack](#avsectesterimplemented_attacksobject_insertion_attackattack) |
<!-- api:end -->

#### See also

[Image object insertion](../examples/image-insertion.md).

### avsectester.implemented_attacks.object_insertion_attack.assets

<!-- api:begin module=avsectester.implemented_attacks.object_insertion_attack.assets -->
Sign, person-poster and traffic-light assets owned by ObjectInsertionAttack.

| Name and signature | Meaning |
|---|---|
| `SignAsset(face: np.ndarray \| str \| Path = STOP_SIGN_FACE, width_m: float = 0.9, mount_height_m: float = 1.5, post_width_m: float = 0.07, post_count: int = 1)` | A local planar sign and optional posts in a bottom-centered object frame. |
| `SignAsset.object_asset(category: str) -> ObjectAsset` | Return this sign's panel and optional supports as a bottom-centered object asset. |
| `SignAsset.planes() -> Sequence[PlaneSurface]` | Resolve the asset into textured planes in its local coordinate frame. |
| `PersonAsset(image: np.ndarray \| str \| Path, mode: str = "billboard", height_m: float = 1.75, width_m: float = 1.4, mount_height_m: float = 0.6)` | A cut-out standee or a printed person poster, with its origin at ground level. |
| `PersonAsset.planes()` | Resolve the asset into textured planes in its local coordinate frame. |
| `TrafficLightAsset(head_count: int = 3, lit: str = "red", texture_height_pixels: int = 300, gap_fraction: float = 0.25, board: tuple \| None = (35, 37, 40), width_m: float = 2.4, mount_height_m: float = 2.2)` | A traffic-light board and post, whose origin is the ground beneath the board. |
| `TrafficLightAsset.planes()` | Resolve the asset into textured planes in its local coordinate frame. |
<!-- api:end -->

#### Constraints and errors

`SignAsset` exposes the configured sign face and physical dimensions. Placement refers to the
selected host frame or the ego origin, not an inferred map landmark.

#### See also

[Image object insertion](../examples/image-insertion.md).

### avsectester.implemented_attacks.object_insertion_attack.attack

<!-- api:begin module=avsectester.implemented_attacks.object_insertion_attack.attack -->
Connect object assets, selection and online image insertion.

| Name and signature | Meaning |
|---|---|
| `ObjectInsertionAttack(Attack)(*, decision_steps, placement, assets=None, orientation=None, sensor="front", min_visible_fraction=0.0, max_segments=None, keep_tail=False, placement_relative_to_ego_origin=False, compositor=None)` | Composite a placed sign, person poster or traffic-light asset in one camera. |
| `ObjectInsertionAttack.insertion(artifacts, placement=None)` | Build a sensor insertion from the selected segment and prepared assets. |
| `ObjectInsertionAttack.selected_placement(segment)` | Rebuild the placement selected by this attack from data-only metadata. |
<!-- api:end -->

#### Constraints and errors

Selection metadata identifies supported placement hosts.
This attack inherits the base no-op `validate_segment`.

#### See also

[Image object insertion](../examples/image-insertion.md).

### avsectester.implemented_attacks.patch_attack

<!-- api:begin module=avsectester.implemented_attacks.patch_attack -->
World-surface patch attack with offline preparation and online deployment.

| Name and signature | Meaning |
|---|---|
| `PatchAttack` | [PatchAttack](#avsectesterimplemented_attackspatch_attackattack) |
<!-- api:end -->

#### See also

[Physical patch](../examples/physical-patch.md).

### avsectester.implemented_attacks.patch_attack.attack

<!-- api:begin module=avsectester.implemented_attacks.patch_attack.attack -->
Connect patch selection, optional optimization and online deployment.

| Name and signature | Meaning |
|---|---|
| `PatchAttack(Attack)(*, decision_steps, assets: np.ndarray \| str \| Path \| Texture \| None = None, candidate_hosts=None, hosts_per_window=1, sensor="front", width_m=1.0, height_m=0.5, anchor="rear_center", offset_m=(-0.01, 0.0, 0.0), orientation=None, max_distance=None, min_visible_fraction=0.0, max_windows=None, keep_tail=False, optimization=None, prop=DEFAULT_PROP, emissive=True, compositor=None)` | Select the best visible hosts per time window, then deploy one patch artifact. |
| `PatchAttack.placement_for(host)` | Resolve patch placement on the selected host actor. |
| `PatchAttack.selected_placement(segment)` | Rebuild this attack's host-local placement from its data-only metadata. |
| `PatchAttack.insertion(artifacts, *, host=None, placement=None)` | Build a patch insertion using the selected host and prepared texture. |
<!-- api:end -->

#### Constraints and errors

Patch selection and deployment use validated host/camera metadata. Optimization is supplied
through explicit offline options; the artifact is passed to each online runtime.

#### See also

[Physical patch](../examples/physical-patch.md).

### avsectester.implemented_attacks.patch_attack.offline

<!-- api:begin module=avsectester.implemented_attacks.patch_attack.offline -->
Patch-specific sampling, preparation and optional candidate rollout validation.

| Name and signature | Meaning |
|---|---|
| `PatchOptimization(algorithm: GradientOptimizer, scorer_id: str \| None = None, size: tuple[int, int] = (64, 64), device: str \| None = None, target_box: tuple[float, float, float, float] \| None = None, brightness: tuple[float, float] = (0.8, 1.2), contrast: tuple[float, float] = (0.8, 1.2), validate_rollout: bool = False, init_texture: np.ndarray \| str \| Path \| Texture \| None = None)` | Typed patch optimization settings; the algorithm owns its optimization seed. |
<!-- api:end -->

#### Constraints and errors

Both optimization paths score rendered or differentiably composed images through a bound
session scorer. NES renders candidate textures; optional `validate_rollout` runs rollouts
after optimization.

#### See also

[Physical patch](../examples/physical-patch.md).

### avsectester.implemented_attacks.patch_attack.online

<!-- api:begin module=avsectester.implemented_attacks.patch_attack.online -->
Deploy calibrated image patches or native CARLA panels from explicit artifacts.

| Name and signature | Meaning |
|---|---|
| `CarlaPatchPanel(location: tuple, rotation: tuple, texture: np.ndarray, prop: str = DEFAULT_PROP, emissive: bool = True)` | A native CARLA panel owned by one run, with an explicit actor-local transform. |
| `CarlaPatchPanel.apply(world: Any, target_actor: Any, *, tick: Callable[[], Any] \| None = None, on_spawn: Callable[[Any], None] \| None = None) -> list` | Spawn the panel on `target_actor` in `world` and paint it; return spawned actors. |
<!-- api:end -->

#### Constraints and errors

`CarlaPatchPanel` is the native CARLA panel transport. It requires native CARLA geometry and
texture support; its setup/cleanup remains scoped to the attack runtime.

#### See also

[Physical patch](../examples/physical-patch.md).

### avsectester.implemented_attacks.phantom_injection_attack

<!-- api:begin module=avsectester.implemented_attacks.phantom_injection_attack -->
Detection-injection attack with offline and online phases.

| Name and signature | Meaning |
|---|---|
| `PhantomInjectionAttack` | [PhantomInjectionAttack](#avsectesterimplemented_attacksphantom_injection_attackattack) |
<!-- api:end -->

#### See also

[Phantom injection](../examples/phantom-injection.md).

### avsectester.implemented_attacks.phantom_injection_attack.attack

<!-- api:begin module=avsectester.implemented_attacks.phantom_injection_attack.attack -->
Fabricate a box in an AVStack detector's declared source frame.

| Name and signature | Meaning |
|---|---|
| `PhantomInjectionAttack(Attack)(*, decision_steps: int, max_segments: int \| None = None, target_xyz=(6.0, 0.0, -0.7), object_type="Car", score=0.9, extent=(4.0, 1.8, 1.6), object_id=90002, keep_tail: bool = False)` | Append one AVStack BoxDetection, including to empty detection containers. |
<!-- api:end -->

#### Constraints and errors

The attack emits `AVStackAttackAction` at the fixed `perception.post` stage.

#### See also

[Phantom injection](../examples/phantom-injection.md).

### avsectester.implemented_attacks.speed_feedback_attack

<!-- api:begin module=avsectester.implemented_attacks.speed_feedback_attack -->
Speed-feedback modification with offline and online phases.

| Name and signature | Meaning |
|---|---|
| `SpeedFeedbackAttack` | [SpeedFeedbackAttack](#avsectesterimplemented_attacksspeed_feedback_attackattack) |
<!-- api:end -->

#### See also

[Speed-feedback attack](../examples/speed-feedback-attack.md).

### avsectester.implemented_attacks.speed_feedback_attack.attack

<!-- api:begin module=avsectester.implemented_attacks.speed_feedback_attack.attack -->
A feedback attack that changes the ego speed seen by the driving system.

| Name and signature | Meaning |
|---|---|
| `SpeedFeedbackAttack(Attack)(*, decision_steps: int, target_speed: float, gain: float = 0.1, max_bias: float = 5.0, initial_bias: float = 0.0, max_segments: int \| None = None, keep_tail: bool = False)` | Adapt a nonnegative model-visible speed bias from actual physical speed. |
<!-- api:end -->

#### Constraints and errors

The attack changes model-visible speed feedback while physical state remains separately
recorded. Constructor parameters are validated; `validate_segment` is the inherited no-op.

#### See also

[Speed-feedback attack](../examples/speed-feedback-attack.md).

### avsectester.implemented_attacks.stop_sign_patch_attack

<!-- api:begin module=avsectester.implemented_attacks.stop_sign_patch_attack -->
Stop-sign patch attack.

| Name and signature | Meaning |
|---|---|
| `StopSignPatchAttack` | [StopSignPatchAttack](#avsectesterimplemented_attacksstop_sign_patch_attackattack) |
<!-- api:end -->

#### See also

[Stop-sign patch attack](../examples/stop-sign-patch.md).

### avsectester.implemented_attacks.stop_sign_patch_attack.attack

<!-- api:begin module=avsectester.implemented_attacks.stop_sign_patch_attack.attack -->
Connect stop-approach selection, optional patch optimization and replica deployment.

| Name and signature | Meaning |
|---|---|
| `StopSignPatchAttack(Attack)(initial_patch: str = "white", optimization: DetectorPatchOptimization \| RolloutPatchOptimization \| None = None, mask_path: str \| Path = DEFAULT_MASK_PATH, noise_std: float = .25, noise_seed: int = 0, approach_m: float = 50., hold_steps: int = 20, offset_m: float = .10, width_scale: float = 1.1, harmonizer: Harmonizer \| None = None, face_orientation: str = "upright", face_size: str = "mutcd_standard", min_approach_m: float = 20.)` | Cover a recorded stop sign with a replica face whose masked pixels form a patch. |
| `StopSignPatchAttack.validate_segment(segment: AttackSegment) -> None` | Validate the sign geometry and recorded-route inputs used by both optimizers. |
| `StopSignPatchAttack.replica_placement(segment: AttackSegment, texture: np.ndarray) -> tuple[ObjectAsset, np.ndarray]` | Share the selected MUTCD/map width and upright/map orientation across all phases. |
| `StopSignPatchAttack.replica_texture(artifacts: object) -> np.ndarray` | Return the replica face with the supplied patch in its editable region. |
<!-- api:end -->

#### Constraints and errors

`replica_placement` is shared by offline sampling and online deployment. The inserted object ID
is `stop_sign_replica`. See the supplied attack workflow for treatment
names and phase behavior.

#### See also

[Stop-sign patch attack](../examples/stop-sign-patch.md).

### avsectester.implemented_attacks.stop_sign_patch_attack.metrics

<!-- api:begin module=avsectester.implemented_attacks.stop_sign_patch_attack.metrics -->
Replica-specific visibility measurements from saved decision captures.

| Name and signature | Meaning |
|---|---|
| `ReplicaVisibility(Metric)(camera_id: str \| None = None, min_visible_pixels: int = 100)` | Count decisions where the inserted replica reaches a camera's pixel threshold. |
<!-- api:end -->

#### Constraints and errors

Automatic camera selection requires one unique recorded replica-visibility camera. A treatment
with no replica returns an unavailable value, not zero.

#### See also

[Stop-sign patch attack](../examples/stop-sign-patch.md).

### avsectester.implemented_attacks.stop_sign_patch_attack.offline

<!-- api:begin module=avsectester.implemented_attacks.stop_sign_patch_attack.offline -->
Validated detector/rollout optimization options and offline dispatch.

| Name and signature | Meaning |
|---|---|
| `DetectorPatchOptimization(optimizer: GradientOptimizer, scorer_id: str \| None = None, device: str = "cpu", max_training_segments: int \| None = None, samples_per_scene: int = 8, min_visible_pixels: int = 100, braking_deceleration_mps2: float = 3.4)` | White-box attack settings (algorithm used: PGD). |
| `RolloutPatchOptimization(optimizer: GradientOptimizer, start_distances_m: tuple[float, ...] = (40., 30., 20.), horizon_s: float = 3., max_training_segments: int \| None = None)` | Black-box attack settings (algorithm used: NES). |
<!-- api:end -->

#### Constraints and errors

`DetectorPatchOptimization` configures a white-box attack; algorithm used: `PGD`. A `scorer_id`
of None requires exactly one differentiable scorer. The descriptor must declare its image sensor
and decision threshold. `samples_per_scene` is at least four. `braking_deceleration_mps2` sets
the sampling bound `d_min = v² / (2a)`. `RolloutPatchOptimization` configures a black-box attack;
algorithm used: `NES`. The rollout budget is starts × training segments × (1 + steps ×
(`population_size` + 1)). `max_training_segments` caps cost after the experiment split; None
uses all training metadata. Validation metadata is not used.

#### See also

[Stop-sign patch attack](../examples/stop-sign-patch.md).

### avsectester.experiment

<!-- api:begin module=avsectester.experiment -->
Prepare attacks and execute independent closed-loop treatments.

| Name and signature | Meaning |
|---|---|
| `ConfigurationMismatchError(ValueError)` | Saved preparation and supplied execution configuration describe different inputs. |
| `Experiment` | [Experiment](#avsectesterexperimentengine) |
| `ExperimentConfiguration` | [ExperimentConfiguration](#avsectesterexperimentconfig) |
| `MetadataConfiguration` | [MetadataConfiguration](#avsectesterexperimentconfig) |
| `RunsConfiguration` | [RunsConfiguration](#avsectesterexperimentconfig) |
| `SplitConfiguration` | [SplitConfiguration](#avsectesterexperimentconfig) |
| `TreatmentConfiguration` | [TreatmentConfiguration](#avsectesterexperimentconfig) |
| `load_configuration` | [load_configuration](#avsectesterexperimentconfig) |
| `PreparedExperiment` | [PreparedExperiment](#avsectesterexperimentmodels) |
| `ExperimentResult` | [ExperimentResult](#avsectesterexperimentmodels) |
| `SelectionDecision` | [SelectionDecision](#avsectesterexperimentmodels) |
| `SegmentUnit` | [SegmentUnit](#avsectesterexperimentmodels) |
| `TreatmentResult` | [TreatmentResult](#avsectesterexperimentmodels) |
| `evaluate` | [evaluate](#avsectesterexperimentresult_evaluation) |
| `load_metadata` | [load_metadata](#avsectesterexperimentmetadata_file) |
| `save_metadata` | [save_metadata](#avsectesterexperimentmetadata_file) |
<!-- api:end -->

#### See also

[Experiments](../guides/experiments.md).

### avsectester.experiment.config

<!-- api:begin module=avsectester.experiment.config -->
Construct experiment objects from Python mappings or YAML files.

| Name and signature | Meaning |
|---|---|
| `HookConfiguration` | A hook set, a seed-to-hook-set factory, or no hooks. |
| `TreatmentConfiguration(name: str, attack_name: str \| None = None, environment: Mapping[str, Any] = dict())` | One attack, or a clean run, with explicit environment overrides. |
| `MetadataConfiguration(save: Path \| None = None, load: Path \| None = None)` | Save selected units or load one shared dataset-specific metadata file. |
| `SplitConfiguration(train_segments: int, validation_segments: int = 0, seed: int = 0, execute: tuple[str, ...] = ("test",))` | Random segment-unit partition sizes and independently selectable execution parts. |
| `RunsConfiguration(treatments: tuple[TreatmentConfiguration, ...] = (), seeds: tuple[int, ...] = (0,), decision_steps: int \| None = None, environment: Mapping[str, Any] = dict(), recording: RecordingPolicy = RecordingPolicy(), output: Path \| None = None, strict: bool = False)` | What to run on the configured components, and how to record and handle those runs. |
| `ExperimentConfiguration(dataset: Dataset, simulator: Simulator, avstack: DrivingAdapter, attacks: Mapping[str, Attack] = dict(), hooks: HookConfiguration = None, evaluation: EvaluationConfiguration = EvaluationConfiguration(), runs: RunsConfiguration = RunsConfiguration(), split: SplitConfiguration \| None = None, metadata: MetadataConfiguration = MetadataConfiguration())` | Reusable components and run settings; every run owns fresh driving and simulator state. |
| `ExperimentConfiguration.configuration_summary: dict[str, Any]` | Describe compatibility on demand, when saving or loading prepared results. |
| `ExperimentConfiguration.describe_configuration(*, scenarios: Sequence[str] \| None = None, metadata_sha256: str \| None = None) -> dict[str, Any]` | Describe preparation inputs without requiring dataset access. |
| `load_configuration(config: ExperimentConfiguration \| Mapping[str, Any] \| str \| Path) -> ExperimentConfiguration` | Load one experiment; file references are relative to their containing YAML file. |
<!-- api:end -->

#### Constraints and errors

Unknown keys and unsupported metric configurations raise `ValueError`.
Injection, paths and compatibility follow [Configuration](configuration.md).

#### See also

[Configuration reference](configuration.md) and [Experiments](../guides/experiments.md).

### avsectester.experiment.engine

<!-- api:begin module=avsectester.experiment.engine -->
Prepare each attack once, then run independent treatments at its selected starts.

| Name and signature | Meaning |
|---|---|
| `Experiment()` | The fixed build_metadata → run_offline → online experiment workflow. |
| `Experiment.validate(config: ExperimentConfiguration \| Mapping[str, Any] \| str \| Path, resume: bool = False) -> None` | Validate preparation inputs without creating a runtime or connecting to services. |
| `Experiment.prepare(config: ExperimentConfiguration \| Mapping[str, Any] \| str \| Path, resume: bool = False) -> PreparedExperiment` | Select all attacks, partition units and checkpoint each completed offline stage. |
| `Experiment.execute(prepared: PreparedExperiment, resume: bool = False) -> ExperimentResult` | Execute unit/seed groups, retaining matching completed runs when resuming. |
<!-- api:end -->

#### Constraints and errors

`validate` checks inputs without runtimes or service connections. Resume checks saved frames and
compatibility; complete preparation is reused, while progress resumes unfinished offline stages.
Invalid loaded metadata fails preparation directly. Unattacked runs create no `AttackRuntime`.

#### See also

[Experiments](../guides/experiments.md).

### avsectester.experiment.html_report

<!-- api:begin module=avsectester.experiment.html_report -->
Build human-readable HTML reports solely from saved experiment files.

| Name and signature | Meaning |
|---|---|
| `write_offline_html(output_root: str \| Path) -> Path` | Rebuild the offline page and series curves solely from saved entries. |
| `write_evaluation_html(output_root: str \| Path) -> Path` | Rebuild root evaluation_report.html from saved evaluation and preparation files. |
<!-- api:end -->

#### Constraints and errors

Rebuild functions read saved files only and write the two top-level pages. Relative file/image
links remain portable within the output tree.

#### See also

[Experiments](../guides/experiments.md).

### avsectester.experiment.metadata_file

<!-- api:begin module=avsectester.experiment.metadata_file -->
Share selected segments as a dataset-specific, versioned JSON metadata file.

| Name and signature | Meaning |
|---|---|
| `save_metadata(segments: Sequence[AttackSegment], path: str \| Path, *, dataset_type: str) -> None` | Save ordered segments without overwriting a different existing document. |
| `load_metadata(path: str \| Path) -> tuple[AttackSegment, ...]` | Load ordered segments after validating their file format and dataset type declaration. |
<!-- api:end -->

#### Constraints and errors

Duplicate segments raise `ValueError` with both item indexes. File format and compatibility
follow [Metadata configuration](configuration.md#metadata).

#### See also

[Configuration reference](configuration.md) and [Experiments](../guides/experiments.md).

### avsectester.experiment.models

<!-- api:begin module=avsectester.experiment.models -->
Preparation inputs, their validation, prepared metadata and per-treatment results.

| Name and signature | Meaning |
|---|---|
| `SelectionDecision(attack_name: str, scenario_id: str, status: Literal["selected", "not_selected", "error"], reasons: tuple[str, ...] = (), error: str \| None = None)` | One metabuilder outcome for an attack and scenario, including any rejection or error. |
| `SelectionDecision.as_dict() -> dict[str, Any]` | Return the shared plain-data representation used by preparation and reports. |
| `SelectionDecision.from_dict(value: Mapping[str, Any]) -> SelectionDecision` | Restore a saved selection outcome. |
| `SegmentUnit(index: int, segment: AttackSegment, attack_names: tuple[str, ...], part: str \| None = None)` | One canonical segment shared by all attacks that selected exactly that segment. |
| `SegmentUnit.as_dict() -> dict[str, Any]` | Serialize this unit's index, segment identity, attack membership and split part. |
| `PreparedExperiment(configuration: ExperimentConfiguration, metadata: Mapping[str, tuple[AttackSegment, ...]], artifacts: Mapping[str, Any], selections: tuple[SelectionDecision, ...] = (), *, preparation: Mapping[str, Any], units: tuple[SegmentUnit, ...] = (), deployment: DeploymentPlan \| None = None, offline_reports: Mapping[str, OfflineReport] = dict(), metadata_source: Mapping[str, str] \| None = None)` | Configured factories, selected metadata and artifacts, never a live simulator or session. |
| `PreparedExperiment.save(directory: str \| Path) -> Path` | Save preparation metadata, artifacts and offline reports to an output directory. |
| `PreparedExperiment.load(config: ExperimentConfiguration \| Mapping[str, Any] \| str \| Path, directory: str \| Path) -> PreparedExperiment` | Load compatible metadata and artifacts without selecting or optimizing again. |
| `TreatmentResult(attack_name: str \| None, segment_index: int, segment: AttackSegment, treatment: str, seed: int, log: RunLog, error: str \| None = None, part: str \| None = None)` | The recorded outcome of one segment, seed and named treatment. |
| `TreatmentResult.group: tuple[int, int]` | Comparison-group key containing the segment index and seed. |
| `ExperimentResult(runs: tuple[TreatmentResult, ...] = (), selections: tuple[SelectionDecision, ...] = (), notices: tuple[str, ...] = (), scenario_maps: Mapping[str, ScenarioMap \| Unavailable] = dict())` | Treatment runs, selection outcomes and saved maps for scenarios with selected segments. |
| `ExperimentResult.groups: dict[tuple[int, int], dict[str, RunLog]]` | Group treatment results by segment index and seed. |
| `ExperimentResult.failures: tuple[TreatmentResult, ...]` | Failed or incomplete treatment runs. |
| `ExperimentResult.selection_errors: tuple[SelectionDecision, ...]` | Selection decisions whose status is error. |
| `ExperimentResult.load(output: str \| Path, configuration: ExperimentConfiguration) -> ExperimentResult` | Read results without loading artifacts, driving models or simulator runtimes. |
<!-- api:end -->

#### Constraints and errors

`PreparedExperiment` owns metadata, artifacts and reports, never live resources. Canonically
serialized segment data, including `user_data`, defines units. `TreatmentResult.group` is
(`segment_index`, seed).

#### See also

[Experiments](../guides/experiments.md).

### avsectester.experiment.result_evaluation

<!-- api:begin module=avsectester.experiment.result_evaluation -->
Organize experiment results for evaluation, saving and visualization.

| Name and signature | Meaning |
|---|---|
| `evaluate(result: ExperimentResult, configuration: ExperimentConfiguration) -> EvaluationReport` | Group and order experiment logs, evaluate them, and save reports and plots. |
<!-- api:end -->

#### Constraints and errors

Evaluation consumes saved logs/maps and saves structured and HTML reports without dataset,
simulator or model connections.

#### See also

[Experiments](../guides/experiments.md).

### avsectester.experiment.type_registry

<!-- api:begin module=avsectester.experiment.type_registry -->
Resolve component names and discover attack classes without constructing components.

| Name and signature | Meaning |
|---|---|
| `resolve_type(value: Any) -> Callable[..., Any]` | Resolve a built-in class name, or user code written as `package.module:name`. |
| `available_attacks() -> tuple[str, ...]` | List exactly the exported Attack classes accepted by class-name lookup. |
<!-- api:end -->

#### Constraints and errors

Built-in short names identify framework components. Custom constructors use
`package.module:Class`. Concrete attacks follow the
[implemented-attack directory rules](../../avsectester/implemented_attacks/README.md).

#### See also

[Configuration reference](configuration.md) and [Experiments](../guides/experiments.md).

### avsectester.cli

<!-- api:begin module=avsectester.cli -->
Prepare, execute and evaluate experiments through the public experiment API.

| Name and signature | Meaning |
|---|---|
| `app` | Command-line application and its service subcommands. |
<!-- api:end -->

#### Constraints and errors

Exit codes and automatic service ownership follow the [CLI reference](cli.md).

#### See also

[CLI reference](cli.md).

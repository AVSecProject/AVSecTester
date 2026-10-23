"""Public sensor-view assertions shared by service and simulator contract tests."""

import numpy as np
import pytest






@pytest.fixture
def assert_inserted_world_object(make_object_in_front):
    """The same public insertion/capture contract for memory and native simulators."""
    from avsectester.data_types import Control, read_camera_rgb
    from avsectester.geometry.projection import project_box
    from avsectester.rendering import PlacedObject, composite_objects

    def check(runtime, start=None, initialization=None):
        runtime.reset(start, initialization)
        original = runtime.sensor_view("front")
        non_camera_data = {sensor: runtime.sensor_view(sensor).data
                           for sensor in runtime.sensors - runtime.camera_sensors}
        asset, world_from_object = make_object_in_front(original)
        runtime.step(Control())
        clean_first = runtime.sensor_view("front")
        runtime.step(Control())
        clean_second = runtime.sensor_view("front")
        runtime.reset(start, initialization)

        runtime.insert_object("sign", asset, world_from_object)
        with pytest.raises(KeyError, match="next capture"):
            runtime.object_visibility("sign")
        state = runtime.state().objects["sign"]
        np.testing.assert_allclose(state.pose, world_from_object)
        np.testing.assert_array_equal(state.velocity, [0., 0., 0.])
        assert state.category == "stop_sign" and state.extent == asset.extent
        assert state.box_center == (0., 0., 1.)
        np.testing.assert_array_equal(runtime.sensor_view("front").data, original.data)
        with pytest.raises(ValueError, match="sign"):
            runtime.insert_object("sign", asset, world_from_object)

        runtime.step(Control())
        view = runtime.sensor_view("front")
        expected = composite_objects(clean_first.data, clean_first.camera,
            clean_first.cam_from_world, (PlacedObject("sign", asset, world_from_object),),
            scene_depth=clean_first.depth, occluders=clean_first.actors)
        np.testing.assert_array_equal(view.data, expected.image)
        planned = clean_first.composite_plan((PlacedObject("sign", asset, world_from_object),))
        np.testing.assert_array_equal(planned.apply(clean_first.data).image, view.data)
        # The view records the capture beneath its objects and the plan that composed them.
        np.testing.assert_array_equal(view.composition.background, clean_first.data)
        np.testing.assert_array_equal(view.composition.plan.visible_mask, planned.visible_mask)
        np.testing.assert_array_equal(view.composition.plan.apply(view.composition.background).image, view.data)
        assert clean_first.composition is None
        with pytest.raises(ValueError, match="already contains"):
            view.composite_plan((PlacedObject("sign", asset, world_from_object),))
        assert runtime.object_visibility("sign")["front"] == expected.object_visibility["sign"]
        assert view.objects.keys() == {"sign"}
        np.testing.assert_array_equal(view.objects["sign"].world_from_object, world_from_object)
        np.testing.assert_array_equal(read_camera_rgb(runtime.observe(), "front")[1], view.data)
        for sensor, data in non_camera_data.items():
            np.testing.assert_array_equal(runtime.sensor_view(sensor).data, data)
        changed = np.any(view.data != clean_first.data, axis=2)
        assert np.any(changed)
        world_from_box = world_from_object.copy()
        world_from_box[:3, 3] += world_from_object[:3, :3] @ np.array([0., 0., 1.])
        x0, y0, x1, y1 = project_box(view.camera, view.cam_from_world, world_from_box, asset.extent)
        ys, xs = np.nonzero(changed)
        assert xs.min() >= np.floor(x0) and xs.max() < np.ceil(x1)
        assert ys.min() >= np.floor(y0) and ys.max() < np.ceil(y1)

        runtime.remove_object("sign")
        assert runtime.state().objects == {}
        np.testing.assert_array_equal(runtime.sensor_view("front").data, view.data)
        assert runtime.sensor_view("front").objects.keys() == {"sign"}
        with pytest.raises(KeyError, match="sign"):
            runtime.remove_object("sign")
        runtime.step(Control())
        np.testing.assert_array_equal(runtime.sensor_view("front").data, clean_second.data)
        assert runtime.sensor_view("front").objects == {}
        return clean_first, view, expected
    return check

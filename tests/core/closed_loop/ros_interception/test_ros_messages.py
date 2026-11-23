from array import array
from types import SimpleNamespace

import numpy as np
import pytest

from avsectester.closed_loop.ros_interception.ros_messages import (
    OCCUPANCY_GRID, POINT_CLOUD, publishable_fields, to_fields,
)


class NativeMessage(SimpleNamespace):
    def get_fields_and_field_types(self):
        return vars(self).keys()


def test_point_cloud_fields_preserve_layout_and_recompute_modified_dimensions():
    layout = [NativeMessage(name="x", offset=0, datatype=7, count=1),
              NativeMessage(name="ring", offset=4, datatype=4, count=2)]
    data = bytearray(36)
    source = np.ndarray((2, 2), dtype=np.dtype([("x", ">f4"), ("ring", ">u2", 2)]),
                        buffer=data, strides=(20, 8))
    source["x"] = [[10., 20.], [30., 40.]]
    source["ring"] = (1, 2)
    cloud = NativeMessage(height=2, width=2, point_step=8, row_step=20, is_bigendian=True,
                          fields=layout, data=data)
    fields = to_fields(cloud, POINT_CLOUD)
    np.testing.assert_equal(fields["data"], source)

    fields["data"] = fields["data"][:, 1]
    new = publishable_fields(fields, POINT_CLOUD)
    assert (new["height"], new["width"], new["point_step"], new["row_step"]) == (1, 2, 8, 16)
    assert new["fields"] == fields["fields"]
    np.testing.assert_equal(np.frombuffer(new["data"], dtype=source.dtype), source[:, 1])


def test_occupancy_grid_fields_preserve_int8_and_map_dimensions():
    source = NativeMessage(info=NativeMessage(height=2, width=2), data=array("b", [-1, 0, 1, 100]))
    fields = to_fields(source, OCCUPANCY_GRID)
    assert fields["data"].dtype == np.int8
    edited = fields["data"].reshape(2, 2)
    edited[1, 1] = 75
    np.testing.assert_equal(publishable_fields({**fields, "data": edited}, OCCUPANCY_GRID)["data"],
                            np.array([-1, 0, 1, 75], dtype=np.int8))
    with pytest.raises(ValueError, match="matching its height and width"):
        publishable_fields({**fields, "data": edited[0]}, OCCUPANCY_GRID)

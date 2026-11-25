"""WGS84 conversion to Earth-fixed and local tangent coordinates."""

import numpy as np

from avsectester.geometry.geodesy import ecef_from_geodetic, enu_from_ecef


def test_ecef_from_geodetic():
    np.testing.assert_allclose(ecef_from_geodetic(0., 90., 50.), [0., 6378187., 0.], atol=1e-8)
    np.testing.assert_allclose(
        ecef_from_geodetic(90., 0., 0.), [0., 0., 6356752.314245179], atol=1e-8,
    )


def test_enu_from_ecef():
    enu = enu_from_ecef(0., 90., 50.)
    np.testing.assert_allclose(enu @ [0., 6378187., 0., 1.], [0., 0., 0., 1.], atol=1e-8)
    np.testing.assert_allclose(enu @ [-3., 6378192., 7., 1.], [3., 7., 5., 1.], atol=1e-8)

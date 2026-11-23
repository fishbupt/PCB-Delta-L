import numpy as np
from types import SimpleNamespace
from pcb_delta_l.comparison import DeltaLData, read_delta_l, write_plts, align


def test_plts_roundtrip_and_undefined_values(tmp_path):
    raw = SimpleNamespace(frequency_hz=np.array([1e9, 2e9]), loss_db_per_in=np.array([.1, .2]))
    fit = SimpleNamespace(fitted_db_per_in=np.array([.11, np.nan]), residual_percent=np.array([1., np.nan]))
    path = tmp_path / 'data.csv'
    write_plts(path, raw, fit)
    data = read_delta_l(path, 'MHz')
    np.testing.assert_array_equal(data.frequency_hz, raw.frequency_hz)
    np.testing.assert_allclose(data.values[:, 0], raw.loss_db_per_in)
    assert np.isnan(data.values[1, 1:]).all()


def test_alignment_does_not_interpolate_over_gaps_or_extrapolate():
    a = DeltaLData(np.arange(7.), np.zeros((7, 3)))
    b = DeltaLData(np.arange(1., 6.), np.tile([1., 2., np.nan, 4., 5.], (3, 1)).T)
    actual = align(a, b)
    np.testing.assert_allclose(actual[:, 0], [np.nan, 1, 2, np.nan, 4, 5, np.nan], equal_nan=True)

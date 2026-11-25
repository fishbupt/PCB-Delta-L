import numpy as np
import pytest
from skrf import Network

from pcb_delta_l.comparison import read_delta_l, write_plts
from pcb_delta_l.core import extract_2l
from pcb_delta_l.fitting import fit_loss
from pcb_delta_l.simulation import generate, generate_stress
from pcb_delta_l.touchstone import write_s4p


def test_recovers_known_fitting_model():
    f = np.linspace(1e7, 40e9, 801)
    x = (f - f[0]) / (f[-1] - f[0])
    y = .4 * x**.55 + .15 * x*x + .8*x + .03
    result = fit_loss(f, y)
    expected = y.copy()
    expected[1] = y[:3].mean()
    np.testing.assert_allclose(result.fitted_db_per_in, expected, atol=1e-12)
    assert result.rms_db_per_in < 1e-12
    assert np.nanmax(result.uncertainty_percent) < 1e-9


def test_uncertainty_window_and_cutoff():
    f = np.arange(101) * 1e8
    y = .5 + np.sqrt(f / 1e9)
    y[20] += 0.2
    y[4] = np.nan
    result = fit_loss(f, y, cutoff_hz=8e9, weight_order=3)
    assert np.isnan(result.uncertainty_percent[4])
    assert np.isnan(result.uncertainty_percent[10])
    assert result.uncertainty_percent[9] == result.uncertainty_percent[71] == 0
    assert np.isnan(result.fitted_db_per_in[f > 8e9]).all()
    residual = y[10:31] - result.fitted_db_per_in[10:31]
    expected = 100 * (abs(residual.mean()) + 3 * residual.std()) / result.fitted_db_per_in[20]
    assert result.uncertainty_percent[20] == pytest.approx(expected)
    assert result.uncertainty_percent[20] > 1


def test_insufficient_points_and_zero_denominator():
    with pytest.raises(ValueError):
        fit_loss(np.arange(5.), np.ones(5))
    result = fit_loss(np.arange(10.), np.zeros(10))
    assert np.isnan(result.uncertainty_percent).all()


@pytest.mark.parametrize('exponent', [0, 1, np.nan, np.inf])
def test_fixed_exponent_must_be_between_zero_and_one(exponent):
    with pytest.raises(ValueError, match='固定拟合指数'):
        fit_loss(np.arange(10.), np.ones(10), exponent=exponent)


@pytest.mark.parametrize('short_in,long_in', [(5, 10), (2, 5)])
def test_default_simulation_reproduces_plts_export(tmp_path, short_in, long_in):
    f = np.linspace(1e7, 40e9, 1601)
    networks, _ = generate(f, lengths=(short_in, long_in))
    loaded = {}
    for length, network in networks.items():
        path = tmp_path / f'line_{length}in.s4p'
        write_s4p(path, network)
        loaded[length] = Network(str(path))
    raw = extract_2l(loaded[short_in], loaded[long_in], short_in, long_in)
    fit = fit_loss(raw.frequency_hz, raw.loss_db_per_in, raw.valid)
    path = tmp_path / 'delta_l.csv'
    write_plts(path, raw, fit)
    exported = read_delta_l(path)
    # Representative rows from the supplied PLTS CSVs, which differ only in
    # the final digits at these points. Boundary and interior rows are included.
    points = [0, 1, 2, 39, 40, 41, 42, 65, 66, 127, 400, 800, 892, 893, 1000, 1200, 1437, 1438, 1560, 1561, 1600]
    expected_fit = [
        .018350000000138, .027237086923843, .034370387794420, .124102476534867,
        .125958713758223, .127803408277515, .129636966722257, .169408837791400,
        .171055152019463, .263949952315289, .613395392323367, 1.068033085161023,
        1.168475363956756, 1.169561314877121, 1.285115455731853, 1.498197292192826,
        1.746915900170385, 1.747958205965093, 1.874731001900552, 1.875767087264788,
        1.916138217707867,
    ]
    expected_uncertainty = [
        0, 0, 0, 0, 2.165446018411587, 2.136545722832163, 1.980469286945918,
        .423777185060448, .405042990884432, .105739926403202, .017128558937251,
        .006916820758332, .002755266745037, .002773327467398, .005568081021212,
        .005861573996005, .003055368072909, .003079256549003, .011149793821659, 0, 0,
    ]
    assert fit.coefficients[1] == .55
    np.testing.assert_allclose(exported.frequency_hz, f, atol=1e-3, rtol=0)
    assert exported.values[1, 0] == pytest.approx(.025411, abs=5e-7)
    np.testing.assert_allclose(exported.values[points, 1], expected_fit, atol=1e-9, rtol=0)
    np.testing.assert_allclose(exported.values[points, 2], expected_uncertainty, atol=1e-7, rtol=0)
    assert not np.count_nonzero(exported.values[:40, 2])
    assert not np.count_nonzero(exported.values[1561:, 2])


def test_tc_2l_007_fixed_exponent_matches_plts(tmp_path):
    f = np.linspace(1e7, 40e9, 1601)
    samples = generate_stress(f, (5, 10), '007', z_line=100, fixture_z=100)
    short_path = tmp_path / 'short.s4p'
    long_path = tmp_path / 'long.s4p'
    write_s4p(short_path, samples.networks[5])
    write_s4p(long_path, samples.networks[10])
    raw = extract_2l(Network(str(short_path)), Network(str(long_path)), 5, 10)
    fit = fit_loss(raw.frequency_hz, raw.loss_db_per_in, raw.valid, exponent=0.540625)
    path = tmp_path / 'delta_l.csv'
    write_plts(path, raw, fit)
    exported = read_delta_l(path)

    # PLTS rows cover the displayed second point, the first complete window,
    # the low-frequency deviation, the noise band and the upper window edge.
    points = [1, 40, 43, 1200, 1558, 1560, 1561, 1600]
    expected_fit = [
        0.029140065733862, 0.176402910330183, 0.185635728096272,
        2.996965022096072, 3.815655871972433, 3.820204924673363,
        3.822479359103224, 3.911135050365190,
    ]
    expected_uncertainty = [
        0, 1.761817576393709, 1.496853624562001, 1.972468781008116,
        4.973910515044153, 5.162013572475937, 0, 0,
    ]
    assert fit.coefficients[1] == 0.540625
    np.testing.assert_allclose(exported.values[points, 1], expected_fit, atol=1e-9, rtol=0)
    np.testing.assert_allclose(exported.values[points, 2], expected_uncertainty, atol=1e-7, rtol=0)
    assert not np.count_nonzero(exported.values[:40, 2])
    assert not np.count_nonzero(exported.values[1561:, 2])

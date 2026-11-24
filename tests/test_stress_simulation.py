from dataclasses import replace

import numpy as np
import pytest

from pcb_delta_l.core import MIX
from pcb_delta_l.simulation import StressOptions, generate, generate_stress


FREQUENCIES = np.linspace(10e6, 40e9, 1601)
LENGTHS = (5., 10.)


def mixed(net):
    return MIX @ net.s @ MIX.T


def test_common_asymmetric_launch_introduces_mode_conversion():
    baseline, _ = generate(FREQUENCIES, LENGTHS, z_line=100, fixture_z=100)
    result = generate_stress(FREQUENCIES, LENGTHS, '005')
    for length, net in result.networks.items():
        before, after = mixed(baseline[length]), mixed(net)
        assert np.abs(before[:, 3, 0]).max() < 1e-12
        assert np.abs(after[:, 3, 0]).max() > .05
        assert np.abs(after[:, 1, 2]).max() > .05
        np.testing.assert_allclose(net.s, net.s.transpose(0, 2, 1), atol=1e-12)


@pytest.mark.parametrize('case', ['006', '011'])
def test_glitch_touches_one_frequency_of_the_longest_line(case):
    options = StressOptions(glitch_ghz=29.8, glitch_db=4, glitch_phase_deg=-30)
    result = generate_stress(FREQUENCIES, LENGTHS[::-1], case, options)
    np.testing.assert_array_equal(result.networks[5.].s, result.reference[5.].s)
    actual = result.networks[10.].s
    clean = result.reference[10.].s
    index = np.argmin(np.abs(FREQUENCIES - 29.8e9))
    np.testing.assert_array_equal(np.flatnonzero(np.any(actual != clean, axis=(1, 2))), [index])
    factor = 10 ** (-4 / 20) * np.exp(-1j * np.pi / 6)
    for i, j in ((1, 0), (0, 1)):
        np.testing.assert_allclose((MIX @ actual[index] @ MIX.T)[i, j],
                                   factor * (MIX @ clean[index] @ MIX.T)[i, j], atol=1e-12)


def test_high_loss_sample_noise_is_repeatable_and_band_limited():
    first = generate_stress(FREQUENCIES, LENGTHS, '007')
    repeated = generate_stress(FREQUENCIES, LENGTHS[::-1], '007')
    changed = generate_stress(FREQUENCIES, LENGTHS, '007', replace(StressOptions(), noise_seed=99))
    low = FREQUENCIES < 30e9
    for length in LENGTHS:
        actual, clean = first.networks[length], first.reference[length]
        np.testing.assert_array_equal(actual.s, repeated.networks[length].s)
        np.testing.assert_array_equal(actual.s[low], clean.s[low])
        np.testing.assert_array_equal(clean.s, changed.reference[length].s)
        assert not np.array_equal(actual.s[~low], changed.networks[length].s[~low])
        residual = mixed(actual)[~low, 1, 0] - mixed(clean)[~low, 1, 0]
        rms_db = 10 * np.log10(np.mean(np.abs(residual) ** 2))
        assert -66 < rms_db < -64
    for net in (first.networks[10.], first.reference[10.]):
        assert -45 <= 20 * np.log10(abs(mixed(net)[-1, 1, 0])) <= -35


def test_periodic_impedance_adds_reflection_to_both_lines():
    baseline, baseline_loss = generate(FREQUENCIES, LENGTHS, z_line=100, fixture_z=100)
    result = generate_stress(FREQUENCIES, LENGTHS, '009')
    np.testing.assert_array_equal(result.material_loss, baseline_loss)
    for length in LENGTHS:
        assert np.abs(mixed(baseline[length])[:, 0, 0]).max() < 1e-12
        assert np.abs(mixed(result.networks[length])[:, 0, 0]).max() > .05


def test_skew_reference_has_notch_near_expected_frequency():
    result = generate_stress(FREQUENCIES, LENGTHS, '011')
    short, long = (np.abs(mixed(result.reference[length])[:, 1, 0]) for length in LENGTHS)
    band = np.flatnonzero((FREQUENCIES >= 25e9) & (FREQUENCIES <= 40e9))
    notch = band[np.argmin(long[band])]
    assert abs(FREQUENCIES[notch] - 31.25e9) < FREQUENCIES[1] - FREQUENCIES[0]
    assert 20 * np.log10(long[notch] / short[notch]) < -40

"""Synthetic coupled-mode fixtures and transmission lines, with known loss."""
from dataclasses import dataclass

import numpy as np
import skrf as rf
from .core import MIX


def line_abcd(f, length_in, z, dc, skin, dielectric, er):
    ghz=f/1e9
    db_per_in=dc+skin*np.sqrt(ghz)+dielectric*ghz
    gamma=(db_per_in/8.685889638 + 1j*2*np.pi*f*np.sqrt(er)*0.0254/299792458)*length_in
    ch=np.cosh(gamma); sh=np.sinh(gamma)
    a=np.empty((len(f),2,2),complex); a[:,0,0]=ch; a[:,1,1]=ch; a[:,0,1]=z*sh; a[:,1,0]=sh/z
    return a,db_per_in


def _periodic_line_abcd(f, length, z, dc, skin, dielectric, er, ripple, period):
    """Cascade an eight-segment approximation of one sinusoidal impedance period."""
    segment_length = period / 8
    offsets = ripple * np.sin(2 * np.pi * (np.arange(8) + .5) / 8)
    cycle = np.broadcast_to(np.eye(2, dtype=complex), (len(f), 2, 2))
    for offset in offsets:
        segment, _ = line_abcd(f, segment_length, z + offset, dc, skin, dielectric, er)
        cycle = cycle @ segment
    whole, remainder = divmod(length, period)
    result = np.linalg.matrix_power(cycle, int(whole))
    for index, offset in enumerate(offsets):
        part = min(segment_length, remainder - index * segment_length)
        if part <= 0:
            break
        segment, _ = line_abcd(f, part, z + offset, dc, skin, dielectric, er)
        result = result @ segment
    return result


def generate(f, lengths=(2.,5.,10.), z_line=95., er=3.5, dc=0.01, skin=0.08, dielectric=0.035, fixture_z=85., fixture_length=0.15, length_impedances=None, *, periodic_impedance=None):
    f=np.asarray(f,float)
    if len(f)<2 or not np.all(np.isfinite(f)) or np.any(f<=0) or np.any(np.diff(f)<=0): raise ValueError('频率须为正且严格递增，至少两个点')
    if not all(np.isfinite(x) and x>=0 for x in (dc,skin,dielectric,fixture_length)) or not all(np.isfinite(x) and x>0 for x in (z_line,er,fixture_z,*lengths)): raise ValueError('长度、阻抗和介电常数须为正，损耗系数须非负')
    length_impedances = {} if length_impedances is None else dict(length_impedances)
    if any(not np.isfinite(v) or v <= 0 for v in length_impedances.values()): raise ValueError('各线阻抗必须为正且有限')
    freq=rf.Frequency.from_f(f,unit='hz'); out={}
    fixture,_=line_abcd(f,fixture_length,fixture_z,dc,skin,dielectric,er)
    for length in lengths:
        z = length_impedances.get(length, z_line)
        line,truth=line_abcd(f,length,z,dc,skin,dielectric,er)
        if periodic_impedance is not None:
            ripple, period = periodic_impedance
            if not (np.isfinite(ripple) and 0 < ripple < z and np.isfinite(period) and period > 0):
                raise ValueError('周期波动幅度须大于 0 并小于走线阻抗，周期须为正')
            line = _periodic_line_abcd(f, length, z, dc, skin, dielectric, er, ripple, period)
        diff=fixture @ line @ fixture
        sd=rf.network.a2s(diff,z0=100)
        # Independent common-mode line. Modal conversion is deliberately zero.
        common,_=line_abcd(f,length+2*fixture_length,25,dc,skin,dielectric,er)
        sc=rf.network.a2s(common,z0=25)
        sm=np.zeros((len(f),4,4),complex); sm[:,:2,:2]=sd; sm[:,2:,2:]=sc
        s=MIX.T @ sm @ MIX
        out[length]=rf.Network(frequency=freq,s=s,z0=50,name=f'line_{length:g}in')
    return out,truth


@dataclass(frozen=True)
class StressOptions:
    launch_skew_ps: float = 2.0
    glitch_ghz: float = 30.0
    glitch_db: float = 3.0
    glitch_phase_deg: float = 20.0
    noise_start_ghz: float = 30.0
    noise_floor_db: float = -65.0
    noise_seed: int = 2026
    ripple_ohm: float = 2.0
    period_in: float = 0.3
    skew_ps_per_in: float = 1.6


@dataclass
class StressSamples:
    networks: dict[float, rf.Network]
    reference: dict[float, rf.Network]
    material_loss: np.ndarray
    notes: list[str]


def _delay_positive_path(network, delay_per_end_ps):
    # Matched delay sections on the input+ and output+ ports.
    phase = np.exp(-2j * np.pi * network.f * delay_per_end_ps * 1e-12)
    factors = np.ones((len(phase), 4), dtype=complex)
    factors[:, :2] = phase[:, None]
    shifted = network.copy()
    shifted.s = network.s * factors[:, :, None] * factors[:, None, :]
    return shifted


def _apply_sdd_transmission(network, indices, change):
    """Change reciprocal SDD21/12 at selected points, preserving other mode terms."""
    changed = network.copy()
    mixed = MIX @ network.s[indices] @ MIX.T
    mixed[:, 1, 0] = change(mixed[:, 1, 0])
    mixed[:, 0, 1] = change(mixed[:, 0, 1])
    changed.s[indices] = MIX.T @ mixed @ MIX
    return changed


def _db(value):
    with np.errstate(divide='ignore'):
        return 20 * np.log10(np.abs(value))


def generate_stress(f, lengths, case, options=StressOptions(), **line_options):
    """Generate test cases 005, 006, 007, 009 and 011 and paired references."""
    if case not in {'005', '006', '007', '009', '011'}:
        raise ValueError('未知仿真用例')
    f = np.asarray(f, dtype=float)
    lengths = tuple(sorted(lengths))
    if len(lengths) < 2 or len(set(lengths)) != len(lengths):
        raise ValueError('至少需要两个不同长度')
    longest = lengths[-1]
    line_options.setdefault('z_line', 100.)
    line_options.setdefault('fixture_z', 100.)
    if case == '007':
        # Compared with the T1 material, add 2 dB/in at 40 GHz.
        line_options['dielectric'] = line_options.get('dielectric', .035) + .05
    if case == '009':
        line_options['periodic_impedance'] = (options.ripple_ohm, options.period_in)
    networks, truth = generate(f, lengths, **line_options)
    reference = {}
    notes = []

    if case == '005':
        delay = options.launch_skew_ps
        if not np.isfinite(delay) or delay <= 0:
            raise ValueError('每端 P/N 引入时延差须为正')
        networks = {length: _delay_positive_path(net, delay) for length, net in networks.items()}
        notes.append(f'长短线每端共用 {delay:g} ps 的 P/N 引入时延差。')
        for length, net in networks.items():
            s = MIX @ net.s @ MIX.T
            scd, sdc = _db(s[:, 3, 0]), _db(s[:, 1, 2])
            cd_idx, dc_idx = np.argmax(scd), np.argmax(sdc)
            notes.append(f'{length:g} in：SCD21 峰值 {scd[cd_idx]:.2f} dB @ {f[cd_idx]/1e9:.6g} GHz，'
                         f'SDC21 峰值 {sdc[dc_idx]:.2f} dB @ {f[dc_idx]/1e9:.6g} GHz。')

    if case == '009':
        notes.append(f'差模阻抗周期波动：正弦幅度 ±{options.ripple_ohm:g} Ω，'
                     f'周期 {options.period_in:g} in，两种线长均从同一相位开始。')

    if case == '011':
        skew = options.skew_ps_per_in
        if not np.isfinite(skew) or skew <= 0:
            raise ValueError('单位长度 P/N 时延差须为正')
        networks = {length: _delay_positive_path(net, skew * length / 2)
                    for length, net in networks.items()}
        delay = skew * longest
        nominal_notch_ghz = 500 / delay
        notes.append(f'{longest:g} in 线的 P/N 时延差为 {delay:g} ps，'
                     f'理想匹配时首个凹陷约 {nominal_notch_ghz:.6g} GHz。')
        band = np.flatnonzero((f >= 25e9) & (f <= 40e9))
        if len(band):
            s = MIX @ networks[longest].s @ MIX.T
            idx = band[np.argmin(np.abs(s[band, 1, 0]))]
            notes.append(f'最长线叠加毛刺前，25–40 GHz 最深 SDD21 '
                         f'{_db(s[idx, 1, 0]):.2f} dB @ {f[idx]/1e9:.6g} GHz。')

    if case in {'006', '011'}:
        target = options.glitch_ghz * 1e9
        if not f[0] <= target <= f[-1]:
            raise ValueError('毛刺频率须在扫频范围内')
        if not np.isfinite(options.glitch_db) or options.glitch_db < 0 or not np.isfinite(options.glitch_phase_deg):
            raise ValueError('毛刺附加衰减须为非负，附加相位须为有限数值')
        idx = int(np.argmin(np.abs(f - target)))
        reference = networks.copy()
        factor = 10 ** (-options.glitch_db / 20) * np.exp(1j * np.deg2rad(options.glitch_phase_deg))
        networks[longest] = _apply_sdd_transmission(networks[longest], [idx], lambda values: values * factor)
        notes.append(f'仅在 {longest:g} in 线的 {f[idx]/1e9:.9g} GHz 频点注入毛刺：'
                     f'附加衰减 {options.glitch_db:g} dB，附加相位 {options.glitch_phase_deg:g}°。')

    if case == '007':
        if not np.isfinite(options.noise_start_ghz) or options.noise_start_ghz < 0:
            raise ValueError('噪声起始频率须为非负且有限')
        indices = np.flatnonzero(f >= options.noise_start_ghz * 1e9)
        if not len(indices):
            raise ValueError('扫频范围内没有达到噪声起始频率的频点')
        if not np.isfinite(options.noise_floor_db) or options.noise_floor_db >= 0:
            raise ValueError('复噪声 RMS 电平须为负 dB')
        if not isinstance(options.noise_seed, (int, np.integer)) or options.noise_seed < 0:
            raise ValueError('随机种子须为非负整数')
        reference = networks.copy()
        rng = np.random.default_rng(options.noise_seed)
        rms = 10 ** (options.noise_floor_db / 20)
        for length, net in networks.items():
            noise = rms / np.sqrt(2) * (rng.standard_normal(len(indices)) + 1j * rng.standard_normal(len(indices)))
            networks[length] = _apply_sdd_transmission(net, indices, lambda values: values + noise)
        notes.append(f'复噪声注入 SDD21/12：起始 {options.noise_start_ghz:g} GHz，'
                     f'RMS {options.noise_floor_db:g} dB，随机种子 {options.noise_seed}。')
        if f[0] <= 40e9 <= f[-1]:
            idx = int(np.argmin(np.abs(f - 40e9)))
            clean = MIX @ reference[longest].s[idx] @ MIX.T
            noisy = MIX @ networks[longest].s[idx] @ MIX.T
            notes.append(f'{longest:g} in 线在 {f[idx]/1e9:.9g} GHz 的 SDD21：'
                         f'无噪声 {_db(clean[1, 0]):.2f} dB，含噪声 {_db(noisy[1, 0]):.2f} dB。')

    return StressSamples(networks, reference, truth, notes)

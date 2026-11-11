"""Conservative Touchstone 1.0 export for four-port single-ended data."""
from pathlib import Path
import numpy as np

MAX_PLTS_POINTS = 16000

def write_s4p(path, network):
    """Write 4 S-matrix rows per frequency, using ASCII and CRLF line endings."""
    p = Path(path)
    f = np.asarray(network.f)
    s = np.asarray(network.s)
    z = np.asarray(network.z0)
    if p.suffix.lower() != '.s4p' or network.nports != 4:
        raise ValueError('输出需要四端口网络与 .s4p 扩展名')
    if len(f) < 2 or len(f) > MAX_PLTS_POINTS:
        raise ValueError(f'PLTS 兼容导出要求 2–{MAX_PLTS_POINTS} 个频点')
    if not np.all(np.isfinite(f)) or np.any(f < 0) or np.any(np.diff(f) <= 0) or not np.all(np.isfinite(s)):
        raise ValueError('频率必须递增，且 S 参数必须为有限数值')
    if not np.allclose(z,50):
        raise ValueError('本导出器要求各端口参考阻抗均为 50 Ω')
    with p.open('w', encoding='ascii', newline='\r\n') as out:
        out.write('# Hz S RI R 50\n')
        for hz, matrix in zip(f, s):
            for row in range(4):
                prefix = f'{hz:.12e} ' if row == 0 else ' '
                pairs = ' '.join(f'{v.real:.15e} {v.imag:.15e}' for v in matrix[row])
                out.write(prefix + pairs + '\n')
    return p

def length_filename(length):
    # Avoid decimal points in the basename, including for older file parsers.
    return 'line_' + format(float(length), '.12g').replace('.', 'p').replace('-', 'm').replace('+', '') + 'in.s4p'

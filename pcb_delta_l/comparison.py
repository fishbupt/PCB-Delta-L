"""Read and compare PLTS-style Delta-L CSV files."""
import csv
from dataclasses import dataclass
import numpy as np

PLTS_HEADER = ['Time(ns)', 'Eigen Value(Delta-L)', 'Curve Fitted(Delta-L)', 'Uncertainty(%)(Delta-L)']
SCALES = {'Hz': 1., 'kHz': 1e3, 'MHz': 1e6, 'GHz': 1e9}

@dataclass
class DeltaLData:
    frequency_hz: np.ndarray
    values: np.ndarray


def read_delta_l(path, unit='MHz'):
    if unit not in SCALES:
        raise ValueError('请选择频率单位')
    with open(path, encoding='utf-8-sig', newline='') as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise ValueError('CSV 文件为空')
    header = [s.strip() for s in rows[0]]
    if header[:4] == PLTS_HEADER:
        indices = [0, 1, 2, 3]
        scale = SCALES[unit]
    elif 'frequency_hz' in header:
        names = ['frequency_hz', 'eigenvalue_loss_db_per_in', 'curve_fitted_db_per_in', 'relative_fit_residual_proxy_percent']
        try: indices = [header.index(n) for n in names]
        except ValueError as exc: raise ValueError('旧版结果文件缺少必要列') from exc
        scale = 1.
    else:
        raise ValueError('无法识别 CSV 列名，请选择 PLTS 格式或旧版程序结果')
    data = []
    for line, row in enumerate(rows[1:], 2):
        if not row or not any(s.strip() for s in row): continue
        try: data.append([float(row[i].strip()) if row[i].strip() else np.nan for i in indices])
        except (ValueError, IndexError) as exc: raise ValueError(f'第 {line} 行数据格式错误') from exc
    a = np.asarray(data, dtype=float)
    if a.ndim != 2 or len(a) < 2 or not np.all(np.isfinite(a[:, 0])) or np.any(np.diff(a[:, 0]) <= 0):
        raise ValueError('需要至少两个频点，横轴须有限且严格递增')
    return DeltaLData(a[:, 0] * scale, a[:, 1:])


def write_plts(path, raw, fit):
    # PLTS export calls this Time(ns), although the displayed values here are MHz.
    with open(path, 'w', newline='', encoding='utf-8-sig') as fh:
        writer = csv.writer(fh, lineterminator='\n')
        writer.writerow(PLTS_HEADER)
        for f, eigen, fitted, residual in zip(raw.frequency_hz, raw.loss_db_per_in, fit.fitted_db_per_in, fit.residual_percent):
            writer.writerow([format(x, '.15f') if np.isfinite(x) else '' for x in (f / 1e6, eigen, fitted, residual)])


def align(reference, candidate):
    """Interpolate candidate to reference frequencies, only inside valid contiguous runs."""
    f = reference.frequency_hz
    output = np.full((len(f), 3), np.nan)
    for j in range(3):
        y = candidate.values[:, j]
        valid = np.isfinite(y)
        edges = np.diff(np.r_[False, valid, False].astype(int))
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            x = candidate.frequency_hz[start:end]
            if len(x) == 1:
                match = np.isclose(f, x[0], rtol=1e-12, atol=1e-6)
                output[match, j] = y[start]
            else:
                mask = (f >= x[0]) & (f <= x[-1])
                output[mask, j] = np.interp(f[mask], x, y[start:end])
    return output

"""2L loss fitting and local-residual uncertainty benchmarked against PLTS."""
from dataclasses import dataclass

import numpy as np

# The supplied PLTS sweeps select b=0.55. Search a discrete grid so that
# other traces can select a different exponent. The vendor grid is unpublished.
_EXPONENTS = np.arange(20, 96, 5) / 100
_HALF_WINDOW_HZ = 1e9


@dataclass
class FitResult:
    fitted_db_per_in: np.ndarray
    uncertainty_percent: np.ndarray
    in_fit_range: np.ndarray
    f0_hz: float
    fmax_hz: float
    coefficients: tuple  # a, b, c, d, offset; frequency normalized to fit span
    weight_order: int
    rms_db_per_in: float
    displayed_loss_db_per_in: np.ndarray


def _local_uncertainty(frequency_hz, loss, fitted, valid):
    """Use population statistics on signed residuals in complete windows."""
    uncertainty = np.full_like(fitted, np.nan)
    usable = valid & (np.abs(fitted) > 1e-12)
    uncertainty[usable] = 0.0
    spacing = np.diff(frequency_hz)
    if np.allclose(spacing, spacing[0], rtol=1e-8, atol=1e-3):
        radius = int(np.floor(_HALF_WINDOW_HZ / np.median(spacing) + 1e-9))
        centers = np.arange(len(spacing) + 1)
        left = centers - radius
        right = centers + radius + 1
        complete = (left >= 0) & (right <= len(fitted))
    else:
        lower = frequency_hz - _HALF_WINDOW_HZ
        upper = frequency_hz + _HALF_WINDOW_HZ
        left = np.searchsorted(frequency_hz, lower, side='left')
        right = np.searchsorted(frequency_hz, upper, side='right')
        complete = (lower >= frequency_hz[0]) & (upper <= frequency_hz[-1])

    residual = loss - fitted
    for center in np.flatnonzero(usable & complete):
        window = residual[left[center]:right[center]]
        if len(window) < 2 or not valid[left[center]:right[center]].all():
            uncertainty[center] = np.nan
            continue
        uncertainty[center] = 100 * (abs(window.mean()) + 3 * window.std()) / abs(fitted[center])
    return uncertainty


def fit_loss(frequency_hz, loss_db_per_in, valid=None, cutoff_hz=None, weight_order=0):
    f = np.asarray(frequency_hz, dtype=float)
    y = np.asarray(loss_db_per_in, dtype=float)
    if f.ndim != 1 or y.shape != f.shape or not np.all(np.isfinite(f)) or np.any(np.diff(f) <= 0):
        raise ValueError('频率必须有限、严格递增，且与损耗数组长度一致')
    if weight_order not in (0, 3):
        raise ValueError('拟合权重阶数仅支持 0 或 3')
    if cutoff_hz is not None and (not np.isfinite(cutoff_hz) or cutoff_hz <= 0):
        raise ValueError('拟合截止频率必须为正')
    mask = np.isfinite(y)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != f.shape:
            raise ValueError('有效标志数组长度不匹配')
        mask &= valid
    if cutoff_hz is not None:
        mask &= f <= cutoff_hz
    points = np.flatnonzero(mask)
    if len(points) < 8:
        raise ValueError('拟合区间内需要至少 8 个有效点')

    start, stop = points[0], points[-1]
    fit_slice = slice(start, stop + 1)
    f0, fmax = float(f[start]), float(f[stop])
    x = (f - f0) / (fmax - f0)
    fit_x = x[mask]
    offset = y[start]
    target = y[mask] - offset
    root_weight = np.ones_like(fit_x) if weight_order == 0 else (1 - fit_x)**1.5
    weighted_target = target * root_weight
    best_cost = np.inf
    coefficients = None
    for b in _EXPONENTS:
        basis = np.column_stack((fit_x**b, fit_x**2, fit_x))
        weighted_basis = basis * root_weight[:, None]
        params, _, rank, _ = np.linalg.lstsq(weighted_basis, weighted_target, rcond=None)
        error = weighted_basis @ params - weighted_target
        cost = float(error @ error)
        if rank == 3 and cost < best_cost:
            a, c, d = params
            coefficients = (float(a), float(b), float(c), float(d), float(offset))
            best_cost = cost
    if coefficients is None:
        raise ValueError('拟合矩阵秩不足，请检查频率网格和拟合频段')

    a, b, c, d, offset = coefficients
    fit_x = x[fit_slice]
    fitted = np.full_like(y, np.nan)
    fitted[fit_slice] = offset + a * fit_x**b + c * fit_x**2 + d * fit_x
    rms = float(np.sqrt(np.mean((y[mask] - fitted[mask])**2)))
    uncertainty = np.full_like(y, np.nan)
    uncertainty[fit_slice] = _local_uncertainty(f[fit_slice], y[fit_slice], fitted[fit_slice], mask[fit_slice])

    # Both PLTS exports average the first three samples at the second point.
    # Apply only to the displayed traces, after fitting and residual statistics.
    displayed_loss = np.where(np.isfinite(y), y, np.nan)
    if valid is not None:
        displayed_loss[~valid] = np.nan
    first_three = slice(start, start + 3)
    if start == 0 and mask[first_three].all() and np.isclose(f[1] - f[0], f[2] - f[1], rtol=1e-8, atol=1e-3):
        displayed_loss[1] = y[first_three].mean()
        fitted[1] = fitted[first_three].mean()

    return FitResult(fitted, uncertainty, mask, f0, fmax, coefficients, weight_order, rms, displayed_loss)

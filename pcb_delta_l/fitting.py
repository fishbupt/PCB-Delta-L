"""Experimental loss fitting and residual metric for comparison with PLTS.

The percent curve is a pointwise relative fit residual, not a metrological
uncertainty and not a verified implementation of Intel/Keysight uncertainty.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares

@dataclass
class FitResult:
    fitted_db_per_in: np.ndarray
    residual_percent: np.ndarray
    in_fit_range: np.ndarray
    f0_hz: float
    fmax_hz: float
    coefficients: tuple  # a, b, c, d, offset; frequency normalized to fit span
    weight_order: int
    rms_db_per_in: float

def fit_loss(frequency_hz, loss_db_per_in, valid=None, cutoff_hz=None, weight_order=0):
    f=np.asarray(frequency_hz,dtype=float); y=np.asarray(loss_db_per_in,dtype=float)
    if f.ndim != 1 or y.shape != f.shape or not np.all(np.isfinite(f)) or np.any(np.diff(f)<=0):
        raise ValueError('频率必须有限、严格递增，且与损耗数组长度一致')
    if weight_order not in (0, 3): raise ValueError('拟合权重阶数仅支持 0 或 3')
    if cutoff_hz is not None and (not np.isfinite(cutoff_hz) or cutoff_hz <= 0): raise ValueError('拟合截止频率必须为正')
    mask=np.isfinite(y)
    if valid is not None:
        v=np.asarray(valid,dtype=bool)
        if v.shape != f.shape: raise ValueError('有效标志数组长度不匹配')
        mask &= v
    if cutoff_hz is not None: mask &= f <= cutoff_hz
    if np.count_nonzero(mask)<8: raise ValueError('拟合区间内需要至少 8 个有效点')
    f0=float(f[mask][0]); fmax=float(f[mask][-1]); span=fmax-f0
    if span<=0: raise ValueError('拟合频段宽度必须大于零')
    x=(f[mask]-f0)/span; ym=y[mask]
    # W multiplies squared residual; scipy residual uses sqrt(W).
    weight=np.ones_like(x) if weight_order==0 else np.maximum(0,1-x)**3
    root=np.sqrt(weight)
    if np.count_nonzero(weight)>5:
        def model(t,p):
            a,b,c,d,offset=p
            return a*np.power(t,b)+c*t*t+d*t+offset
        def residual(p): return (model(x,p)-ym)*root
        best=None
        for initial_b in (0.4,0.6,0.8):
            p0=[float(ym[-1]-ym[0]), initial_b, 0., 0., float(ym[0])]
            res=least_squares(residual,p0,bounds=([-np.inf,.2,-np.inf,-np.inf,-np.inf],[np.inf,.95,np.inf,np.inf,np.inf]),max_nfev=3000)
            if res.success and np.all(np.isfinite(res.x)) and (best is None or res.cost<best.cost): best=res
        if best is None: raise ValueError('拟合未收敛，请检查测量数据和拟合频段')
    else: raise ValueError('有效加权数据点不足')
    in_range=(f>=f0)&(f<=fmax); fitted=np.full_like(y,np.nan); t=(f[in_range]-f0)/span; fitted[in_range]=model(t,best.x)
    residual_pct=np.full_like(y,np.nan)
    # Do not hide divergent residuals by substituting an arbitrary denominator.
    ok=mask & (np.abs(fitted)>1e-12)
    residual_pct[ok]=100*np.abs(y[ok]-fitted[ok])/np.abs(fitted[ok])
    rms=float(np.sqrt(np.mean((y[mask]-fitted[mask])**2)))
    return FitResult(fitted,residual_pct,mask,f0,fmax,tuple(float(v) for v in best.x),weight_order,rms)

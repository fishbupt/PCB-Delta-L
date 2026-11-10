"""Two-line propagation loss extraction; not a certified Intel implementation."""
from dataclasses import dataclass
import numpy as np
import skrf as rf

# Physical order: input+, output+, input-, output-.
MIX = np.array([[1,0,-1,0],[0,1,0,-1],[1,0,1,0],[0,1,0,1]], float) / np.sqrt(2)

@dataclass
class Result:
    frequency_hz: np.ndarray
    loss_db_per_in: np.ndarray
    eigenvalues: np.ndarray
    condition: np.ndarray
    valid: np.ndarray


def differential_network(net, ports=(1, 2, 3, 4)):
    """ports specify input+, output+, input-, output- using one-based indices."""
    if net.nports == 2:
        return net
    if net.nports != 4 or sorted(ports) != [1,2,3,4]:
        raise ValueError('需要二端口文件或四个不重复的端口号 1–4')
    z = np.asarray(net.z0)
    if not np.allclose(z, z[:, :1]) or not np.allclose(z.imag, 0):
        raise ValueError('混模转换要求各端口参考阻抗相同且为实数')
    order = np.array(ports) - 1
    s = net.s[:, order][:, :, order]
    sm = MIX @ s @ MIX.T
    return rf.Network(frequency=net.frequency, s=sm[:, :2, :2], z0=2*z[:,0])


def extract_2l(short, long, short_in, long_in, ports=(1,2,3,4)):
    if not np.isfinite(short_in) or not np.isfinite(long_in) or short_in < 0 or long_in <= short_in:
        raise ValueError('长度必须有限、短线非负且长线大于短线')
    if len(short.f) != len(long.f) or not np.allclose(short.f, long.f, rtol=0, atol=1e-3):
        raise ValueError('长短线频率网格不一致；请先使用相同频点测量')
    a, b = differential_network(short, ports), differential_network(long, ports)
    if not np.allclose(a.z0,b.z0):
        raise ValueError('长短线参考阻抗不一致')
    n=len(a.f); eig=np.full((n,2),np.nan+1j*np.nan); loss=np.full(n,np.nan); cond=np.full(n,np.nan); valid=np.zeros(n,bool)
    # ABCD conversion is singular for vanishing S21; leave such points invalid.
    for i in range(n):
        try:
            if abs(a.s[i,1,0]) < 1e-12 or abs(b.s[i,1,0]) < 1e-12: continue
            ta=rf.network.s2a(a.s[i:i+1], z0=a.z0[i:i+1])[0]
            tb=rf.network.s2a(b.s[i:i+1], z0=b.z0[i:i+1])[0]
            cond[i]=np.linalg.cond(ta)
            if not np.isfinite(cond[i]) or cond[i]>1e12: continue
            mat=np.linalg.solve(ta.T, tb.T).T
            e=np.linalg.eigvals(mat); eig[i]=e
            if not np.all(np.isfinite(e)) or min(abs(e)) < 1e-300: continue
            # Loss from reciprocal eigenvalue pair: avoid clipping noisy active points.
            mags=np.abs(e); idx=np.argmin(mags); loss[i]=-20*np.log10(mags[idx])/(long_in-short_in)
            valid[i]=True
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            continue
    return Result(a.f.copy(),loss,eig,cond,valid)

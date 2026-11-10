"""Synthetic coupled-mode fixtures and transmission lines, with known loss."""
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


def generate(f, lengths=(2.,5.,10.), z_line=95., er=3.5, dc=0.01, skin=0.08, dielectric=0.035, fixture_z=85., fixture_length=0.15):
    f=np.asarray(f,float)
    if len(f)<2 or not np.all(np.isfinite(f)) or np.any(f<=0) or np.any(np.diff(f)<=0): raise ValueError('频率须为正且严格递增，至少两个点')
    if not all(np.isfinite(x) and x>=0 for x in (dc,skin,dielectric,fixture_length)) or min(z_line,er,fixture_z)<=0 or any(x<=0 for x in lengths): raise ValueError('长度、阻抗和介电常数须为正，损耗系数须非负')
    freq=rf.Frequency.from_f(f,unit='hz'); out={}
    fixture,_=line_abcd(f,fixture_length,fixture_z,dc,skin,dielectric,er)
    for length in lengths:
        line,truth=line_abcd(f,length,z_line,dc,skin,dielectric,er)
        diff=fixture @ line @ fixture
        sd=rf.network.a2s(diff,z0=100)
        # Independent common-mode line. Modal conversion is deliberately zero.
        common,_=line_abcd(f,length+2*fixture_length,25,dc,skin,dielectric,er)
        sc=rf.network.a2s(common,z0=25)
        sm=np.zeros((len(f),4,4),complex); sm[:,:2,:2]=sd; sm[:,2:,2:]=sc
        s=MIX.T @ sm @ MIX
        out[length]=rf.Network(frequency=freq,s=s,z0=50,name=f'line_{length:g}in')
    return out,truth

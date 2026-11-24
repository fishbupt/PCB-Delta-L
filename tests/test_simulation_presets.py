import numpy as np
import pytest
from pcb_delta_l.simulation import generate
from pcb_delta_l.core import differential_network
from pcb_delta_l.touchstone import write_s4p
import skrf as rf

def test_default_compatibility_and_s4p_roundtrip(tmp_path):
    f=np.linspace(10e6,40e9,101)
    before,truth=generate(f,(5.,10.))
    after,truth2=generate(f,(5.,10.),length_impedances={})
    np.testing.assert_allclose(before[10.].s,after[10.].s)
    np.testing.assert_allclose(truth,truth2)
    path=tmp_path/'test.s4p'
    write_s4p(path,after[10.])
    parsed=rf.Network(str(path))
    np.testing.assert_allclose(parsed.f,f,rtol=1e-12)
    np.testing.assert_allclose(parsed.s,after[10.].s,atol=1e-12)

def test_reflection_and_independent_impedance():
    f=np.linspace(10e6,40e9,201)
    good,_=generate(f,(5.,10.),z_line=100,fixture_z=100)
    bad,_=generate(f,(5.,10.),z_line=100,fixture_z=60)
    rg=np.abs(differential_network(good[5.]).s[:,0,0])
    rb=np.abs(differential_network(bad[5.]).s[:,0,0])
    assert np.max(rg)<1e-10
    assert np.max(rb)>.1
    changed,_=generate(f,(5.,10.),z_line=100,fixture_z=100,length_impedances={10.:102.})
    np.testing.assert_allclose(changed[5.].s,good[5.].s)
    assert not np.allclose(changed[10.].s,good[10.].s)

@pytest.mark.parametrize('z',[0,-1,float('nan')])
def test_invalid_independent_impedance(z):
    with pytest.raises(ValueError):
        generate(np.array([1e6,2e6]),(5.,10.),length_impedances={10.:z})

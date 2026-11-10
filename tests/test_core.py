import numpy as np
import pytest
import skrf as rf
from pcb_delta_l.simulation import generate
from pcb_delta_l.core import extract_2l, differential_network

def test_extract_known_loss_with_mismatched_fixture_and_line():
    f=np.linspace(1e8,40e9,401)
    nets,truth=generate(f,fixture_z=75,z_line=110)
    result=extract_2l(nets[2.],nets[5.],2,5)
    assert result.valid.all()
    np.testing.assert_allclose(result.loss_db_per_in,truth,rtol=1e-9,atol=1e-9)

def test_s4p_roundtrip(tmp_path):
    f=np.linspace(1e8,10e9,101); nets,truth=generate(f,lengths=(2,5))
    for n,net in nets.items(): net.write_touchstone(str(tmp_path / str(n)))
    result=extract_2l(rf.Network(str(tmp_path/'2.s4p')),rf.Network(str(tmp_path/'5.s4p')),2,5)
    np.testing.assert_allclose(result.loss_db_per_in,truth,atol=1e-8)

def test_reject_incompatible_input():
    n,_=generate(np.linspace(1e8,1e9,10),lengths=(2,5))
    with pytest.raises(ValueError): extract_2l(n[2],n[5],5,2)
    with pytest.raises(ValueError): extract_2l(n[2],n[5],2,5,(1,1,2,3))
    m,_=generate(np.linspace(1e8,2e9,10),lengths=(2,5))
    with pytest.raises(ValueError): extract_2l(n[2],m[5],2,5)

def test_port_permutation():
    nets,truth=generate(np.linspace(1e8,5e9,31),lengths=(2,5))
    # Reorder physical ports to 1+, 1-, 2+, 2-.
    perm=[0,2,1,3]
    for net in nets.values(): net.renumber([0,1,2,3],perm)
    result=extract_2l(nets[2],nets[5],2,5,(1,3,2,4))
    np.testing.assert_allclose(result.loss_db_per_in,truth,atol=1e-9)

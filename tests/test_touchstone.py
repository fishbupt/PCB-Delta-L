import numpy as np
import pytest
import skrf as rf
from pcb_delta_l.simulation import generate
from pcb_delta_l.touchstone import write_s4p, length_filename

def test_plts_compatible_format_and_port_order(tmp_path):
    freq=np.array([1e7, 2e7, 3e7]); s=np.arange(48).reshape(3,4,4)*.001 + 1j*np.arange(48,96).reshape(3,4,4)*.001
    network=rf.Network(frequency=rf.Frequency.from_f(freq,unit='hz'),s=s,z0=50)
    path=write_s4p(tmp_path/'line.s4p',network)
    content=path.read_bytes(); assert content.startswith(b'# Hz S RI R 50\r\n')
    rows=content.decode('ascii').splitlines(); assert len(rows)==13
    assert [len(row.split()) for row in rows[1:5]] == [9,8,8,8]
    read=rf.Network(str(path)); np.testing.assert_allclose(read.s,s); np.testing.assert_allclose(read.f,freq)

def test_simulation_file_roundtrip(tmp_path):
    nets,_=generate(np.linspace(1e7,40e9,201), lengths=(2,5,5.5))
    for length, net in nets.items():
        path=write_s4p(tmp_path/length_filename(length),net)
        np.testing.assert_allclose(rf.Network(str(path)).s,net.s,atol=1e-14)
    assert length_filename(5.5)=='line_5p5in.s4p'

def test_reject_too_many_points(tmp_path):
    net=rf.Network(frequency=rf.Frequency.from_f(np.arange(16001)+1,unit='hz'),s=np.zeros((16001,4,4)),z0=50)
    with pytest.raises(ValueError, match='16000'): write_s4p(tmp_path/'large.s4p',net)

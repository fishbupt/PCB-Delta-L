import csv
import numpy as np
import skrf as rf
from pcb_delta_l.core import MIX
from pcb_delta_l.mixed_mode import curves, write_curve
from pcb_delta_l.simulation import generate


def test_default_simulation_matches_plts_sdd11_and_csv(tmp_path):
    frequency=np.array([10e6, 34.99375e6])
    networks,_=generate(frequency)
    values=curves(networks[2.])['SDD11']
    np.testing.assert_allclose(values,[-56.27352322261838,-45.582251173803556],atol=1e-8)
    path=tmp_path/'SDD11.csv'; write_curve(path,frequency,'SDD11',values)
    with path.open(encoding='utf-8-sig',newline='') as fh: rows=list(csv.reader(fh))
    assert rows[0]==['Frequency(MHz)','SDD11(Log Mag)']
    np.testing.assert_allclose(np.asarray(rows[1:],float),np.column_stack((frequency/1e6,values)))


def test_mixed_mode_preserves_direction_and_port_mapping():
    sm=np.zeros((1,4,4),complex); sm[0,0,1]=.1; sm[0,1,0]=.01; sm[0,2,3]=.001
    s=MIX.T @ sm @ MIX
    n=rf.Network(frequency=rf.Frequency.from_f([1e9],unit='hz'),s=s,z0=50)
    result=curves(n)
    np.testing.assert_allclose([result['SDD12'][0],result['SDD21'][0],result['SCC12'][0]],[-20,-40,-60],atol=1e-10)

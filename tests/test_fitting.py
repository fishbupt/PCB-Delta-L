import numpy as np
import pytest
from pcb_delta_l.fitting import fit_loss

def test_recovers_known_fitting_model():
    f=np.linspace(1e7,40e9,801); x=(f-f[0])/(f[-1]-f[0]); y=.4*x**.55+.15*x*x+.8*x+.03
    result=fit_loss(f,y)
    np.testing.assert_allclose(result.fitted_db_per_in,y,atol=1e-7)
    assert result.rms_db_per_in<1e-7
    assert np.nanmax(result.residual_percent)<1e-3

def test_residual_mask_and_cutoff():
    f=np.linspace(1e8,10e9,101); y=.5+np.sqrt(f/1e9); y[20]+=0.2; y[4]=np.nan
    result=fit_loss(f,y,cutoff_hz=8e9,weight_order=3)
    assert np.isnan(result.residual_percent[4])
    assert np.isnan(result.fitted_db_per_in[f>8e9]).all()
    mask=result.in_fit_range
    np.testing.assert_allclose(result.residual_percent[mask],100*abs(y[mask]-result.fitted_db_per_in[mask])/abs(result.fitted_db_per_in[mask]))
    assert result.residual_percent[20]>1

def test_insufficient_points_and_zero_denominator():
    with pytest.raises(ValueError): fit_loss(np.arange(5.),np.ones(5))
    r=fit_loss(np.arange(10.),np.zeros(10))
    assert np.isnan(r.residual_percent).all()

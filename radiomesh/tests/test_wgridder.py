import sys
import numpy as np
from ducc0.wgridder import vis2dirty
from time import time
from pyrap.tables import table
from scipy.constants import c as lightspeed

def test_grid_fourcorr(uvw, freq, vis, wgt, npix, cell_rad,
                       epsilon=1e-4, precision='double', nthreads=1):
    
    if precision == 'double':
        rtype = 'f8'
        ctype = 'c16'
    else:
        rtype = 'f4'
        ctype = 'c8'

    vis = vis.astype(ctype)
    wgt = wgt.astype(rtype)

    nrow, nchan, ncorr = vis.shape

    dirty = np.zeros((ncorr, npix, npix), dtype=rtype)

    ti = time()
    for c in range(ncorr):
        vis2dirty(
            uvw=uvw,
            freq=freq,
            vis=vis[:, :, c],
            wgt=wgt[:, :, c],
            npix_x=npix,
            npix_y=npix,
            pixsize_x=cell_rad,
            pixsize_y=cell_rad,
            epsilon=epsilon,
            do_wgridding=False,
            nthreads=nthreads,
            verbosity=1,
            dirty=dirty[c]
        )

    print(f'Time taken to map ({nrow},{nchan},{ncorr}) to ({ncorr},{npix},{npix}) = {time()-ti}')


if __name__=='__main__':
    ms_name = sys.argv[1]
    ms = table(ms_name)
    uvw = ms.getcol('UVW')
    vis = ms.getcol('DATA')
    try:
        wgt = ms.getcol('WEIGHT_SPECTRUM')
    except:
        wgt = np.ones(vis.shape, dtype='f4')
    ms.close()
    freq = table(f'{ms_name}::SPECTRAL_WINDOW').getcol('CHAN_FREQ')[0]
    uv_max = np.maximum(np.abs(uvw[:, 0]).max(), np.abs(uvw[:, 1]).max())
    max_freq = freq.max()
    cell_N = 1.0 / (2 * uv_max * max_freq / lightspeed)  # max cell size
    cell_rad = cell_N/2.0  # oversample by a factor of two
    fov = 1.0  # field of view degrees
    # import ipdb; ipdb.set_trace()
    npix = int(fov/np.rad2deg(cell_rad))
    if npix % 2:
        npix += 1
    test_grid_fourcorr(uvw, freq, vis, wgt, npix, cell_rad)



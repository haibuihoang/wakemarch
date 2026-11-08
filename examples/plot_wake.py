"""Generate a data-free wake figure and a reproducibility record from the installed API."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import scipy
import wakemarch
from wakemarch import Numerics, WakeModel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model = WakeModel(D=126, H=90, U=8, CT=0.8, yaw_deg=25, TI=0.06,
                      numerics=Numerics(h_D=1/20))
    start = time.perf_counter()
    result = model.solve(10, progress=False)
    x, y, z = np.linspace(-2*model.D, 10*model.D, 181), np.linspace(-2*model.D, 2*model.D, 101), np.linspace(0, model.H+1.5*model.D, 81)
    xx, yy = np.meshgrid(x, y)
    yp, zp = np.meshgrid(y, z)
    hub = result.sample_deficit(xx, yy, model.H)/model.U
    plane = result.sample_deficit(5*model.D, yp, zp)/model.U
    assert np.isfinite(hub).all() and np.isfinite(plane).all()
    elapsed = time.perf_counter()-start
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    # diverging scale: deficit red, speed-up beside the near wake and induction zone upstream visible
    vmax = max(float(hub.max()), float(plane.max()))
    rdbu = plt.get_cmap('RdBu_r')     # white at zero on the range [-0.2 vmax, vmax]
    cmap = ListedColormap(rdbu(np.r_[np.linspace(0.3, 0.5, 40, endpoint=False), np.linspace(0.5, 1.0, 200)]))
    norm = dict(vmin=-0.2*vmax, vmax=vmax, cmap=cmap)
    axes[0].pcolormesh(x/model.D, y/model.D, hub, **norm, shading='auto')
    axes[0].set(xlabel='x / D', ylabel='y / D', title='Hub-height deficit (yaw 25°)')
    mesh = axes[1].pcolormesh(y/model.D, z/model.D, plane, **norm, shading='auto')
    axes[1].set(xlabel='y / D', ylabel='Height / D', title='Deficit at x = 5D')
    fig.colorbar(mesh, ax=axes, label='Deficit / U (negative: speed-up)')
    fig.savefig(args.output/'wake.png', dpi=150)
    plt.close(fig)
    record = dict(version=wakemarch.__version__, python=sys.version, platform=platform.platform(),
                  numpy=np.__version__, scipy=scipy.__version__, parameters=asdict(model),
                  max_dist_D=10, compute_and_sample_s=elapsed,
                  stored_x_D=(result.x/model.D).tolist(),
                  sampling=dict(x_m=x.tolist(), y_m=y.tolist(), z_m=z.tolist(), plane_x_D=5))
    (args.output/'run.json').write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main()

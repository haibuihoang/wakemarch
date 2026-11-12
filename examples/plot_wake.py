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
import numpy as np
import scipy
import wakemarch
from wakemarch import WakeModel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model = WakeModel(D=126, H=90, U=8, CT=0.8, yaw_deg=25, TI=0.06, shear_alpha=0.14)
    start = time.perf_counter()
    result = model.solve(10, progress=False)
    x, y, z = np.linspace(-2*model.D, 10*model.D, 181), np.linspace(-2*model.D, 2*model.D, 101), np.linspace(0, 2.2*model.D, 81)
    xx, yy = np.meshgrid(x, y)
    yp, zp = np.meshgrid(y, z)
    hub = result.sample_deficit(xx, yy, model.H)/model.U
    plane = result.sample_deficit(5*model.D, yp, zp)/model.U
    assert np.isfinite(hub).all() and np.isfinite(plane).all()
    elapsed = time.perf_counter()-start
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.1), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[3, 1.82]))
    # diverging scale: deficit (slower flow) blue, speed-up beside the near wake and in the induction zone red
    norm = dict(cmap='RdBu', vmin=-0.45, vmax=0.45)
    axes[0].pcolormesh(x/model.D, y/model.D, hub, **norm, shading='auto')
    axes[0].set(xlabel='x / D', ylabel='y / D', title='Hub height, yaw 25°', aspect='equal')
    mesh = axes[1].pcolormesh(y/model.D, z/model.D, plane, **norm, shading='auto')
    axes[1].set(xlabel='y / D', ylabel='Height / D', title='x = 5D, looking downstream',
                aspect='equal', xlim=(2, -2))
    fig.colorbar(mesh, ax=axes, label='Velocity deficit Δu / U', shrink=0.9)
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

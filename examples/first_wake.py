"""A data-free example using the installed public API. Prints a JSON summary."""
import json

import numpy as np

from wakemarch import Numerics, WakeModel, __version__


def main():
    model = WakeModel(D=126, H=90, U=8, CT=0.8, yaw_deg=25, TI=0.06,
                      numerics=Numerics(h_D=1 / 20))
    result = model.solve(max_dist_D=6)
    y = np.linspace(-model.D, model.D, 101)
    deficit = result.sample_deficit(5 * model.D, y, model.H)
    velocity = result.sample_streamwise_velocity(5 * model.D, 0, model.H)
    assert np.isfinite(deficit).all() and deficit.max() > 0
    assert np.argmax(deficit) < len(y) // 2, "Positive yaw should deflect towards -y"
    print(json.dumps(dict(version=__version__, peak_deficit_m_s=float(deficit.max()),
                          hub_axis_velocity_m_s=velocity,
                          peak_y_D=float(y[np.argmax(deficit)] / model.D),
                          stored_planes=len(result.x)), indent=2))


if __name__ == "__main__":
    main()

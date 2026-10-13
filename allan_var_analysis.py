import allantools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

resid_data = pd.read_csv("data/residuals_clean_v2.csv")

#print(resid_data.head(5))

#Checking for constant sample rate
dt = np.diff(resid_data['t'].values)
print("Sample rate stats:", 1/dt.mean(), dt.std())

fs = 198.40401607529185

results = {}
for axis in ["res_wx", "res_wy", "res_wz", "res_ax", "res_ay", "res_az"]:
    data = resid_data[axis].values
    taus, adev, adev_err, n = allantools.oadev(data, rate = fs, data_type = "freq", taus = "octave")
    results[axis] = (taus, adev)

figs, axes = plt.subplots(1, 2, figsize = (12,5))
figs, axes = plt.subplots(1, 2, figsize = (12,5))
for axis in ["res_wx", "res_wy", "res_wz"]:
    taus, adev = results[axis]
    axes[0].loglog(taus, adev, label=axis)
axes[0].set_title("Gyro Allan Deviation")
axes[0].set_xlabel("tau(s)"); axes[0].set_ylabel("Allan deviation")
axes[0].legend(); axes[0].grid(True, which = 'both')

for axis in ["res_ax", "res_ay", "res_az"]:
    taus, adev = results[axis]
    axes[1].loglog(taus, adev, label=axis)
axes[1].set_title("Accel Allan Deviation")
axes[1].set_xlabel("tau(s)"); axes[1].set_ylabel("Allan deviation")
axes[1].legend(); axes[1].grid(True, which = 'both')
plt.tight_layout()
plt.show()

for axis in ["res_wx", "res_wy", "res_wz", "res_ax", "res_ay", "res_az"]:
    taus, adev = results[axis]
    idx = np.argmin(np.abs(taus - 1.0))
    print(f"{axis}: noise density at tau=1s ≈ {adev[idx]:.6f}")
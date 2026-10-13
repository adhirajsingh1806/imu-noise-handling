import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy.signal import medfilt, correlate, correlation_lags
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy import stats

imu = pd.read_csv("data/data_1.csv")
gt = pd.read_csv("data/data_2.csv")

"""
print("IMU columns:", imu.columns.tolist())
print(imu.head())

print("Ground truth columns:", gt.columns.tolist())
print(gt.head())

"""
imu.columns = ["t", "wx", "wy", "wz", "ax", "ay", "az"]
gt.columns = ["t", "px", "py", "pz", "qw", "qx", "qy", "qz"]

#Converting nanoseconds into seconds for numerical stability
imu["t"] = imu["t"] * 1e-9
gt["t"] = gt["t"] * 1e-9

#Trimming IMU to the time range covered by ground truth
imu = imu[(imu.t >= gt.t.min()) & (imu.t <= gt.t.max())].reset_index()

#Smoothing out ground truth using Cubic Spline interpolation
pos_spline = [CubicSpline(gt.t, gt[c], bc_type = "natural") for c in ["px", "py", "pz"]]

#Calculating angular velocity and linear acceleration from orientation using Spherical Linear interpolation and finite difference on rotation vectors
rots = Rotation.from_quat(gt[["qx", "qy", "qz", "qw"]].values)
slerp = Slerp(gt.t, rots)

def gt_angular_velocity(t, dt = 1e-3):
    t = np.clip(t, gt.t.min() + dt, gt.t.max() - dt)
    r1 = slerp(t - dt)
    r2 = slerp(t + dt)
    re1 = r2 * r1.inv()
    return re1.as_rotvec()/(2*dt)

def gt_linear_accel(t):
    a_world = np.array([sp(t,2) for sp in pos_spline]) #2nd derivative from spline
    g = np.array([0, 0, 9.81])
    r = slerp(np.clip(t, gt.t.min(), gt.t.max()))
    return r.inv().apply(a_world + g)

#Computing ideal IMU signal at each timestamp
ideal_w = np.array([gt_angular_velocity(t) for t in imu.t])
ideal_a = np.array([gt_linear_accel(t) for t in imu.t])

#Residuals
residual_w = imu[["wx", "wy", "wz"]].values - ideal_w
residual_a = imu[["ax", "ay", "az"]].values - ideal_a

resid_data = pd.DataFrame(
    np.hstack([imu[["t"]].values, residual_w, residual_a]),
    columns = ["t", "res_wx", "res_wy", "res_wz", "res_ax", "res_ay", "res_az"]
)

resid_data.to_csv("data/residuals.csv", index = False)
print(resid_data.describe())
print(resid_data[["res_ax","res_ay","res_az","res_wx","res_wy","res_wz"]].kurtosis())

plt.plot(resid_data["t"], resid_data["res_az"])
plt.show()

dt_gt = np.diff(gt["t"].values)
print(dt_gt.min(), dt_gt.max(), dt_gt.mean())
print((dt_gt <= 0).sum())   #non-monotonic or duplicate timestamps

worst_idx = resid_data["res_az"].abs().idxmax()
worst_t = resid_data.loc[worst_idx, "t"]
print("Worst spike at t =", worst_t)

#looking at ground truth samples around the worst spike
window = gt[(gt.t > worst_t - 0.1) & (gt.t < worst_t + 0.1)]
print(window[["t", "px", "py", "pz"]])

gt_clean = gt.copy()
for c in ["px", "py", "pz"]:
    gt_clean[c] = medfilt(gt[c].values, kernel_size=5)

for c in ["qw", "qx", "qy", "qz"]:
    gt_clean[c] = medfilt(gt[c].values, kernel_size=5)

#Rebuilding pos_spline from gt_clean

quats_raw = gt[["qw", "qx", "qy", "qz"]].values.copy()

for i in range(1, len(quats_raw)):
    if np.dot(quats_raw[i], quats_raw[i - 1]) < 0:
        quats_raw[i] *= -1

gt_unwrapped = gt.copy()
gt_unwrapped[["qw", "qx", "qy", "qz"]] = quats_raw

gt_clean = gt_unwrapped.copy()
for c in ["px", "py", "pz", "qw", "qx", "qy", "qz"]:
    gt_clean[c] = medfilt(gt_unwrapped[c].values, kernel_size=5)

for c in ["px", "py", "pz"]:
    gt_clean[c] = medfilt(gt_unwrapped[c].values, kernel_size=9)

quats = gt_clean[["qx", "qy", "qz", "qw"]].values
quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
gt_clean[["qx", "qy", "qz", "qw"]] = quats

dt = np.diff(gt_unwrapped.t.values)
dpos = np.diff(gt_unwrapped[["px","py","pz"]].values, axis=0)
speed_est = np.linalg.norm(dpos, axis=1) / dt
bad_idx = np.where(speed_est > np.percentile(speed_est, 99.5))[0]
print(len(bad_idx), "suspected bad frames")

print(bad_idx)
print(np.diff(bad_idx))

gt_interp = gt_clean.copy()

pos_spline = [UnivariateSpline(gt_clean.t, gt_clean[c], k=4, s=len(gt_clean) * 1e-6) for c in ["px", "py", "pz"]]
rots = Rotation.from_quat(gt_clean[["qx", "qy", "qz", "qw"]].values)
slerp = Slerp(gt_clean.t, rots)

def gt_angular_velocity(t, dt=1e-3):
    t = np.clip(t, gt_interp.t.min() + dt, gt_interp.t.max() - dt)
    r1 = slerp(t - dt)
    r2 = slerp(t + dt)
    rel = r2 * r1.inv()
    return rel.as_rotvec() / (2 * dt)

def gt_linear_accel(t):
    a_world = np.array([sp(t, 2) for sp in pos_spline])
    g = np.array([0, 0, 9.81])
    r = slerp(np.clip(t, gt_interp.t.min(), gt_interp.t.max()))
    return r.inv().apply(a_world + g)

imu_trimmed = imu[(imu.t >= gt_interp.t.min()) & (imu.t <= gt_interp.t.max())].reset_index(drop=True)

#Computing ideal IMU signal at each timestamp
ideal_w = np.array([gt_angular_velocity(t) for t in imu_trimmed.t])
ideal_a = np.array([gt_linear_accel(t) for t in imu_trimmed.t])

#Residuals
residual_w = imu_trimmed[["wx", "wy", "wz"]].values - ideal_w
residual_a = imu_trimmed[["ax", "ay", "az"]].values - ideal_a

resid_data_v2 = pd.DataFrame(
    np.hstack([imu_trimmed[["t"]].values, residual_w, residual_a]),
    columns = ["t", "res_wx", "res_wy", "res_wz", "res_ax", "res_ay", "res_az"]
)

loose_bad_indices = [13745,13747,13749,13750,13751,13752,13753,13754,13755,13756,
                      13757,13759,13762,13765,13831,13874,13877,13892,
                      14024,14027,14030,14032,14033,14036,14057,14063,14069,
                      14073,14123,14129,14130,14132,14137]

exclude_mask_orient = np.ones(len(gt_clean), dtype=bool)

for start_i, end_i in [(11587, 11627), (12685, 12708), (13068, 13086), (13598, 13616)]:
    exclude_mask_orient[start_i:end_i+1] = False

for idx in loose_bad_indices:
    exclude_mask_orient[idx] = False

gt_for_spline = gt_clean[exclude_mask_orient].reset_index(drop=True)

pos_spline = [
    UnivariateSpline(gt_for_spline.t, gt_for_spline[c], k=4, s=len(gt_for_spline) * 1e-6)
    for c in ["px", "py", "pz"]
]
rots = Rotation.from_quat(gt_for_spline[["qx", "qy", "qz", "qw"]].values)
slerp = Slerp(gt_for_spline.t, rots)

def gt_angular_velocity(t, dt=1e-3):
    t = np.clip(t, gt_for_spline.t.min() + dt, gt_for_spline.t.max() - dt)
    r1 = slerp(t - dt)
    r2 = slerp(t + dt)
    rel = r2 * r1.inv()
    return rel.as_rotvec() / (2 * dt)

def gt_linear_accel(t):
    a_world = np.array([sp(t, 2) for sp in pos_spline])
    g = np.array([0, 0, 9.81])
    r = slerp(np.clip(t, gt_for_spline.t.min(), gt_for_spline.t.max()))
    return r.inv().apply(a_world + g)

imu_trimmed = imu[(imu.t >= gt_clean.t.min()) & (imu.t <= gt_clean.t.max())].reset_index(drop=True)

#Computing ideal IMU signal at each timestamp
ideal_w = np.array([gt_angular_velocity(t) for t in imu_trimmed.t])
ideal_a = np.array([gt_linear_accel(t) for t in imu_trimmed.t])

#Residuals
residual_w = imu_trimmed[["wx", "wy", "wz"]].values - ideal_w
residual_a = imu_trimmed[["ax", "ay", "az"]].values - ideal_a

resid_data_v2 = pd.DataFrame(
    np.hstack([imu_trimmed[["t"]].values, residual_w, residual_a]),
    columns = ["t", "res_wx", "res_wy", "res_wz", "res_ax", "res_ay", "res_az"]
)

exclude_windows = [
    # original position dropout clusters
    (gt_clean.t.iloc[11587], gt_clean.t.iloc[11627]),
    (gt_clean.t.iloc[12685], gt_clean.t.iloc[12708]),
    (gt_clean.t.iloc[13068], gt_clean.t.iloc[13086]),

    # newly found orientation dropout clusters
    (gt_clean.t.iloc[13598], gt_clean.t.iloc[13616]),
    (gt_clean.t.iloc[13745], gt_clean.t.iloc[13892]),
    (gt_clean.t.iloc[14024], gt_clean.t.iloc[14137]),
]

mask = np.ones(len(resid_data_v2), dtype=bool)
for t_start, t_end in exclude_windows:
    mask &= ~((resid_data_v2["t"] >= t_start) & (resid_data_v2["t"] <= t_end))

resid_data_v2_clean = resid_data_v2[mask].reset_index(drop=True)

print(f"Dropped {(~mask).sum()} rows out of {len(resid_data_v2)} total "
      f"({(~mask).sum() / len(resid_data_v2) * 100:.2f}%)")

resid_data_v2_clean.to_csv("data/residuals_clean_v2.csv", index = False)
print(resid_data_v2_clean.describe())
print(resid_data_v2_clean[["res_ax","res_ay","res_az","res_wx","res_wy","res_wz"]].kurtosis())

plt.plot(resid_data_v2_clean["t"], resid_data_v2_clean["res_az"])
plt.show()

quats = gt_clean[["qx", "qy", "qz", "qw"]].values
rots_check = Rotation.from_quat(quats)

dt = np.diff(gt_clean.t.values)

rel_rots = rots_check[1:] * rots_check[:-1].inv()
angular_dist = rel_rots.magnitude() 

angular_speed = angular_dist / dt  # rad/s

threshold = np.percentile(angular_speed, 99.5)
bad_idx = np.where(angular_speed > threshold)[0]

print(f"{len(bad_idx)} suspected bad frames (orientation)")
print(bad_idx)
print(np.diff(bad_idx)) 

print("Any flagged frames in the 8000-11000 range?")
print(bad_idx[(bad_idx > 8000) & (bad_idx < 11000)])

worst_t = 1520530364.36  

window = gt_clean[(gt_clean.t > worst_t - 0.3) & (gt_clean.t < worst_t + 0.3)]
print(window[["t", "qw", "qx", "qy", "qz"]])

#Using gyro axis for strong dynamics
real_signal = imu_trimmed["wz"].values
ideal_signal = ideal_w[:, 2]  #reconstructed ideal wz

#removing means
real_c = real_signal - real_signal.mean()
ideal_c = ideal_signal - ideal_signal.mean()

corr = correlate(real_c, ideal_c, mode="full")
lags = correlation_lags(len(real_c), len(ideal_c), mode="full")

best_lag_samples = lags[np.argmax(corr)]
dt_nominal = 1 / 198.4  #measured sample rate
best_lag_seconds = best_lag_samples * dt_nominal

print(f"Estimated time offset: {best_lag_seconds*1000:.2f} ms "
      f"({best_lag_samples} samples)")

#Sub-sample refinement
peak_idx = np.argmax(corr)
y0, y1, y2 = corr[peak_idx-1], corr[peak_idx], corr[peak_idx+1]
sub_sample_offset = 0.5 * (y0 - y2) / (y0 - 2*y1 + y2)
refined_lag_samples = lags[peak_idx] + sub_sample_offset
refined_lag_seconds = refined_lag_samples * dt_nominal
print(f"Refined offset: {refined_lag_seconds*1000:.3f} ms")

def residual_cost(delta):
    shifted_t = imu_trimmed["t"].values + delta
    ideal_w_shifted = np.array([gt_angular_velocity(t) for t in shifted_t])
    resid = imu_trimmed[["wx","wy","wz"]].values - ideal_w_shifted
    return np.sum(resid**2)

"""
result = minimize_scalar(residual_cost, bounds=(-0.02, 0.02), method="bounded")
print(f"Optimization-based offset estimate: {result.x*1000:.3f} ms")


deltas = np.linspace(-0.02, 0.02, 200)
costs = [residual_cost(d) for d in deltas]

plt.plot(deltas * 1000, costs)
plt.xlabel("delta (ms)")
plt.ylabel("total residual energy")
plt.axvline(-1.719, color='r', linestyle='--', label='optimizer result')
plt.axvline(7.441, color='g', linestyle='--', label='cross-corr result')
plt.legend()
plt.show()


deltas_fine = np.linspace(-0.006, 0.004, 200)
costs_fine = [residual_cost(d) for d in deltas_fine]

best_idx = np.argmin(costs_fine)
best_delta_fine = deltas_fine[best_idx]
print(f"Refined minimum: {best_delta_fine*1000:.3f} ms")

plt.plot(np.array(deltas_fine)*1000, costs_fine)
plt.axvline(best_delta_fine*1000, color='r', linestyle='--')
plt.xlabel("delta (ms)"); plt.ylabel("residual energy")
plt.show()


deltas_fine2 = np.linspace(-0.003, -0.001, 100)
costs_fine2 = [residual_cost(d) for d in deltas_fine2]
best_delta_final = deltas_fine2[np.argmin(costs_fine2)]
print(f"Final offset estimate: {best_delta_final*1000:.3f} ms")


best_delta = 0.007441

imu_trimmed["t_corrected"] = imu_trimmed["t"] + best_delta
ideal_w_corrected = np.array([gt_angular_velocity(t) for t in imu_trimmed["t_corrected"]])
ideal_a_corrected = np.array([gt_linear_accel(t) for t in imu_trimmed["t_corrected"]])

residual_w_corrected = imu_trimmed[["wx","wy","wz"]].values - ideal_w_corrected
residual_a_corrected = imu_trimmed[["ax","ay","az"]].values - ideal_a_corrected

resid_df_sync_corrected = pd.DataFrame(
    np.hstack([imu_trimmed[["t"]].values, residual_w_corrected, residual_a_corrected]),
    columns=["t","res_wx","res_wy","res_wz","res_ax","res_ay","res_az"]
)

mask = np.ones(len(resid_df_sync_corrected), dtype=bool)
for t_start, t_end in exclude_windows:
    mask &= ~((resid_df_sync_corrected["t"] >= t_start) & (resid_df_sync_corrected["t"] <= t_end))
resid_df_sync_corrected = resid_df_sync_corrected[mask].reset_index(drop=True)

print(f"{'axis':<8} {'kurtosis before':<18} {'kurtosis after':<15} {'change'}")
for axis in ["res_wx","res_wy","res_wz","res_ax","res_ay","res_az"]:
    before = stats.kurtosis(resid_data_v2_clean[axis])
    after = stats.kurtosis(resid_df_sync_corrected[axis])
    print(f"{axis:<8} {before:<18.2f} {after:<15.2f} {after-before:+.2f}")
"""

def kurtosis_cost(delta, axis_index=2):  #axis_index=2 -> wz
    shifted_t = imu_trimmed["t"].values + delta
    ideal_w_shifted = np.array([gt_angular_velocity(t) for t in shifted_t])
    resid = imu_trimmed[["wx","wy","wz"]].values[:, axis_index] - ideal_w_shifted[:, axis_index]
    return stats.kurtosis(resid)

deltas = np.linspace(-0.01, 0.01, 100)
kurts = [kurtosis_cost(d) for d in deltas]

plt.plot(np.array(deltas)*1000, kurts)
plt.xlabel("delta (ms)")
plt.ylabel("res_wz kurtosis")
plt.axvline(5.04, color='orange', linestyle='--', label='5.04ms (energy sub-optimal)')
plt.axvline(-1.727, color='red', linestyle='--', label='-1.727ms (energy-optimal)')
plt.legend()
plt.show()


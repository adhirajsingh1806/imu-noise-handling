import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

resid_data = pd.read_csv("data/residuals_clean_v2.csv")

"""
for axis in ["res_wx", "res_wy", "res_wz", "res_ax", "res_ay", "res_az"]:
    result = adfuller(resid_data[axis].dropna())
    print(f"{axis}: ADF stat={result[0]:.3f}, p-value={result[1]:.4f}")
axis = "res_wz"
figs, axes = plt.subplots(1, 2, figsize = (12,4))
plot_acf(resid_data[axis], lags = 40, ax=axes[0])
plot_pacf(resid_data[axis], lags = 40, ax=axes[1])
plt.show()
"""
def select_order(series, max_p=6, max_q=6):
    best_aic = np.inf
    best_order = None
    all_results = {}
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0:
                continue
            try:
                model = ARIMA(series, order=(p, 0, q)).fit()
                all_results[(p, q)] = model.aic
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_order = (p, 0, q)
            except Exception:
                continue
    return best_order, best_aic, all_results

axes = ["res_wx", "res_wy", "res_wz", "res_ax", "res_ay", "res_az"]

"""
print("\nOrder selection grid search")
results_by_axis = {}
all_grid_results = {}

for axis in axes:
    order, aic, grid = select_order(resid_data[axis].values)
    results_by_axis[axis] = order
    all_grid_results[axis] = grid
    print(f"{axis}: best order = {order}, AIC = {aic:.1f}")


print("\nBoundary check")
max_p, max_q = 6, 6
for axis, order in results_by_axis.items():
    p, _, q = order
    at_boundary = (p == max_p) or (q == max_q)
    flag = "AT BOUNDARY, extend search" if at_boundary else ""
    print(f"{axis}: order={order}{flag}")

print("\nBIC comparison near each selected order")
for axis, (p, _, q) in results_by_axis.items():
    print(f"\n{axis}:")
    candidates = set()
    for dp in [-1, 0]:
        for dq in [-1, 0]:
            cp, cq = max(0, p + dp), max(0, q + dq)
            if cp == 0 and cq == 0:
                continue
            candidates.add((cp, cq))
    for cp, cq in sorted(candidates):
        try:
            m = ARIMA(resid_data[axis].values, order=(cp, 0, cq)).fit()
            print(f"  ({cp},0,{cq})  AIC={m.aic:.1f}  BIC={m.bic:.1f}")
        except Exception:
            print(f"  ({cp},0,{cq})  failed")

for p in range(6, 10):
    for q in range(4, 8):
        try:
            m = ARIMA(resid_data["res_wz"].values, order=(p, 0, q)).fit()
            print(f"p={p}, q={q}: AIC={m.aic:.1f}, BIC={m.bic:.1f}")
        except Exception:
            print(f"p={p}, q={q}: failed")

for p in range(6, 12):
    m = ARIMA(resid_data["res_ay"].values, order=(p, 0, 0)).fit()
    print(f"p={p}, q=0: AIC={m.aic:.1f}, BIC={m.bic:.1f}")

for p in range(2, 7):
    for q in range(0, 4):
        try:
            m = ARIMA(resid_data["res_ay"].values, order=(p, 0, q)).fit()
            print(f"p={p}, q={q}: AIC={m.aic:.1f}, BIC={m.bic:.1f}")
        except Exception:
            print(f"p={p}, q={q}: failed")
"""
orders = {
    "res_wx": (5, 0, 4),
    "res_wy": (4, 0, 3),
    "res_wz": (6, 0, 5),
    "res_ax": (4, 0, 5),
    "res_ay": (2, 0, 3),
    "res_az": (2, 0, 3),
}

fitted_models = {}
innovations = {}
t_params = {}

for axis, order in orders.items():
    print(f"{axis}: ARMA{order}")
    
    #Fit
    model = ARIMA(resid_data[axis].values, order=order).fit()
    fitted_models[axis] = model
    
    #Extract innovations 
    innov = model.resid
    innovations[axis] = innov

    #Ljung-Box
    lb = acorr_ljungbox(innov, lags=[10, 20, 40], return_df=True)
    print(lb)
    
    #Fit Student's-t to the innovations
    df, loc, scale = stats.t.fit(innov)
    t_params[axis] = (df, loc, scale)
    
    raw_kurt = stats.kurtosis(resid_data[axis].values)
    innov_kurt = stats.kurtosis(innov)
    
    print(f"AIC={model.aic:.1f}  BIC={model.bic:.1f}")
    print(f"kurtosis  raw residual={raw_kurt:.2f}  -->  innovations={innov_kurt:.2f}")
    print(f"Student-t fit: df={df:.2f}, loc={loc:.5f}, scale={scale:.5f}\n")


innov_data = pd.DataFrame(innovations)
innov_data.to_csv("data/innovations.csv", index=False)

summary_rows = []
for axis, order in orders.items():
    df_, loc, scale = t_params[axis]
    summary_rows.append({
        "axis": axis,
        "order": orders[axis],
        "AIC": fitted_models[axis].aic,
        "BIC": fitted_models[axis].bic,
        "raw_kurtosis": stats.kurtosis(resid_data[axis].values),
        "innovation_kurtosis": stats.kurtosis(innovations[axis]),
        "ljungbox_p_lag10": lb["lb_pvalue"].iloc[0],
        "ljungbox_p_lag20": lb["lb_pvalue"].iloc[1],
        "ljungbox_p_lag40": lb["lb_pvalue"].iloc[2],
        "t_df": df_,
        "t_loc": loc,
        "t_scale": scale,
    })
summary_data = pd.DataFrame(summary_rows)
print(summary_data)
summary_data.to_csv("data/arma_student_t_summary.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(innovations["res_ax"])
axes[0].set_title("res_ax innovations")
axes[1].plot(innovations["res_ay"])
axes[1].set_title("res_ay innovations")
plt.tight_layout()
plt.show()

for axis in ["res_ax", "res_ay"]:
    innov = innovations[axis]
    worst_idx = np.argsort(np.abs(innov))[-10:] 
    print(f"\n{axis} — 10 largest innovations:")
    for i in sorted(worst_idx):
        print(f"  index {i}: value={innov[i]:.4f}")

def get_contiguous_segments(df, time_col="t", max_gap_factor=3):
    dt = np.diff(df[time_col].values)
    typical_dt = np.median(dt)
    gap_indices = np.where(dt > typical_dt * max_gap_factor)[0]
    
    segments = []
    start = 0
    for gap_i in gap_indices:
        segments.append(df.iloc[start:gap_i+1].reset_index(drop=True))
        start = gap_i + 1
    segments.append(df.iloc[start:].reset_index(drop=True))
    
    return [s for s in segments if len(s) > 50]  

segments = get_contiguous_segments(resid_data)
print(f"{len(segments)} contiguous segments, lengths: {[len(s) for s in segments]}")

def fit_arma_multisegment(segments, axis, order, burn_in=10):
    all_innovations = []
    for seg in segments:
        if len(seg) < 50:
            continue
        model = ARIMA(seg[axis].values, order=order).fit()
        all_innovations.append(model.resid[burn_in:])  
    return np.concatenate(all_innovations)

innovations_fixed = {}
for axis, order in orders.items():
    innovations_fixed[axis] = fit_arma_multisegment(segments, axis, order)
    print(f"{axis}: kurtosis = {stats.kurtosis(innovations_fixed[axis]):.2f}")

final_summary_rows = []
for axis, order in orders.items():
    all_innov = []
    for seg in segments:
        if len(seg) < 50:
            continue
        m = ARIMA(seg[axis].values, order=order).fit()
        all_innov.append(m.resid[10:])
    innov_all = np.concatenate(all_innov)

    df_, loc, scale = stats.t.fit(innov_all)
    raw_kurt = stats.kurtosis(resid_data[axis].values)
    innov_kurt = stats.kurtosis(innov_all)

    final_summary_rows.append({
        "axis": axis, "order": order,
        "raw_kurtosis": raw_kurt, "innovation_kurtosis": innov_kurt,
        "t_df": df_, "t_loc": loc, "t_scale": scale,
    })

final_summary_df = pd.DataFrame(final_summary_rows)
print(final_summary_df)
final_summary_df.to_csv("data/final_arma_student_t_summary.csv", index=False)
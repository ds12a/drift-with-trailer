import numpy as np
from pathlib import Path
from src.learning.datasets.trailer_data import DataStore

ds = DataStore.load(Path("./experiments/exp_007_vehicle_residual_dynamics/data_raw_aug.npz"))
vx = ds.data[:, 2]

print(f"N={len(vx):,}   min={vx.min():.1f}  max={vx.max():.1f}  "
      f"mean={vx.mean():.2f}  std={vx.std():.2f}")
print(f"reverse frac: {(vx < 0).mean():.3f}   |vx|<0.5: {(np.abs(vx) < 0.5).mean():.4f}")

# signed histogram, 2 m/s bins
edges = np.arange(np.floor(vx.min() / 2) * 2, np.ceil(vx.max() / 2) * 2 + 2, 2.0)
cnt, _ = np.histogram(vx, edges)
peak = cnt.max()
for lo, hi, c in zip(edges[:-1], edges[1:], cnt):
    bar = "#" * int(60 * c / peak)
    print(f"{lo:6.0f}..{hi:<5.0f} {c/1e3:8.1f}k {c/len(vx)*100:5.2f}%  {bar}")

# where the tail starts, forward only
fwd = vx[vx > 0]
print("\nforward |vx| quantiles (m/s / kph):")
for q in (0.5, 0.9, 0.95, 0.99, 0.999):
    v = np.quantile(fwd, q)
    print(f"  p{q*100:5.1f}  {v:6.2f}  {v*3.6:6.1f}")
hi = np.abs(vx) > 11.0                      # 40 kph
print(f"\n>40kph: {hi.mean()*100:.2f}%  ({hi.sum()/1e3:.0f}k rows)")
print("  by mu:", {round(float(m),1): round(float(hi[ds.data[:,6]==m].mean()*100), 2)
                   for m in np.unique(ds.data[:, 6])})
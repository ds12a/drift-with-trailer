"""Single-panel figure: the final Catalunya environment (snow + slope), rotated
so the start/finish straight is horizontal, with spawn markers and direction
arrows at track indices 0, 800, 1500.

Usage:
    python make_env_final.py LEVEL.zip CENTERLINE.csv [OUT_PREFIX]

LEVEL.zip is a BeamNG level export containing a `.ter` terrain file anywhere
inside it (found automatically -- does not need to be named terrain8.ter or
sit at any particular path). Nothing is extracted to disk; the zip is read
directly via `zipfile`, and the parsed heightmap/layer arrays are also cached
to OUT_PREFIX_arrays.npz so a second run (e.g. to retune the figure) can load
that instead of re-parsing the zip -- pass the .npz in place of the zip and
it will be detected and used directly.
"""
import sys, struct, zipfile
from matplotlib.path import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from scipy.ndimage import map_coordinates

SNOW_IDX = 4
SPAWNS = [0, 800, 1500]           # track indices to mark
ARROW_LEN = 70.0                   # m
MARGIN = 70.0

XMARGIN = MARGIN
YMARGIN = MARGIN + 90.0

TRI_LEN = 30.0      # m, tip-to-base along travel direction
TRI_BACK = 7.5
TRI_WIDTH = 30.0    # m, full width of the base
TRI_EDGE_LW = 0.75   # outline width (points)

def tri_verts(cx, cy, dx, dy, length=TRI_LEN, width=TRI_WIDTH, back=TRI_BACK):
    """Concave dart centred on (cx,cy), pointing along unit vector (dx,dy)."""
    fwd  = np.array([dx, dy], float)
    perp = np.array([-dy, dx], float)
    c    = np.array([cx, cy], float)
    tip    = c + fwd * (length * 0.5)
    base_c = c - fwd * (length * 0.5)
    return np.array([tip,
                     base_c + perp * (width * 0.5) - fwd * back,
                     base_c,
                     base_c - perp * (width * 0.5) - fwd * back])

# Legend marker built from the SAME vertex function, in a canonical frame
# pointing +x and normalised into a ~unit box (matplotlib scales Path markers
# by markersize). Derived rather than hardcoded so it tracks any retune of
# TRI_LEN / TRI_WIDTH / TRI_BACK automatically.
_v = tri_verts(0, 0, 1, 0)
_v = _v - _v.mean(0)
_v = _v / np.abs(_v).max() * 0.5
TRI_MARKER = Path(np.vstack([_v, _v[:1]]),
                  [Path.MOVETO] + [Path.LINETO]*3 + [Path.CLOSEPOLY])


def read_ter_bytes(b, max_height=256.0):
    """Parse a BeamNG/Torque3D .ter (v9) from raw bytes.

    Layout is header/heightMap/layerMap/materialNames -- note the sidecar
    `<name>.terrain.json` some levels ship documents an extra layerTextureMap
    plane between layerMap and materialNames that is NOT actually present in
    the file; this layout is what consumes real exports to the exact byte.
    """
    size = struct.unpack_from("<I", b, 1)[0]
    n, off = size * size, 5
    h = np.frombuffer(b, "<u2", n, off).reshape(size, size).copy(); off += 2 * n
    layer = np.frombuffer(b, np.uint8, n, off).reshape(size, size).copy(); off += n
    cnt = struct.unpack_from("<I", b, off)[0]; off += 4
    names = []
    for _ in range(cnt):
        L = b[off]; off += 1
        names.append(b[off:off + L].decode()); off += L
    assert off == len(b), f".ter parse error: {len(b)-off} trailing bytes"
    z = h.astype(np.float32) * max_height / 65535.0
    return z, layer, names


def load_terrain(path):
    """Accepts a level .zip (searches it for the first *.ter member) or a
    previously-cached _arrays.npz (z, layer, names)."""
    if path.endswith(".npz"):
        d = np.load(path, allow_pickle=True)
        return d["z"], d["layer"], list(d["names"])
    with zipfile.ZipFile(path) as zf:
        ter_names = [n for n in zf.namelist() if n.lower().endswith(".ter")]
        if not ter_names:
            raise FileNotFoundError(f"no .ter file found inside {path}")
        if len(ter_names) > 1:
            print(f"warning: {len(ter_names)} .ter files in zip, using {ter_names[0]}")
        b = zf.read(ter_names[0])
    return read_ter_bytes(b)


if len(sys.argv) < 3:
    print(__doc__); sys.exit(1)
LEVEL_PATH = sys.argv[1]
CSV = sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else "env_final"

z, layer, names = load_terrain(LEVEL_PATH)
print(f"terrain: {z.shape[0]}x{z.shape[1]}, elev {z.min():.1f}..{z.max():.1f} m, "
      f"materials {names}")
if not LEVEL_PATH.endswith(".npz"):
    np.savez_compressed(f"{OUT}_arrays.npz", z=z, layer=layer, names=np.array(names))
    print(f"cached arrays -> {OUT}_arrays.npz")

cl = np.genfromtxt(CSV, delimiter=",", names=True)
X, Y = cl["x"], cl["y"]
n = len(X)
s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(X), np.diff(Y)))])
L = s[-1]

# --- main-straight heading: longest run with chord/arc > 0.998, no wraparound ---
def chord_arc(i0, npts):
    i1 = i0 + npts
    if i1 >= n:
        return 0.0
    return np.hypot(X[i1] - X[i0], Y[i1] - Y[i0]) / (s[i1] - s[i0])

best_npts = max(npts for npts in range(50, 600, 2)
                if max(chord_arc(i0, npts) for i0 in range(n - npts)) > 0.998)
ratios = np.array([chord_arc(i0, best_npts) for i0 in range(n)])
i0s = int(ratios.argmax())
heading = np.degrees(np.arctan2(Y[i0s + best_npts] - Y[i0s], X[i0s + best_npts] - X[i0s]))
print(f"main straight: idx {i0s}-{i0s+best_npts} ({s[i0s+best_npts]-s[i0s]:.0f} m), "
      f"heading {heading:.1f} deg")

# --- rotate everything so that heading -> 0 (horizontal) ---
# Forward transform applied to the TRACK: world_rot = R(theta) @ world_orig.
# scipy.ndimage.rotate operates on (row, col) array indices, not on (x, y) world
# coordinates, and the two are NOT the same transform once you account for the
# array's row/col <-> y/x mapping -- using it here silently rotated the terrain
# by the wrong handedness relative to the track (checked visually: snow patches
# and contours no longer tracked the road at all). Resampling explicitly with
# map_coordinates and the SAME rot() used for the track sidesteps the ambiguity
# entirely, since both go through one transform definition.
theta = np.radians(-heading)
c, si = np.cos(theta), np.sin(theta)
def rot(px, py):
    return c * px - si * py, si * px + c * py

def rot_inv(px, py):
    return c * px + si * py, -si * px + c * py

Xr, Yr = rot(X, Y)

ORIGIN, SQUARE = -1024.0, 1.0
OUT_SIZE = z.shape[0]
extent = [-1024, 1024, -1024, 1024]
gy_out, gx_out = np.mgrid[extent[2]:extent[3]:OUT_SIZE*1j, extent[0]:extent[1]:OUT_SIZE*1j]
gx0, gy0 = rot_inv(gx_out, gy_out)               # rotated-frame -> original world coords
col0 = (gx0 - ORIGIN) / SQUARE
row0 = (gy0 - ORIGIN) / SQUARE
z_r = map_coordinates(z, [row0, col0], order=1, mode="constant", cval=np.nan)
layer_r = map_coordinates(layer.astype(np.float32), [row0, col0], order=0,
                          mode="constant", cval=-1)

# spawn points + direction arrows (tangent at each index; reverse flips it)
spawn_xy, spawn_dir = [], []
for i in SPAWNS:
    p0 = np.array([X[i], Y[i]])
    p1 = np.array([X[(i + 3) % n], Y[(i + 3) % n]])
    t = p1 - p0
    t = t / (np.linalg.norm(t) + 1e-9)
    spawn_xy.append(p0)
    spawn_dir.append(t)
spawn_xy = np.array(spawn_xy)
spawn_dir = np.array(spawn_dir)
sxr, syr = rot(spawn_xy[:, 0], spawn_xy[:, 1])
dxr, dyr = rot(spawn_dir[:, 0], spawn_dir[:, 1])

x0, x1 = Xr.min() - XMARGIN, Xr.max() + XMARGIN
y0, y1 = Yr.min() - YMARGIN, Yr.max() + YMARGIN
pad_y = 0.08 * (y1 - y0)   # a little more headroom than MARGIN alone gives

TRACK = LinearSegmentedColormap.from_list(
    "track", ["#2c3e70", "#3b8ea5", "#e0b83c", "#c0392b"])
GRADE_CMAP = plt.get_cmap("viridis")
    # diverging, centered at 0%: blue = downhill, white = flat, red = uphill

aspect = (Yr.max()-Yr.min()+2*MARGIN) / (Xr.max()-Xr.min()+2*MARGIN)
fig, ax = plt.subplots(figsize=(7.0, max(2.6, 7.0*aspect*1.12)), dpi=300,
                       constrained_layout=True)

gy, gx = gy_out, gx_out
lo, hi = np.nanmin(z_r), np.nanmax(z_r)
ax.contour(gx, gy, z_r, levels=np.arange(lo, hi, 5.0),
          colors="#b8a888", linewidths=0.2, alpha=0.8, zorder=1)
ax.contour(gx, gy, z_r, levels=np.arange(lo, hi, 25.0),
          colors="#8a7654", linewidths=0.5, alpha=0.9, zorder=2)

snow = np.ma.masked_where(layer_r != SNOW_IDX, np.ones_like(z_r))
ax.imshow(snow, origin="lower", extent=extent, cmap="Blues_r",
         vmin=0, vmax=1.6, alpha=0.55, interpolation="nearest", zorder=3)


zt = map_coordinates(z, [Y + 1024, X + 1024], order=1)
zrel = zt - zt.min()                       # still used for the "N m relief" title
grade = 100.0 * np.gradient(zt, s)         # signed percent grade, dz/ds along arc
grade_seg = 0.5 * (grade[:-1] + grade[1:]) # one value per segment, matching zrel[:-1] before
gmax = np.abs(grade).max()

pts = np.array([Xr, Yr]).T.reshape(-1, 1, 2)
lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1), cmap=GRADE_CMAP,
                    norm=Normalize(-gmax, gmax), array=grade_seg,
                    linewidth=2.5, zorder=5, capstyle="round", joinstyle="round")
ax.add_collection(lc)

SPAWN_COLORS = ["#e207ff", "#ff6f00", "#00cb98"]
for (sx, sy, dx, dy, idx, col) in zip(sxr, syr, dxr, dyr, SPAWNS, SPAWN_COLORS):
    # Triangle as an explicit world-coordinate polygon: tip TRI_LEN/2 ahead of the
    # spawn point along travel, base TRI_WIDTH across, centred on the spawn point.
    fwd = np.array([dx, dy])
    perp = np.array([-dy, dx])
    tip = np.array([sx, sy]) + fwd * (TRI_LEN * 0.5)
    base_c = np.array([sx, sy]) - fwd * (TRI_LEN * 0.5)
    tri = np.array([tip,
                    base_c + perp * (TRI_WIDTH * 0.5) - fwd * TRI_BACK,
                    base_c,
                    base_c - perp * (TRI_WIDTH * 0.5) - fwd * TRI_BACK])
    ax.add_patch(Polygon(tri, closed=True, facecolor=col, edgecolor="black",
                         linewidth=TRI_EDGE_LW, zorder=8))

ax.set_xlim(x0, x1); ax.set_ylim(y0 - pad_y*0.15, y1 + pad_y*0.15)
ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_linewidth(0.6); sp.set_color("0.4")

on = layer[np.clip((Y+1024).astype(int),0,2047), np.clip((X+1024).astype(int),0,2047)] == SNOW_IDX
# ax.set_title(
#     f"Snow + slope   |   {100*on.mean():.0f}% snow on track   $\\cdot$   "
#     f"{zrel.max():.0f} m relief", fontsize=9.5)

xb, yb = x1 - 30 - 200, y0 + 26
ax.plot([xb, xb+200], [yb]*2, color="0.25", lw=1.4, solid_capstyle="butt", zorder=6)
ax.text(xb+100, yb+12, "200 m", ha="center", va="bottom", fontsize=7, color="0.25", zorder=6)

sm_ = plt.cm.ScalarMappable(cmap=GRADE_CMAP, norm=Normalize(-gmax, gmax))
cb = fig.colorbar(sm_, ax=ax, fraction=0.045, pad=0.02)
cb.set_label("track grade [%]", fontsize=8)
cb.ax.tick_params(labelsize=7)

handles = [Patch(fc="#a8cfec", ec="0.6", lw=0.4, alpha=0.55, label="Snow"),
          Line2D([], [], color="#8a7654", lw=0.6, label="Contours (5/25 m)")]
handles += [Line2D([0],[0], marker=TRI_MARKER, color="none", markerfacecolor=col,
                   markersize=11, markeredgecolor="black", markeredgewidth=0.7,
                   label=f"Spawnpoint {i}")
           for i, (idx, col) in enumerate(zip(SPAWNS, SPAWN_COLORS))]
ax.legend(handles=handles, loc="lower left", fontsize=6.4, frameon=True,
         framealpha=0.9, borderpad=0.4, handlelength=1.5)

for e in ("pdf", "png"):
    fig.savefig(f"{OUT}.{e}", bbox_inches="tight")
print("wrote", f"{OUT}.pdf", "/", f"{OUT}.png")
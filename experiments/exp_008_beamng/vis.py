import glob
import os
import re
import numpy as np
import cv2

files = list(reversed(sorted(glob.glob('/home/dshen/Downloads/snowwpath/*') )))
files = files[::2] + [files[-1]] # every other frame, so overlapping trailers have more separation
frames = [cv2.imread(f, cv2.IMREAD_COLOR).astype(np.float32) for f in files]
H, W = frames[0].shape[:2]

# --- Regions occupied by the game/OS HUD (title bar, icon row, time counter,
#     gauge cluster + indicator icons, taskbar). Coordinates in pixel space.
os_chrome = [(0, 0, W, 40), (0, 1540, W, H)]              # OS title bar / taskbar -> crop away
hud_widgets = [
    (1950, 40, W, 155),       # in-game time counter widget
    (1850, 1280, W, H),       # bottom-right dash cluster (gauge + H2/indicator icons)
    (2060, 460, W, 1600),     # Spectacle screenshot-notification popup stack (variable per frame)
]

def rects_to_mask(h, w, rects):
    m = np.zeros((h, w), dtype=np.uint8)
    for (x0, y0, x1, y1) in rects:
        m[y0:y1, x0:x1] = 255
    return m

chrome_mask = rects_to_mask(H, W, os_chrome)
hud_mask = rects_to_mask(H, W, hud_widgets)
ui_mask = (chrome_mask.astype(bool)) | (hud_mask.astype(bool))

# --- Background: per-pixel median across all frames
stack = np.stack(frames, axis=0)
background = np.median(stack, axis=0)

# Remove the in-game HUD widgets with a clean solid fill (large structured
# regions like the gauge dial inpaint poorly, producing smeared artifacts)
bg_clean = background.copy()
for (x0, y0, x1, y1) in hud_widgets:
    bg_clean[y0:y1, x0:x1] = (255, 255, 255)

composite = bg_clean.copy()

# The RGB distance between a low-contrast vehicle and the snow can be only a
# few 8-bit levels.  The captures are lossless and the rest of the scene is
# static, so a much smaller cutoff is safe here.  Keep this configurable for
# captures with more screenshot noise, where a higher value may be useful.
diff_thresh = float(os.environ.get('VIS_DIFF_THRESH', '22.0'))
min_blob_area = 10

centroids = []
close_kernel = np.ones((15, 15), np.uint8)

for frame in frames:
    diff = np.linalg.norm(frame - background, axis=2)
    mask = diff > diff_thresh
    mask[ui_mask] = False

    mask_u8 = (mask.astype(np.uint8)) * 255

    # Bridge small gaps (e.g. low-contrast hitch/coupling against snow) so the
    # tractor and trailer are grouped as one component instead of being split
    dilated = cv2.dilate(mask_u8, close_kernel, iterations=1)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    if n <= 1:
        centroids.append(None)
        continue
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    if stats[best, cv2.CC_STAT_AREA] < min_blob_area:
        centroids.append(None)
        continue

    # paste the full merged (dilated) region — the coupling/connector area
    # between cab and trailer is often near-identical to the snow background,
    # so restricting to only "detected" pixels there leaves a visible gap
    clean = labels == best
    ys, xs = np.nonzero(mask & clean)
    if len(xs) == 0:
        ys, xs = np.nonzero(clean)
    centroids.append((float(xs.mean()), float(ys.mean())))

    composite[clean] = frame[clean]

# --- Crop away OS chrome (title bar, taskbar); keep in-game viewport only
composite = composite[40:1540, 0:W]

cv2.imwrite('/home/dshen/Downloads/out.png', np.clip(composite, 0, 255).astype(np.uint8))

print("Capture times, for the figure caption:")
for order, i in enumerate([i for i, c in enumerate(centroids) if c is not None], start=1):
    m = re.search(r'_(\d{2})(\d{2})(\d{2})\.png$', files[i])
    print(f"  {order}: {m.group(1)}:{m.group(2)}:{m.group(3)}")

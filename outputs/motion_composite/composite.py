"""Video adaptation of the user's bounding-box compositing script."""
from pathlib import Path
import glob
import json
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent
VIDEO = '/home/dshen/Downloads/Screencast from 09-21-2026 02:09:52 PM.webm'
TIMES = [3, 6, 9, 16, 24, 30, 36]
FRAME_GLOB = str(ROOT / 'selected' / '*.png')
OUTPUT_PATH = str(ROOT / 'composite.png')
hud_widgets = [(1375, 0, 1821, 85), (1515, 700, 1821, 1009), (1405, 895, 1520, 1009)]
(ROOT / 'selected').mkdir(exist_ok=True)
for old in (ROOT / 'selected').glob('*.png'):
    old.unlink()
cap = cv2.VideoCapture(VIDEO)
for t in TIMES:
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f'Cannot read frame at {t}s')
    cv2.imwrite(str(ROOT / 'selected' / f'{t:05.1f}s.png'), frame)
cap.release()
files = sorted(glob.glob(FRAME_GLOB))
frames = [cv2.imread(f, cv2.IMREAD_COLOR).astype(np.float32) for f in files]
H, W = frames[0].shape[:2]

def rects_to_mask(h, w, rects):
    m = np.zeros((h, w), dtype=np.uint8)
    for x0, y0, x1, y1 in rects:
        m[y0:y1, x0:x1] = 255
    return m

ui_mask = rects_to_mask(H, W, hud_widgets).astype(bool)
stack = np.stack(frames, axis=0)
background = np.median(stack, axis=0)
del stack
composite = background.copy()
diff_thresh = 22.0
min_blob_area = 40
close_kernel = np.ones((9, 9), np.uint8)
centroids = []
records = []
for file, frame in zip(files, frames):
    diff = np.linalg.norm(frame - background, axis=2)
    mask = diff > diff_thresh
    mask[ui_mask] = False
    mask_u8 = mask.astype(np.uint8) * 255
    dilated = cv2.dilate(mask_u8, close_kernel, iterations=1)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    if n <= 1:
        centroids.append(None)
        continue
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[best, cv2.CC_STAT_AREA] < min_blob_area:
        centroids.append(None)
        continue
    bx, by, bw, bh, _ = stats[best]
    anchor_cx, anchor_cy = bx + bw / 2., by + bh / 2.
    x0f, y0f, x1f, y1f = bx, by, bx + bw, by + bh
    for lbl in range(1, n):
        if lbl == best or stats[lbl, cv2.CC_STAT_AREA] < min_blob_area:
            continue
        lx, ly, lw, lh, _ = stats[lbl]
        lcx, lcy = lx + lw / 2., ly + lh / 2.
        if np.hypot(lcx - anchor_cx, lcy - anchor_cy) < (200 if anchor_cy > 550 else 35):
            x0f, y0f = min(x0f, lx), min(y0f, ly)
            x1f, y1f = max(x1f, lx + lw), max(y1f, ly + lh)
    x, y, w, h = x0f, y0f, x1f - x0f, y1f - y0f
    pad = 6
    bx0, by0 = max(0, x-pad), max(0, y-pad)
    bx1, by1 = min(W, x+w+pad), min(H, y+h+pad)
    centroids.append((x+w/2., y+h/2.))
    composite[by0:by1, bx0:bx1] = frame[by0:by1, bx0:bx1]
    records.append({'file': file, 'bbox': list(map(int,[bx0,by0,bx1,by1])), 'centroid': centroids[-1]})
cv2.imwrite(OUTPUT_PATH, np.clip(composite,0,255).astype(np.uint8))
(ROOT / 'detections.json').write_text(json.dumps(records, indent=2))
for r in records: print(Path(r['file']).name,r['bbox'],r['centroid'])

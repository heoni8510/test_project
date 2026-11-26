import numpy as np, sys, os
from PIL import Image, ImageFilter
src, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)
rng = np.random.default_rng(52)
im = Image.open(src).convert('RGB').crop((3, 3, 373, 495))   # remove label band & edge
a = np.asarray(im).astype(np.float32)

# 1) flatten uneven scan lighting (keep fine grain, remove big gradients)
def blur(arr, r):
    return np.asarray(Image.fromarray(np.clip(arr,0,255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(r))).astype(np.float32)
low = blur(a, 45)
a = a - low + low.reshape(-1, 3).mean(0)

# 2) rotate: grain runs along the 1500mm length
a = np.rot90(a, 1).copy()            # now H=370 (across grain), W=492 (along grain)

# 3) make seamless both ways with cross-fade overlap
def seam(arr, axis, ov):
    n = arr.shape[axis]
    t = np.linspace(0, 1, ov, dtype=np.float32)
    t = t*t*(3-2*t)
    shp = [1, 1, 1]; shp[axis] = ov; t = t.reshape(shp)
    head = np.take(arr, range(ov), axis=axis)
    tail = np.take(arr, range(n-ov, n), axis=axis)
    body = np.take(arr, range(ov, n-ov), axis=axis)
    blend = tail*(1-t) + head*t
    return np.concatenate([body, blend], axis=axis)
a = seam(a, 1, 110)   # along grain
a = seam(a, 0, 24)    # across grain

# 4) real scale: sample sheet ~ A4 (297 x 210 mm) -> 2 px/mm
PX = 2
tile_h = int(round(210*PX*a.shape[0]/370)); tile_w = int(round(297*PX*a.shape[1]/492))
tile = np.asarray(Image.fromarray(np.clip(a,0,255).astype(np.uint8)).resize((tile_w, tile_h), Image.LANCZOS)).astype(np.float32)
tile_img = Image.fromarray(np.clip(tile,0,255).astype(np.uint8)).filter(ImageFilter.UnsharpMask(1.2, 40, 2))
tile_img.save(f'{outdir}/SPW52_seamless_tile.jpg', quality=95)
tile = np.asarray(tile_img).astype(np.float32)
print('tile px', tile_w, tile_h, 'mm', tile_w/PX, tile_h/PX)

# 5) full face 1500 x 760 mm, broken into grain-parallel strips to avoid visible repetition
W, H = 1500*PX, 760*PX
canvas = np.zeros((H, W, 3), np.float32)
def strip(h):
    ox, oy = rng.integers(0, tile_w), rng.integers(0, tile_h)
    t = tile[::-1] if rng.random() < .5 else tile
    ys = (np.arange(h) + oy) % tile_h
    xs = (np.arange(W) + ox) % tile_w
    s = t[ys][:, xs]
    return s * np.float32(rng.uniform(0.99, 1.01))  # subtle tone shift per board
# boundaries with gentle waviness, feathered
bounds = [0]
while bounds[-1] < H:
    bounds.append(bounds[-1] + int(rng.integers(220, 460)))
strips = [strip(H) for _ in bounds]
x = np.arange(W)
acc = strips[0].copy()
for i, b in enumerate(bounds[1:-1], start=1):
    wave = b + 10*np.sin(x/W*2*np.pi*rng.uniform(.5, 1.5) + rng.uniform(0, 6))
    yy = np.arange(H)[:, None]
    m = np.clip((yy - wave[None, :]) / 6 + .5, 0, 1)[..., None]
    acc = acc*(1-m) + strips[i]*m
img = Image.fromarray(np.clip(acc, 0, 255).astype(np.uint8))
img.save(f'{outdir}/SPW52_1500x760mm_horizontal.jpg', quality=95)
img.rotate(90, expand=True).save(f'{outdir}/SPW52_760x1500mm_vertical.jpg', quality=95)
img.resize((W//4, H//4), Image.LANCZOS).save(f'{outdir}/_preview.jpg', quality=90)

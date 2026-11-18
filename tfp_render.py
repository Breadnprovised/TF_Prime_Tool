"""Small software renderer (numpy + Pillow) used for model previews."""
import numpy as np
from PIL import Image, ImageDraw


def rot(yaw, pitch):
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return rx @ ry


def render(meshes, size=(640, 640), yaw=0.6, pitch=-0.25, zoom=1.0, pan=(0, 0), tex_colors=None,
           wire=False, bg=(40, 44, 52)):
    """meshes: list of dicts with 'position' (N,3), 'indices' (M,3), optional 'uv', 'material'.
    tex_colors: optional callable(material, uv(M,2)) -> (M,3) colours for triangles."""
    W, H = size
    img = Image.new('RGB', size, bg)
    allp = [m['position'] for m in meshes if 'position' in m and len(m['indices'])]
    if not allp:
        return img
    P = np.concatenate(allp)
    lo, hi = P.min(0), P.max(0)
    center = (lo + hi) / 2
    radius = max(float(np.linalg.norm(hi - lo)) / 2, 1e-6)
    R = rot(yaw, pitch)
    scale = min(W, H) * 0.45 / radius * zoom
    polys, depth, cols = [], [], []
    light = np.array([0.3, 0.5, 0.8]); light /= np.linalg.norm(light)
    for m in meshes:
        if 'position' not in m or not len(m['indices']):
            continue
        v = (m['position'] - center) @ R.T
        tri = m['indices'].astype(np.int64)
        a, b, c = v[tri[:, 0]], v[tri[:, 1]], v[tri[:, 2]]
        n = np.cross(b - a, c - a)
        nl = np.linalg.norm(n, axis=1, keepdims=True); nl[nl == 0] = 1
        n /= nl
        shade = np.abs(n @ light) * 0.75 + 0.25
        if tex_colors is not None and 'uv' in m:
            uvc = (m['uv'][tri[:, 0]] + m['uv'][tri[:, 1]] + m['uv'][tri[:, 2]]) / 3
            base = tex_colors(m.get('material', 0), uvc)
        else:
            hue = (m.get('material', 0) * 0.37) % 1.0
            base = np.tile(np.array(_hsv(hue, 0.25, 0.9)) * 255, (len(tri), 1))
        col = np.clip(base * shade[:, None], 0, 255).astype(np.uint8)
        sx = lambda q: np.stack([q[:, 0] * scale + W / 2 + pan[0], -q[:, 1] * scale + H / 2 + pan[1]], -1)
        polys.append(np.stack([sx(a), sx(b), sx(c)], 1))
        depth.append((a[:, 2] + b[:, 2] + c[:, 2]) / 3)
        cols.append(col)
    polys = np.concatenate(polys); depth = np.concatenate(depth); cols = np.concatenate(cols)
    order = np.argsort(depth)  # far (negative z) first
    dr = ImageDraw.Draw(img)
    pl = polys[order].tolist(); cl = cols[order].tolist()
    if wire:
        for p in pl:
            dr.line([tuple(p[0]), tuple(p[1]), tuple(p[2]), tuple(p[0])], fill=(200, 210, 220))
    else:
        for p, c in zip(pl, cl):
            dr.polygon([tuple(p[0]), tuple(p[1]), tuple(p[2])], fill=tuple(c))
    return img


def _hsv(h, s, v):
    import colorsys
    return colorsys.hsv_to_rgb(h, s, v)


def texture_sampler(textures_rgba, materials, flip_v=False):
    """Build a tex_colors callable: material -> layer0 texture -> sample at uv."""
    def f(mat, uv):
        t = None
        if mat < len(materials) and materials[mat]['layers']:
            ti = materials[mat]['layers'][0]['texture']
            if ti < len(textures_rgba):
                t = textures_rgba[ti]
        if t is None:
            return np.full((len(uv), 3), 200.0)
        h, w = t.shape[:2]
        u = np.mod(uv[:, 0], 1.0); v = np.mod(uv[:, 1], 1.0)
        if flip_v:
            v = 1 - v
        x = np.clip((u * w).astype(int), 0, w - 1); y = np.clip((v * h).astype(int), 0, h - 1)
        return t[y, x, :3].astype(np.float32)
    return f


def render_hq(meshes, size=(640, 640), yaw=0.6, pitch=-0.25, zoom=1.0, pan=(0, 0), textures=None,
              materials=None, flip_v=False, use_textures=True, bg=(40, 44, 52), max_tris=200000):
    """Per-pixel z-buffered renderer: interpolated UVs (textures), vertex colours and smooth normals."""
    W, H = size
    allp = [m['position'] for m in meshes if 'position' in m and len(m['indices'])]
    img = np.empty((H, W, 3), np.float32); img[:] = bg
    if not allp:
        return Image.fromarray(img.astype(np.uint8))
    P = np.concatenate(allp)
    lo, hi = P.min(0), P.max(0)
    center = (lo + hi) / 2
    radius = max(float(np.linalg.norm(hi - lo)) / 2, 1e-6)
    R = rot(yaw, pitch)
    scale = min(W, H) * 0.45 / radius * zoom
    zbuf = np.full((H, W), -np.inf, np.float32)
    light = np.array([0.3, 0.5, 0.8]); light /= np.linalg.norm(light)
    total = 0
    for m in meshes:
        if 'position' not in m or not len(m['indices']):
            continue
        v = (m['position'][:, :3] - center) @ R.T
        sx = v[:, 0] * scale + W / 2 + pan[0]
        sy = -v[:, 1] * scale + H / 2 + pan[1]
        sz = v[:, 2]
        if 'normal' in m:
            nrm = m['normal'][:, :3] @ R.T
        else:
            nrm = None
        tex = None
        mi = m.get('material', 0)
        if use_textures and textures and materials and mi < len(materials) and materials[mi]['layers'] and 'uv' in m:
            ti = materials[mi]['layers'][0]['texture']
            if ti < len(textures):
                tex = textures[ti]
        uv = m.get('uv')
        vcol = m['color'][:, :3].astype(np.float32) if 'color' in m else None
        tris = m['indices'].astype(np.int64)
        total += len(tris)
        if total > max_tris:
            break
        for t in tris:
            x0, x1, x2 = sx[t]; y0, y1, y2 = sy[t]
            minx = max(int(np.floor(min(x0, x1, x2))), 0); maxx = min(int(np.ceil(max(x0, x1, x2))), W - 1)
            miny = max(int(np.floor(min(y0, y1, y2))), 0); maxy = min(int(np.ceil(max(y0, y1, y2))), H - 1)
            if minx > maxx or miny > maxy:
                continue
            den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(den) < 1e-12:
                continue
            ys, xs = np.mgrid[miny:maxy + 1, minx:maxx + 1]
            px = xs + 0.5; py = ys + 0.5
            w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / den
            w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / den
            w2 = 1 - w0 - w1
            inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
            if not inside.any():
                continue
            z = w0 * sz[t[0]] + w1 * sz[t[1]] + w2 * sz[t[2]]
            zb = zbuf[miny:maxy + 1, minx:maxx + 1]
            mask = inside & (z > zb)
            if not mask.any():
                continue
            a, b, c = w0[mask], w1[mask], w2[mask]
            if nrm is not None:
                n = a[:, None] * nrm[t[0]] + b[:, None] * nrm[t[1]] + c[:, None] * nrm[t[2]]
                n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
                shade = np.abs(n @ light) * 0.7 + 0.3
            else:
                fn = np.cross(v[t[1]] - v[t[0]], v[t[2]] - v[t[0]]); fn /= max(np.linalg.norm(fn), 1e-9)
                shade = np.full(a.shape, abs(fn @ light) * 0.7 + 0.3)
            if tex is not None:
                th, tw = tex.shape[:2]
                u = a * uv[t[0], 0] + b * uv[t[1], 0] + c * uv[t[2], 0]
                vv = a * uv[t[0], 1] + b * uv[t[1], 1] + c * uv[t[2], 1]
                u = np.mod(u, 1.0); vv = np.mod(vv, 1.0)
                if flip_v:
                    vv = 1 - vv
                col = tex[np.clip((vv * th).astype(int), 0, th - 1), np.clip((u * tw).astype(int), 0, tw - 1), :3].astype(np.float32)
            elif vcol is not None:
                col = a[:, None] * vcol[t[0]] + b[:, None] * vcol[t[1]] + c[:, None] * vcol[t[2]]
            else:
                hue = (mi * 0.37) % 1.0
                col = np.tile(np.array(_hsv(hue, 0.25, 0.9), np.float32) * 255, (len(a), 1))
            zb[mask] = z[mask]
            img[miny:maxy + 1, minx:maxx + 1][mask] = col * shade[:, None]
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

"""
tfp_core.py - Transformers Prime: The Game (Wii U) data decoding.

Archives (content/*.dat): files aligned to 0x800 bytes. Each file is
  'CPNB'/'CPWB'/'CPDB' header (u8 version @4, u32 BE size @12) + data
    v0 = stored, v1 = LZSS (flag bit 0 -> 8-bit literal, 1 -> len 4b+3, dist 12b)
    v2 = per-block static Huffman of 9-bit symbols + LZ (dist 12b, len = sym-0xFD)
    Bits are taken LSB-first from big-endian 32-bit words.
  'NIL ' = empty slot, or an uncompressed headerless package.
Packages: u32 0, u32 count/1, u32 size, u32 type.  UTX = GX2 texture set, GEO = model.

GX2 tiling math ported from AboodXD's addrlib (GTX-Extractor, GPL-3.0); this file is GPL-3.0.
"""
import struct
import numpy as np
try:
    import tfp_fast as _fast        # optional compiled kernels (needs numba)
except Exception:
    _fast = None

SECTOR = 0x800
GEO_MAGIC = b'GEO\0WIIU'
UTX_MAGIC = b'UTX\0\x01\0\0\0'

# ------------------------------------------------------------------ decompression
def parse_cp_header(h):
    if len(h) < 16 or h[0:2] != b'CP' or h[2:3] not in (b'N', b'W', b'D') or h[3:4] not in (b'B', b'L'):
        return None
    ver = h[4]
    if ver > 2 or any(h[5:12]):
        return None
    size = struct.unpack_from('>I', h, 12)[0]
    return {'type': chr(h[2]), 'ver': 0 if h[2:3] == b'N' else ver, 'size': size}


def _words(src, start):
    n = (len(src) - start) // 4
    w = np.frombuffer(src, dtype='>u4', count=n, offset=start).tolist()
    return w


def decompress_cp(src, limit=None):
    """src: bytes beginning at the CP header. Returns bytes (up to limit)."""
    hd = parse_cp_header(src[:16])
    if hd is None:
        raise ValueError('not a CP file')
    size = hd['size'] if limit is None else min(limit, hd['size'])
    if hd['ver'] == 0:
        return bytes(src[16:16 + size])
    if _fast is not None and _fast.HAVE_NUMBA and hd['ver'] in (1, 2):
        nb = (len(src) - 16) // 4 * 4
        Wn = np.concatenate([np.frombuffer(bytes(src[16:16 + nb]), dtype='>u4').astype(np.uint32),
                             np.zeros(2, np.uint32)])
        return _fast.decompress(Wn, hd['ver'], size)
    W = _words(src, 16)
    W.append(0); W.append(0)
    out = bytearray(size)
    wi = 0; buf = 0; n = 0; o = 0
    if hd['ver'] == 1:
        while o < size:
            if n < 17:
                buf |= W[wi] << n; wi += 1; n += 32
            flag = buf & 1; buf >>= 1; n -= 1
            if flag == 0:
                out[o] = buf & 0xff; buf >>= 8; n -= 8; o += 1
            else:
                l = (buf & 15) + 3; d = (buf >> 4) & 0xfff; buf >>= 16; n -= 16
                s = o - d
                if l > size - o:
                    l = size - o
                if d >= l:
                    out[o:o + l] = out[s:s + l]
                else:
                    for k in range(l):
                        out[o + k] = out[s + k]
                o += l
        return bytes(out)
    # version 2
    def getbit():
        nonlocal buf, n, wi
        if n == 0:
            buf = W[wi]; wi += 1; n = 32
        b = buf & 1; buf >>= 1; n -= 1
        return b

    def getbits(k):
        nonlocal buf, n, wi
        if n < k:
            buf |= W[wi] << n; wi += 1; n += 32
        v = buf & ((1 << k) - 1); buf >>= k; n -= k
        return v

    while o < size:
        cnt = getbits(16)
        L = []; R = []; V = []

        def tree():
            i = len(V); V.append(0); L.append(0); R.append(0)
            if getbit():
                V[i] = -1; a = tree(); b = tree(); L[i] = a; R[i] = b
            else:
                V[i] = getbits(9)
            return i
        tree()
        while cnt > 0 and o < size:
            nd = 0
            while V[nd] < 0:
                nd = L[nd] if getbit() else R[nd]
            s = V[nd]; cnt -= 1
            if s < 256:
                out[o] = s; o += 1
            else:
                d = getbits(12); l = min(s - 0xfd, size - o); cnt -= 1
                st = o - d
                if d >= l:
                    out[o:o + l] = out[st:st + l]
                else:
                    for k in range(l):
                        out[o + k] = out[st + k]
                o += l
    return bytes(out)


# ------------------------------------------------------------------ archives
def _raw_package_size(d, o):
    """If d[o:] looks like a headerless package, return its size (0 if unknown), else None."""
    if o + 0x40 > len(d):
        return None
    z, n, h, t = struct.unpack_from('>4I', d, o)
    if z != 0 or not (1 <= n <= 4096) or t > 1:
        return None
    if t == 1:
        p20 = struct.unpack_from('>I', d, o + 0x20)[0]
        if n > 64 or h < 0x100 or h & 3 or not (0x20 <= p20 < min(h, 0x10000)) or p20 & 0x1f:
            return None
        return h + 8
    if h != 16 + 20 * n:
        return None
    end = 0
    for i in range(n):
        off, sz = struct.unpack_from('>2I', d, o + 16 + 12 * i)
        end = max(end, off + sz)
    return end


def _looks_raw_package(d, o):
    return _raw_package_size(d, o) is not None


class Entry:
    __slots__ = ('dat', 'off', 'kind', 'size', 'span', 'cls', 'index', 'textures', 'nparts', 'pkgcount',
                 'part', 'tset', 'attach')

    def __init__(self, dat, off, kind, size):
        self.dat = dat; self.off = off; self.kind = kind; self.size = size
        self.span = 0; self.cls = '?'; self.index = 0; self.textures = None; self.nparts = 1; self.pkgcount = 0
        self.part = -1; self.tset = -1; self.attach = ()

    def clone(self, **kw):
        c = Entry(self.dat, self.off, self.kind, self.size)
        for k in self.__slots__:
            setattr(c, k, getattr(self, k))
        for k, v in kw.items():
            setattr(c, k, v)
        return c

    @property
    def key(self):
        return (self.dat, self.off, self.part, self.tset)

    @property
    def name(self):
        import os
        n = '%s_%04d_%08x' % (os.path.splitext(os.path.basename(self.dat))[0], self.index, self.off)
        if self.part >= 0:
            n += '_part%d' % self.part
        elif self.part == -2:
            n += '_assembled'
        if self.tset >= 0:
            n += '_set%d' % self.tset
        return n


def scan_dat(path, progress=None):
    import os
    fsize = os.path.getsize(path)
    starts = []
    skip_until = 0
    CH = 32 << 20
    with open(path, 'rb') as f:
        base = 0
        while base < fsize:
            d = f.read(CH)
            o = 0
            while o < len(d) - 15:
                if base + o < skip_until:
                    o += 0x40; continue
                h = d[o:o + 16]
                hd = parse_cp_header(h)
                if hd:
                    starts.append((base + o, 'CP%s%d' % (hd['type'], hd['ver']), hd['size']))
                elif h[:4] == b'NIL ' and o % SECTOR == 0:
                    starts.append((base + o, 'NIL', 0))
                else:
                    sz = _raw_package_size(d, o)
                    if sz is not None:
                        starts.append((base + o, 'RAW', 0))
                        if sz:
                            skip_until = base + o + sz
                o += 0x40
            base += len(d)
            if progress:
                progress(min(1.0, base / fsize))
    ents = []
    for i, (off, kind, size) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else fsize
        if kind == 'NIL':
            continue
        e = Entry(path, off, kind, size)
        e.span = end - off
        if kind == 'RAW':
            e.size = e.span
        ents.append(e)
    for i, e in enumerate(ents):
        e.index = i
    return ents


# ------------------------------------------------------------------ file table from the executable
def read_rpx_file_table(rpx_path):
    """Returns {dat name (lower): [(offset, stored size, compressed?, uncompressed size)]} from appwiiu.rpx."""
    import zlib
    d = open(rpx_path, 'rb').read()
    if d[:4] != b'\x7fELF':
        return {}
    shoff = struct.unpack_from('>I', d, 0x20)[0]
    shentsize, shnum = struct.unpack_from('>HH', d, 0x2e)
    secs = []
    for i in range(shnum):
        name, typ, flags, addr, off, size = struct.unpack_from('>6I', d, shoff + i * shentsize)
        if typ == 8 or not size:
            continue
        data = d[off:off + size]
        if flags & 0x08000000:
            try:
                data = zlib.decompress(data[4:])
            except Exception:
                continue
        secs.append((addr, data))
    ro = [(a, x) for a, x in secs if a == 0x10000000]
    if not ro:
        return {}
    base, rod = ro[0]

    def cstr(a):
        o = a - base
        if not 0 <= o < len(rod):
            return None
        e = rod.find(b'\0', o)
        t = rod[o:e]
        return t.decode('latin1') if 0 < len(t) <= 12 else None
    best = []
    for align in range(0, 16, 4):
        o = align
        run = []
        while o + 16 <= len(rod):
            a, fl, off, cs = struct.unpack_from('>4I', rod, o)
            n = cstr(a) if (a & 0xfff00000) == base else None
            if n and n.lower().endswith('.dat'):
                run.append((n.lower(), fl, off, cs)); o += 16
            else:
                if len(run) > len(best):
                    best = run
                run = []; o += 16
        if len(run) > len(best):
            best = run
    table = {}
    for n, fl, off, cs in best:
        table.setdefault(n, []).append((off, cs, bool(fl & 0x80000000), fl & 0x1fffffff))
    return table


def entries_from_table(path, rows):
    import os
    fsize = os.path.getsize(path)
    ents = []
    with open(path, 'rb') as f:
        for off, cs, comp, us in sorted(rows):
            if off + cs > fsize:
                continue
            f.seek(off); h = f.read(16)
            hd = parse_cp_header(h)
            if hd:
                e = Entry(path, off, 'CP%s%d' % (hd['type'], hd['ver']), hd['size'])
            else:
                e = Entry(path, off, 'RAW', cs)
            e.span = cs
            ents.append(e)
    for i, e in enumerate(ents):
        e.index = i
    return ents


def load_entry(e, limit=None):
    with open(e.dat, 'rb') as f:
        f.seek(e.off)
        if e.kind == 'RAW':
            return f.read(e.span if limit is None else min(limit, e.span))
        if limit is None:
            return decompress_cp(f.read(e.span))
        raw = f.read(min(e.span, 16 + max(8192, limit * 3)))
    try:
        return decompress_cp(raw, limit)
    except IndexError:
        with open(e.dat, 'rb') as f:
            f.seek(e.off)
            return decompress_cp(f.read(e.span), limit)


def classify(head):
    lim = head[:0x800]
    if GEO_MAGIC in lim:
        return 'model'
    if UTX_MAGIC in lim:
        return 'textures'
    if len(head) >= 16 and struct.unpack_from('>4I', head) == (0, 1, 0x1c, 0):
        return 'empty'
    if head[:3] == b'TXT':
        return 'text'
    if head[:2] == b'<?':
        return 'xml'
    return 'other'


def index_game(paths, progress=None, table=None, part_cache=None):
    """Index .dat files (using the executable's file table when given), classify entries,
    and pair each model package with its texture package."""
    import os
    allents = []
    for pi, p in enumerate(paths):
        rows = (table or {}).get(os.path.basename(p).lower())
        if rows:
            ents = entries_from_table(p, rows)
        else:
            ents = scan_dat(p, (lambda f, pi=pi: progress((pi + f * 0.5) / len(paths))) if progress else None)
        for k, e in enumerate(ents):
            try:
                head = load_entry(e, 0x800)
                e.cls = classify(head)
                if len(head) >= 16:
                    z, n, h, t = struct.unpack_from('>4I', head)
                    if z == 0 and 1 <= n <= 4096:
                        e.pkgcount = n
                        if t == 1 and e.cls == 'model':
                            e.nparts = n
            except Exception:
                e.cls = 'error'
            if progress and k % 50 == 0:
                progress((pi + 0.5 + 0.5 * k / max(1, len(ents))) / len(paths))
        for i, e in enumerate(ents):
            if e.cls != 'model':
                continue
            nxt = ents[i + 1] if i + 1 < len(ents) else None
            prv = ents[i - 1] if i > 0 else None
            if nxt is not None and nxt.cls == 'textures' and nxt.pkgcount == e.nparts and e.nparts >= 1 and e.kind == 'RAW':
                e.textures = nxt            # d02+ layout: model package, then texture package
            elif prv is not None and prv.cls in ('textures', 'empty'):
                e.textures = prv            # d01 layout: texture file, then model file
            elif nxt is not None and nxt.cls == 'textures':
                e.textures = nxt
        for e in ents:
            if e.cls == 'model' and e.nparts > 1:
                # the header counts slots, not models: some slots hold other data
                ck = '%s|%d|%x' % (os.path.basename(e.dat).lower(), os.path.getsize(e.dat), e.off)
                if part_cache is not None and ck in part_cache:
                    e.nparts = part_cache[ck]
                else:
                    try:
                        e.nparts = max(1, len(geo_parts(load_entry(e))))
                        if part_cache is not None:
                            part_cache[ck] = e.nparts
                    except Exception:
                        pass
        for i, e in enumerate(ents):
            if e.cls == 'model' and e.nparts > 1:
                # one model package holds several GEO models (body, weapons, alt modes...)
                # attachments (weapons etc.): the single models that follow, up to the next multi-part package
                att = []
                for x in ents[i + 1:]:
                    if x.cls == 'model' and x.nparts > 1:
                        break
                    if x.cls == 'model' and x.size != 0xc48:
                        att.append(x)
                allents.append(e.clone(part=-2, attach=tuple(att)))     # all parts put together as in-game
                allents.extend(e.clone(part=k) for k in range(e.nparts))
            else:
                allents.append(e)
    return allents


def geo_parts(d):
    """Split a decompressed model package into its GEO blocks: list of (start, end)."""
    pos = []
    i = d.find(GEO_MAGIC)
    while i >= 0:
        pos.append(i)
        i = d.find(GEO_MAGIC, i + 8)
    return [(p, pos[k + 1] if k + 1 < len(pos) else len(d)) for k, p in enumerate(pos)]


def part_texture_sets(d, nparts):
    """For a raw texture package with one entry per model part: index (into texture_sets(d))
    of the set each part uses. Empty entries reuse the previous entry's set. None if not applicable."""
    if len(d) < 16:
        return None
    z, n, h, t = struct.unpack_from('>4I', d, 0)
    if z != 0 or t != 0 or n != nparts or not (16 + 12 * n <= h <= len(d)):
        return None
    starts = []
    i = d.find(UTX_MAGIC)
    while i >= 0:
        if parse_utx(d, i):
            starts.append(i)
        i = d.find(UTX_MAGIC, i + 8)
    out, cur = [], 0
    for k in range(n):
        off, size = struct.unpack_from('>2I', d, 16 + 12 * k)[:2]
        if size:
            inside = [j for j, p in enumerate(starts) if off <= p < off + size]
            if inside:
                cur = inside[0]
        out.append(cur)
    return out


def texture_sets(d):
    """All UTX texture sets inside a decompressed file."""
    out = []
    i = d.find(UTX_MAGIC)
    while i >= 0:
        ts = parse_utx(d, i)
        if ts:
            out.append(ts)
        i = d.find(UTX_MAGIC, i + 8)
    return out


def strip_variants(model, hidden=None, keep=(), vmap=None):
    """Copy of model without variant mesh groups, except those in keep. A kept variant replaces its base group
    (vmap: {variant: base}, computed if not given)."""
    if vmap is None:
        vmap = variant_map(model) if hidden is None else {}
    if hidden is None:
        hidden = set(vmap)
    keep = set(keep)
    hidden = (set(hidden) - keep) | {vmap[g] for g in keep if g in vmap}
    if not hidden:
        return model
    m = dict(model)
    m['meshes'] = [s for s in model['meshes'] if s['mesh'] not in hidden]
    m['hidden_groups'] = sorted(hidden)
    return m


def variant_map(model, tol=0.04):
    """{variant group: base group} for mesh groups that duplicate an earlier group (alternate versions of the
    same piece, e.g. Knock Out's faces/bodies). A variant replaces its base when shown."""
    groups = {}
    for s in model['meshes']:
        groups.setdefault(s['mesh'], []).append(s)
    info = []
    for g, subs in sorted(groups.items()):
        P = np.concatenate([s['position'][:, :3] for s in subs if 'position' in s] or [np.zeros((1, 3))])
        info.append((g, sum(s['vcount'] for s in subs), P.min(0), P.max(0)))
    out = {}
    for i, (g, vc, lo, hi) in enumerate(info):
        size = max(float(np.linalg.norm(hi - lo)), 1e-6)
        for g0, vc0, lo0, hi0 in info[:i]:
            if g0 in out:
                continue
            if abs(vc - vc0) <= tol * max(vc, vc0) and vc > 50 and \
                    np.linalg.norm(lo - lo0) < 0.05 * size and np.linalg.norm(hi - hi0) < 0.05 * size:
                out[g] = g0
                break
    return out


def variant_groups(model, tol=0.04):
    """Mesh groups that duplicate an earlier group (see variant_map)."""
    return set(variant_map(model, tol))


# ------------------------------------------------------------------ UTX textures
FMT_NAMES = {0x31: 'BC1', 0x32: 'BC2', 0x33: 'BC3', 0x34: 'BC4', 0x35: 'BC5', 0x1a: 'RGBA8',
             0x08: 'RGB565', 0x01: 'R8', 0x07: 'RG8'}
BLOCK_BYTES = {0x31: 8, 0x32: 16, 0x33: 16, 0x34: 8, 0x35: 16}


def fmt_name(f):
    return FMT_NAMES.get(f & 0x3f, 'fmt%x' % f) + ('_sRGB' if f & 0x400 else '') + ('_SNORM' if f & 0x200 else '')


def parse_utx(d, at):
    if d[at:at + 8] != UTX_MAGIC:
        return []
    cnt, hs = struct.unpack_from('>2I', d, at + 8)
    if not 0 < cnt < 1000:
        return []
    offs = [hs] + list(struct.unpack_from('>%dI' % (cnt - 1), d, at + 16)) if cnt > 1 else [hs]
    texs = []
    for i, o in enumerate(offs):
        w = struct.unpack_from('>17I', d, at + o)
        texs.append(dict(index=i, width=w[0], height=w[1], depth=w[2], mips=w[4], format=w[5],
                         tileMode=w[6], swizzle=w[7], pitch=w[8], imageSize=w[9], compSel=w[16],
                         dataOff=at + o + 0x44, data=d))
    return texs


def find_textures(d):
    i = d.find(UTX_MAGIC, 0, 0x1000)
    return parse_utx(d, i) if i >= 0 else []


def _pixel_index(x, y, bpp):
    if bpp == 8:
        b = [x & 1, (x & 2) >> 1, (x & 4) >> 2, (y & 2) >> 1, y & 1, (y & 4) >> 2]
    elif bpp == 16:
        b = [x & 1, (x & 2) >> 1, (x & 4) >> 2, y & 1, (y & 2) >> 1, (y & 4) >> 2]
    elif bpp == 64:
        b = [x & 1, y & 1, (x & 2) >> 1, (x & 4) >> 2, (y & 2) >> 1, (y & 4) >> 2]
    elif bpp == 128:
        b = [y & 1, x & 1, (x & 2) >> 1, (x & 4) >> 2, (y & 2) >> 1, (y & 4) >> 2]
    else:
        b = [x & 1, (x & 2) >> 1, y & 1, (x & 4) >> 2, (y & 2) >> 1, (y & 4) >> 2]
    return b[0] | 2 * b[1] | 4 * b[2] | 8 * b[3] | 16 * b[4] | 32 * b[5]


def _addresses(w, h, bpe, pitch, tile_mode, swizzle):
    """Vectorised GX2 address computation; returns byte address per element (h, w)."""
    y, x = np.mgrid[0:h, 0:w].astype(np.int64)
    bpp = bpe * 8
    if tile_mode in (0, 1, 16):
        return (y * pitch + x) * bpe
    pix = _pixel_index(x, y, bpp)
    if tile_mode in (2, 3):
        mtb = (64 * bpp + 7) >> 3
        return ((bpp * pix) >> 3) + mtb * ((x >> 3) + (y >> 3) * (pitch >> 3))
    elem = (bpp * pix + 7) >> 3
    pipe = ((y >> 3) ^ (x >> 3)) & 1
    bank = (((y >> 5) ^ (x >> 3)) & 1) | (2 * (((y >> 4) ^ (x >> 4)) & 1))
    pipe_sw = (swizzle >> 8) & 1
    bank_sw = (swizzle >> 9) & 3
    bp = (pipe + 2 * bank) ^ (pipe_sw + 2 * bank_sw)
    bp %= 8
    pipe = bp % 2; bank = bp // 2
    mtp, mth = 32, 16
    if tile_mode in (5, 9):
        mtp, mth = 16, 32
    elif tile_mode in (6, 10):
        mtp, mth = 8, 64
    per_row = pitch // mtp
    mt_bytes = (bpp * mth * mtp + 7) >> 3
    mt_off = ((x // mtp) + per_row * (y // mth)) * mt_bytes
    total = elem + (mt_off >> 3)
    return (bank << 9) | (pipe << 8) | (total & 255) | ((total & ~255) << 3)


def _bytes_per_element(fmt):
    f = fmt & 0x3f
    return BLOCK_BYTES.get(f) or {0x1a: 4, 0x19: 4, 0x08: 2, 0x0a: 2, 0x0b: 2, 0x07: 2, 0x01: 1, 0x02: 1}.get(f, 4)


def _deswizzle(t):
    f = t['format'] & 0x3f
    bc = f in BLOCK_BYTES
    w = (t['width'] + 3) >> 2 if bc else t['width']
    h = (t['height'] + 3) >> 2 if bc else t['height']
    bpe = _bytes_per_element(t['format'])
    src = np.frombuffer(t['data'], dtype=np.uint8, count=min(t['imageSize'], len(t['data']) - t['dataOff']),
                        offset=t['dataOff'])
    addr = _addresses(w, h, bpe, t['pitch'], t['tileMode'], t['swizzle']).reshape(-1)
    idx = addr[:, None] + np.arange(bpe)[None, :]
    idx = np.clip(idx, 0, len(src) - 1)
    return src[idx].reshape(h, w, bpe), w, h


def _unpack565(c):
    r = ((c >> 11) & 31) * 255 // 31
    g = ((c >> 5) & 63) * 255 // 63
    b = (c & 31) * 255 // 31
    return np.stack([r, g, b], -1).astype(np.int32)


def _bc_color(blk, opaque):
    """blk: (h,w,8) uint8 -> (h,w,16,4) RGBA"""
    if _fast is not None:
        r = _fast.bc_color(blk, opaque)
        if r is not None:
            return r
    c0 = blk[..., 0].astype(np.int32) | (blk[..., 1].astype(np.int32) << 8)
    c1 = blk[..., 2].astype(np.int32) | (blk[..., 3].astype(np.int32) << 8)
    p0 = _unpack565(c0); p1 = _unpack565(c1)
    four = (c0 > c1) | opaque
    p2 = np.where(four[..., None], (2 * p0 + p1) // 3, (p0 + p1) // 2)
    p3 = np.where(four[..., None], (p0 + 2 * p1) // 3, 0)
    a = np.full(c0.shape, 255, np.int32)
    pal = np.stack([np.concatenate([p, a[..., None]], -1) for p in (p0, p1, p2)] +
                   [np.concatenate([p3, np.where(four, 255, 0)[..., None]], -1)], -2)  # h,w,4,4
    bits = (blk[..., 4].astype(np.uint32) | (blk[..., 5].astype(np.uint32) << 8) |
            (blk[..., 6].astype(np.uint32) << 16) | (blk[..., 7].astype(np.uint32) << 24))
    sh = np.arange(16, dtype=np.uint32) * 2
    ix = ((bits[..., None] >> sh) & 3).astype(np.int64)  # h,w,16
    out = np.take_along_axis(pal, ix[..., None].repeat(4, -1), axis=-2)  # h,w,16,4
    return out


def _bc_alpha(blk, signed=False):
    if _fast is not None:
        r = _fast.bc_alpha(blk, signed)
        if r is not None:
            return r
    a0 = blk[..., 0].astype(np.int32); a1 = blk[..., 1].astype(np.int32)
    if signed:
        a0 = ((a0 + 128) & 255); a1 = ((a1 + 128) & 255)
    k = np.arange(1, 7)
    pal8 = [a0, a1] + [((7 - i) * a0 + i * a1) // 7 for i in range(1, 7)]
    pal6 = [a0, a1] + [((5 - i) * a0 + i * a1) // 5 for i in range(1, 5)] + [np.zeros_like(a0), np.full_like(a0, 255)]
    pal = np.where((a0 > a1)[..., None], np.stack(pal8, -1), np.stack(pal6, -1))
    bits = np.zeros(a0.shape, np.uint64)
    for i in range(6):
        bits |= blk[..., 2 + i].astype(np.uint64) << np.uint64(8 * i)
    sh = (np.arange(16, dtype=np.uint64) * np.uint64(3))
    ix = ((bits[..., None] >> sh) & np.uint64(7)).astype(np.int64)
    return np.take_along_axis(pal, ix, axis=-1)  # h,w,16


def decode_texture(t):
    """Returns HxWx4 uint8 RGBA (base mip level)."""
    blocks, bw, bh = _deswizzle(t)
    f = t['format'] & 0x3f
    W, H = t['width'], t['height']
    if f in BLOCK_BYTES:
        signed = bool(t['format'] & 0x200)
        if f == 0x31:
            px = _bc_color(blocks, False)
        elif f == 0x32:
            px = _bc_color(blocks[..., 8:], True)
            ab = blocks[..., :8].astype(np.int32)
            a = np.stack([(ab[..., i >> 1] >> ((i & 1) * 4)) & 15 for i in range(16)], -1) * 17
            px[..., 3] = a
        elif f == 0x33:
            px = _bc_color(blocks[..., 8:], True)
            px[..., 3] = _bc_alpha(blocks[..., :8])
        elif f == 0x34:
            r = _bc_alpha(blocks, signed)
            px = np.stack([r, r, r, np.full_like(r, 255)], -1)
        elif f == 0x35:
            r = _bc_alpha(blocks[..., :8], signed).astype(np.float32)
            g = _bc_alpha(blocks[..., 8:], signed).astype(np.float32)
            nx = r / 127.5 - 1; ny = g / 127.5 - 1
            b = (np.sqrt(np.clip(1 - nx * nx - ny * ny, 0, 1)) * 0.5 + 0.5) * 255
            px = np.stack([r, g, b, np.full_like(r, 255)], -1)
        img = px.reshape(bh, bw, 4, 4, 4).transpose(0, 2, 1, 3, 4).reshape(bh * 4, bw * 4, 4)
        return np.ascontiguousarray(img[:H, :W].astype(np.uint8))
    b = blocks.astype(np.int32)
    if f in (0x1a, 0x19):
        img = b[..., :4]
    elif f == 0x08:
        c = (b[..., 0] << 8) | b[..., 1]
        img = np.concatenate([_unpack565(c), np.full(c.shape + (1,), 255)], -1)
    elif f == 0x07:
        img = np.stack([b[..., 0], b[..., 1], np.zeros_like(b[..., 0]), np.full_like(b[..., 0], 255)], -1)
    else:
        img = np.stack([b[..., 0]] * 3 + [np.full_like(b[..., 0], 255)], -1)
    return np.ascontiguousarray(img[:H, :W].astype(np.uint8))


# ------------------------------------------------------------------ GEO models
def _u32(d, o):
    return struct.unpack_from('>I', d, o)[0]


def _read_list(d, p):
    if p < 0 or p + 8 > len(d):
        return None
    n, hs = struct.unpack_from('>2I', d, p)
    if not (1 <= n <= 4096) or hs != 8 + 4 * (n - 1):
        return None
    return [p + hs] + [p + _u32(d, p + 8 + 4 * (i - 1)) for i in range(1, n)]


def _find_list(d, p):
    for k in (0, 4, -4, 8):
        r = _read_list(d, p + k)
        if r:
            return r
    return None


SEM = {0: 'position', 1: 'weights', 2: 'joints', 3: 'normal', 4: 'tangent', 5: 'binormal', 6: 'uv', 7: 'color'}


def parse_geo(d):
    g = d.find(GEO_MAGIC)
    if g < 0:
        raise ValueError('no GEO block')
    version = _u32(d, g + 8)
    ptrA, ptrB = _u32(d, g + 0x14), _u32(d, g + 0x18)
    base = mesh_items = None
    for b in (g + 0x14, g + 0x10, g + 0x18, g + 0xc):
        m = _read_list(d, b + ptrB)
        if m:
            base, mesh_items = b, m
            break
    if mesh_items is None:
        for b in (g + 0x14, g + 0x10):
            m = _find_list(d, b + ptrB)
            if m:
                base, mesh_items = b, m
                break
    if mesh_items is None:
        raise ValueError('mesh list not found')
    mats = []
    for i, m in enumerate(_find_list(d, base + ptrA) or []):
        color = struct.unpack_from('>4f', d, m)
        nl = struct.unpack_from('>H', d, m + 0x4a)[0]
        layers = []
        sig = bytes(d[m:m + 0x5c])           # raw record (incl. the material id at +0x48), texture slots excluded
        if 0 < nl < 16:
            for k in range(nl):
                lp = m + 0x5c + 0x2c * k
                layers.append(dict(texture=d[lp], kind=d[lp + 1], uscale=struct.unpack_from('>f', d, lp + 8)[0],
                                   vscale=struct.unpack_from('>f', d, lp + 12)[0]))
                sig += bytes(d[lp + 1:lp + 0x1c])
        mats.append(dict(index=i, color=color, layers=layers, id=struct.unpack_from('>H', d, m + 0x48)[0], sig=sig))
    meshes, errors = [], []
    for mi, mp in enumerate(mesh_items):
        bbox = struct.unpack_from('>6f', d, mp)
        subs = _read_list(d, mp + 0x1c) or _find_list(d, mp + 0x18)
        if not subs:
            errors.append('mesh %d: no submesh list' % mi)
            continue
        for si, s in enumerate(subs):
            try:
                meshes.append(_parse_submesh(d, s, mi, si))
            except Exception as ex:
                errors.append('mesh %d sub %d: %s' % (mi, si, ex))
    return dict(version=version, materials=mats, meshes=meshes, errors=errors)


def _parse_submesh(d, s, mi, si):
    material = struct.unpack_from('>H', d, s)[0]
    pal_count = _u32(d, s + 8)
    if pal_count > 1024:
        raise ValueError('bad palette size')
    palette = list(struct.unpack_from('>%di' % pal_count, d, s + 12)) if pal_count else []
    q = s + 12 + 4 * pal_count
    o_attr, o_vert, o_idx = struct.unpack_from('>3I', d, q)
    n_attr = _u32(d, s + o_attr)
    if not 1 <= n_attr <= 16:
        raise ValueError('bad attribute count %d' % n_attr)
    decl = [_u32(d, s + o_attr + 4 + 4 * k) for k in range(n_attr)]
    vcount = _u32(d, s + o_vert)
    n_w = _u32(d, s + o_vert + 4)
    words = list(struct.unpack_from('>%dI' % n_w, d, s + o_vert + 8))
    stride = sum(words) * 4
    vbase = s + o_vert + 8 + 4 * n_w
    raw = np.frombuffer(d, dtype=np.uint8, count=vcount * stride, offset=vbase).reshape(vcount, stride)
    out = dict(mesh=mi, sub=si, material=material, palette=palette, vcount=vcount)
    off = 0
    seen = set()
    for k, w in enumerate(decl):
        sem = w >> 24
        nw = words[k] if k < len(words) else 0
        name = SEM.get(sem, 'attr%d' % sem)
        if name in seen:
            name += '2'
        seen.add(name)
        col = raw[:, off:off + nw * 4]
        if name in ('position', 'normal', 'uv', 'uv2', 'tangent') and nw in (2, 3, 4):
            out[name] = np.ascontiguousarray(col).view('>f4').astype(np.float32).reshape(vcount, nw)
        elif nw == 1 and name in ('color', 'joints', 'weights', 'color2'):
            out[name] = np.ascontiguousarray(col)
        off += nw * 4
    p = s + o_idx
    if _u32(d, p) >> 24 == 1 and _u32(d, p + 4) < 4096:
        p += 4
    n_strips, hs = struct.unpack_from('>2I', d, p)
    if not 1 <= n_strips <= 4096:
        raise ValueError('index list not found')
    it = p + 8
    tris = []
    for _ in range(n_strips):
        typ, n = struct.unpack_from('>2I', d, it)
        idx = np.frombuffer(d, dtype='>u2', count=n, offset=it + 8).astype(np.int64)
        it = (it + 8 + 2 * n + 3) & ~3
        if typ >> 24 == 1:  # triangle strip
            if n < 3:
                continue
            a, b, c = idx[:-2], idx[1:-1], idx[2:]
            odd = (np.arange(n - 2) & 1).astype(bool)
            t = np.stack([np.where(odd, b, a), np.where(odd, a, b), c], -1)
            keep = (a != b) & (b != c) & (a != c)
            tris.append(t[keep])
        else:
            tris.append(idx[:n - n % 3].reshape(-1, 3))
    tri = np.concatenate(tris) if tris else np.zeros((0, 3), np.int64)
    tri = tri[(tri < vcount).all(1)]
    out['indices'] = np.ascontiguousarray(tri[:, ::-1]).astype(np.uint32)  # game is clockwise; store CCW
    return out


# ------------------------------------------------------------------ assembling multi-part packages
def _bbox(model):
    P = [s['position'][:, :3] for s in model['meshes'] if 'position' in s and len(s['position'])]
    if not P:
        return np.zeros(3), np.zeros(3)
    P = np.concatenate(P)
    return P.min(0), P.max(0)


def is_textured(model):
    return any(m['layers'] for m in model['materials']) and any('uv' in s for s in model['meshes'])


def is_effect(model, limit=60.0):
    """Effect geometry (light streaks, beams): most triangles are long needles. Real meshes have a
    median aspect (longest edge^2 / area) around 5-17; d19 0013's streak part has ~156."""
    asp = []
    for s in model['meshes']:
        P = s['position'][:, :3]; t = s['indices']
        if not len(t):
            continue
        A, B, C = P[t[:, 0]], P[t[:, 1]], P[t[:, 2]]
        e = np.max(np.stack([np.linalg.norm(B - A, axis=1), np.linalg.norm(C - B, axis=1),
                             np.linalg.norm(A - C, axis=1)], 1), 1)
        asp.append(e ** 2 / np.maximum(np.linalg.norm(np.cross(B - A, C - A), axis=1) / 2, 1e-12))
    return bool(asp) and float(np.median(np.concatenate(asp))) > limit


def default_assembly(models):
    """Which parts an assembled package shows by default: every textured part, effect geometry included (it is
    under Visibility > Effects). Untextured proxies (the Proxy toggle) start off. If nothing is textured, part 0
    is shown."""
    use = [bool(m['meshes']) and is_textured(m) for m in models]
    if models and not any(use):
        use[0] = True
    return use




def merge_models(models, tex_offsets):
    """Combine several part models into one. tex_offsets[i] is added to part i's texture indices."""
    out = dict(version=models[0]['version'] if models else 0, materials=[], meshes=[], errors=[])
    for i, (m, to) in enumerate(zip(models, tex_offsets)):
        mo = len(out['materials'])
        for mt in m['materials']:
            nm = dict(mt)
            nm['index'] = mo + mt['index'] if isinstance(mt.get('index'), int) else mt.get('index')
            nm['layers'] = [dict(l, texture=l['texture'] + to) for l in mt['layers']]
            out['materials'].append(nm)
        for s in m['meshes']:
            ns = dict(s)
            ns['material'] = s['material'] + mo
            ns['mesh'] = i * 1000 + s['mesh']
            ns['part'] = i
            out['meshes'].append(ns)
        out['errors'] += ['part %d: %s' % (i, er) for er in m['errors']]
    return out


# ------------------------------------------------------------------ node hierarchy, animations, attachment posing
# GEO models carry a node hierarchy (section pointer at GEO+0x1c): each node has parent/child/sibling links,
# a local rotation/translation/scale and a list of mesh groups with an offset transform.
# ANIMWIIU files (the small "other" files after a model) hold clips of per-node TRS tracks.

def quat_mat(q):
    x, y, z, w = q
    n = x*x + y*y + z*z + w*w
    if n < 1e-12: return np.eye(3)
    s = 2.0 / n
    return np.array([[1 - s*(y*y + z*z), s*(x*y - z*w), s*(x*z + y*w)],
                     [s*(x*y + z*w), 1 - s*(x*x + z*z), s*(y*z - x*w)],
                     [s*(x*z - y*w), s*(y*z + x*w), 1 - s*(x*x + y*y)]])

def trs(r, t, s):
    M = np.eye(4); M[:3, :3] = quat_mat(r) * np.array(s)[None, :]; M[:3, 3] = t; return M

def parse_nodes(d, p):
    n = struct.unpack_from('>I', d, p)[0]; p += 4; nodes = []
    for _ in range(n):
        idx, par, child, sib, fl, _z = struct.unpack_from('>6h', d, p); p += 12
        r = struct.unpack_from('>4f', d, p); t = struct.unpack_from('>3f', d, p + 16); s = struct.unpack_from('>3f', d, p + 28); p += 40
        k = struct.unpack_from('>I', d, p)[0]; p += 4; meshes = []
        for _ in range(k):
            mid = struct.unpack_from('>I', d, p)[0]
            mr = struct.unpack_from('>4f', d, p + 4); mt = struct.unpack_from('>3f', d, p + 20); ms = struct.unpack_from('>3f', d, p + 32)
            meshes.append((mid, trs(mr, mt, ms))); p += 44
        nodes.append(dict(idx=idx, parent=par, child=child, sib=sib, flags=fl, local=trs(r, t, s), r=r, t=t, s=s, meshes=meshes))
    return nodes, p

def world(nodes, locals_=None):
    W = {}
    for i, nd in enumerate(nodes):
        L = locals_[i] if locals_ is not None else nd['local']
        W[i] = (W[nd['parent']] @ L) if nd['parent'] >= 0 else L
    return W

def q16(d, p):
    return tuple((v - 32767.5) / 32767.5 for v in struct.unpack_from('>4H', d, p))

def parse_anim(d):
    """ANIMWIIU file (after its 0x40 package header): list of clips {name, duration, tracks{node: track}}."""
    a = d.find(b'ANIMWIIU')
    if a < 0: return []
    base = a + 0x14
    n = struct.unpack_from('>I', d, a + 0x14)[0] if False else struct.unpack_from('>I', d, a + 0x14)[0]
    cnt = struct.unpack_from('>I', d, a + 0x14)[0]
    # header: +0x10 = 1, +0x14 = clip count, then clip offsets relative to base
    cnt = struct.unpack_from('>I', d, a + 0x14)[0]
    offs = struct.unpack_from('>%dI' % cnt, d, a + 0x18)
    clips = []
    ends = [base + o for o in offs[1:]] + [len(d)]
    for ci, o in enumerate(offs):
        c = base + o
        name = d[c:c + 0x40].split(b'\0')[0].decode('latin1')
        nt = struct.unpack_from('>I', d, c + 0x40)[0]
        toffs = list(struct.unpack_from('>%dI' % nt, d, c + 0x44))
        tends = [c + x for x in toffs[1:]] + [ends[ci]]
        tracks, dur = {}, 0.0
        for ti, to in enumerate(toffs):
            p = c + to
            node, _z, nk, _z2 = struct.unpack_from('>4H', d, p)
            dur = max(dur, struct.unpack_from('>f', d, p + 8)[0])
            chans = d[p + 13]; enc = d[p + 12]
            us = bool(enc & 0x20)          # uniform scale: one float instead of three
            hasR, hasT, hasS = bool(chans & 0x20), bool(chans & 0x10), bool(chans & 0x40)
            q = p + 16; keys = []
            for _ in range(nk):
                t = struct.unpack_from('>f', d, q)[0]; q += 4; k = {'t': t}
                if hasR:
                    if enc & 0x10: k['r'] = q16(d, q); q += 8          # 16-bit quantised quaternion
                    else: k['r'] = struct.unpack_from('>4f', d, q); q += 16
                if hasT: k['t3'] = struct.unpack_from('>3f', d, q); q += 12
                if hasS:
                    if us: v = struct.unpack_from('>f', d, q)[0]; k['s'] = (v, v, v); q += 4
                    else: k['s'] = struct.unpack_from('>3f', d, q); q += 12
                keys.append(k)
            const = {}
            if q + 4 <= tends[ti]:
                mask = d[q]; q += 4
                if mask & 1:
                    if enc & 0x10 or not hasR: const['r'] = q16(d, q); q += 8
                    else: const['r'] = struct.unpack_from('>4f', d, q); q += 16
                if mask & 2: const['t3'] = struct.unpack_from('>3f', d, q); q += 12
                if mask & 4:
                    if us: v = struct.unpack_from('>f', d, q)[0]; const['s'] = (v, v, v); q += 4
                    else: const['s'] = struct.unpack_from('>3f', d, q); q += 12
            if q > tends[ti] or (q != tends[ti] and ti < nt - 1):
                raise ValueError('track %d of %s: decoded to %x, expected %x' % (ti, name, q, tends[ti]))
            tracks[node] = (keys, const)
        clips.append(dict(name=name, duration=dur, tracks=tracks))
    return clips

def _lerp_key(keys, name, time):
    ks = [k for k in keys if name in k]
    if not ks: return None
    if time <= ks[0]['t']: return ks[0][name]
    for a, b in zip(ks, ks[1:]):
        if a['t'] <= time <= b['t']:
            f = (time - a['t']) / max(b['t'] - a['t'], 1e-9)
            va, vb = np.array(a[name]), np.array(b[name])
            if name == 'r' and np.dot(va, vb) < 0: vb = -vb
            v = va + (vb - va) * f
            return tuple(v / np.linalg.norm(v)) if name == 'r' else tuple(v)
    return ks[-1][name]

def pose_locals(nodes, clip, time):
    """Local (rotation xyzw, translation, scale) of every node at time in clip (bind pose without a clip)."""
    out = []
    for nd in nodes:
        r, t, s = nd['r'], nd['t'], nd['s']
        tr = clip['tracks'].get(nd['idx']) if clip else None
        if tr:
            keys, const = tr
            r = const.get('r') or _lerp_key(keys, 'r', time) or r
            t = const.get('t3') or _lerp_key(keys, 't3', time) or t
            s = const.get('s') or _lerp_key(keys, 's', time) or s
        out.append((tuple(r), tuple(t), tuple(s)))
    return out


def pose(nodes, clip, time):
    return world(nodes, [trs(r, t, s) for r, t, s in pose_locals(nodes, clip, time)])

def geo_nodes(d):
    """Node hierarchy of a GEO model (section pointer at GEO+0x1c, relative to GEO+0x14)."""
    g = d.find(b'GEO\0WIIU')
    p1 = struct.unpack_from('>I', d, g + 0x1c)[0]
    return parse_nodes(d, g + 0x14 + p1)[0]

def attach_transforms(nodes, clip=None, time=0.0):
    """Per mesh group: 4x4 matrix taking stored vertices to body space, posed by clip at time.
    The anchor (the hand/attach bone) is inferred as the inverse of the most common bind transform:
    pieces stored in body space all share it."""
    import collections
    W0 = world(nodes)
    bind = {mid: W0[i] @ O for i, nd in enumerate(nodes) for mid, O in nd['meshes']}
    if not bind: return {}
    key = lambda M: tuple(np.round(M[:3, :].ravel(), 2))
    top = collections.Counter(key(M) for M in bind.values()).most_common(1)[0][0]
    H = np.linalg.inv(next(M for M in bind.values() if key(M) == top))
    Wp = pose(nodes, clip, time) if clip else W0
    return {mid: H @ Wp[i] @ O for i, nd in enumerate(nodes) for mid, O in nd['meshes']}


def _children(nodes):
    ch = {}
    for i, nd in enumerate(nodes):
        ch.setdefault(nd['parent'], []).append(i)
    return ch


def forearm_nodes(nodes):
    """(right, left) forearm node of a skeleton: the parent of the outermost wrist on each side
    (a wrist is a node that branches into two or more finger chains). Right hand = -X side."""
    W = world(nodes)
    ch = _children(nodes)
    best = {-1: (0.0, None), 1: (0.0, None)}
    for i in range(len(nodes)):
        kids = ch.get(i, [])
        if len(kids) < 2 or nodes[i]['parent'] < 0:
            continue
        x = W[i][0, 3]
        side = -1 if x < 0 else 1
        if abs(x) > best[side][0]:
            best[side] = (abs(x), nodes[i]['parent'])
    return best[-1][1], best[1][1]


def pick_clip(clips):
    """Clip used to pose an attachment in its 'in use' state."""
    for c in clips:
        if 'stay' in c['name'].lower():
            return c, 0.0
    for c in clips:
        if 'trans' in c['name'].lower():
            return c, c['duration']
    return (clips[0], clips[0]['duration']) if clips else (None, 0.0)


def attachment_tag(clips):
    """Mount tag from the clip names, e.g. 'RH' in BOT_25_Garm_RH_Ctrans (RH/LH = right/left hand)."""
    for c in clips:
        for tok in c['name'].upper().split('_'):
            if len(tok) == 2 and tok[0] in 'RL' and tok.isalpha():
                return tok
    return None


def _group_centres(model):
    g = {}
    for s in model['meshes']:
        if 'joints' not in s:
            g.setdefault(s['mesh'], []).append(s['position'][:, :3])
    return {k: np.concatenate(v).mean(0) for k, v in g.items()}


def attachment_anchor(att_nodes, att_model, clips, body_nodes, max_err=0.5):
    """Body bone an attachment hangs off, or None.
    Hands (RH/LH) use the forearm. Otherwise: the bone that best puts the attachment's
    body-space pieces back where they are stored (median centre error below max_err)."""
    if not body_nodes:
        return None
    tag = attachment_tag(clips)
    if tag in ('RH', 'LH'):
        r, l = forearm_nodes(body_nodes)
        k = r if tag == 'RH' else l
        if k is not None:
            return k
    W0 = world(att_nodes)
    cen = _group_centres(att_model)
    pairs = [(cen[mid], W0[i] @ O) for i, nd in enumerate(att_nodes) for mid, O in nd['meshes'] if mid in cen]
    if not pairs:
        return None
    BW = world(body_nodes)
    # 1) the bone that puts the most pieces back exactly where they are stored
    P = np.concatenate([s['position'][:, :3] for s in att_model['meshes']] or [np.zeros((1, 3))])
    tol = 0.01 * max(float(np.linalg.norm(P.max(0) - P.min(0))), 1e-6)
    scored = []
    for k in range(len(body_nodes)):
        d = [np.linalg.norm((BW[k] @ M)[:3, :3] @ c + (BW[k] @ M)[:3, 3] - c) for c, M in pairs]
        scored.append((sum(x < tol for x in d), -float(np.median(d)), -k))
    cnt, _, negk = max(scored)
    if cnt >= 1:
        return -negk
    # 2) otherwise the best approximate fit
    best = (max_err, None)
    for k in range(len(body_nodes)):
        e = float(np.median([np.linalg.norm((BW[k] @ M)[:3, :3] @ c + (BW[k] @ M)[:3, 3] - c) for c, M in pairs]))
        if e < best[0]:
            best = (e, k)
    return best[1]


def _fallback_anchor(att_nodes):
    import collections
    W0 = world(att_nodes)
    bind = {mid: W0[i] @ O for i, nd in enumerate(att_nodes) for mid, O in nd['meshes']}
    if not bind:
        return np.eye(4)
    key = lambda M: tuple(np.round(M[:3, :].ravel(), 2))
    top = collections.Counter(key(M) for M in bind.values()).most_common(1)[0][0]
    return np.linalg.inv(next(M for M in bind.values() if key(M) == top))


def pose_attachment(att_model, att_nodes, clips, H=None):
    """Attachment moved to body space by anchor matrix H (None: stored placement inferred) and posed
    with its 'stay' / 'trans' clip. Node-linked groups are moved rigidly; skinned submeshes are skinned
    to the attachment's own skeleton."""
    clip, time = pick_clip(clips)
    W0 = world(att_nodes)
    Wp = pose(att_nodes, clip, time) if clip else W0
    if H is None:
        H = _fallback_anchor(att_nodes)
    m = transform_model(att_model, {mid: H @ Wp[i] @ O for i, nd in enumerate(att_nodes) for mid, O in nd['meshes']})
    skin = None
    out = []
    for s in m['meshes']:
        if 'joints' not in s or not s.get('palette'):
            out.append(s); continue
        if skin is None:      # zero-scale (hidden) nodes have no inverse: use the pseudo-inverse
            skin = [H @ Wp[j] @ np.linalg.pinv(W0[j]) for j in range(len(att_nodes))]
        pal = np.array([p if 0 <= p < len(skin) else -1 for p in s['palette']] + [-1])
        J = pal[np.minimum(s['joints'].astype(int), len(pal) - 1)]
        w = s['weights'].astype(np.float32)
        w = w / np.maximum(w.sum(1, keepdims=True), 1e-6)
        S = np.stack(skin + [H])                       # index -1 -> anchor only
        P = np.c_[s['position'][:, :3], np.ones(len(s['position']))]
        acc = np.zeros((len(P), 3)); nacc = np.zeros((len(P), 3))
        for k in range(J.shape[1]):
            Mk = S[J[:, k]]
            acc += w[:, k:k + 1] * np.einsum('nij,nj->ni', Mk, P)[:, :3]
            if 'normal' in s:
                nacc += w[:, k:k + 1] * np.einsum('nij,nj->ni', Mk[:, :3, :3], s['normal'][:, :3])
        ns = dict(s)
        ns['position'] = np.concatenate([acc, s['position'][:, 3:]], 1).astype(np.float32)
        if 'normal' in s:
            nacc /= np.maximum(np.linalg.norm(nacc, axis=1, keepdims=True), 1e-9)
            ns['normal'] = np.concatenate([nacc, s['normal'][:, 3:]], 1).astype(np.float32)
        out.append(ns)
    return dict(m, meshes=out)


def transform_model(model, X):
    """Copy of model with mesh group g transformed by X[g] (4x4)."""
    out = []
    for s in model['meshes']:
        M = X.get(s['mesh'])
        if M is None:
            out.append(s); continue
        ns = dict(s)
        P = s['position']
        ns['position'] = np.concatenate([P[:, :3] @ M[:3, :3].T + M[:3, 3], P[:, 3:]], 1).astype(np.float32)
        if 'normal' in s:
            N3 = s['normal'][:, :3] @ M[:3, :3].T
            N3 /= np.maximum(np.linalg.norm(N3, axis=1, keepdims=True), 1e-9)
            ns['normal'] = np.concatenate([N3, s['normal'][:, 3:]], 1).astype(np.float32)
        if np.linalg.det(M[:3, :3]) < 0 and 'indices' in s:     # mirrored: keep the front faces counter-clockwise
            ns['indices'] = np.ascontiguousarray(s['indices'].reshape(-1, 3)[:, ::-1].reshape(-1))
        out.append(ns)
    return dict(model, meshes=out)


def place_by_nodes(model, nodes):
    """Move rigid (non-skinned) mesh groups to their node's bind-pose place. Models store such groups in
    their node's local space (e.g. scene props, opened lids, transformable vehicle pieces)."""
    W0 = world(nodes)
    skinned = {s['mesh'] for s in model['meshes'] if 'joints' in s}
    X = {mid: W0[i] @ O for i, nd in enumerate(nodes) for mid, O in nd['meshes'] if mid not in skinned}
    X = {k: v for k, v in X.items() if not np.allclose(v, np.eye(4), atol=1e-6)}
    return transform_model(model, X) if X else model


def face_flags(meshes, rel_tol=0.002):
    """Per submesh: (cull, overlay) boolean triangle masks.
    cull: always empty now (everything is drawn back-face culled); kept for callers.
    overlay: the triangle lies on top of an earlier coplanar triangle facing the same way (a decal layer)
          -> draw with a small depth bias so it wins consistently. Needs scipy (else nothing is flagged)."""
    cull = [np.zeros(len(s['indices']), bool) for s in meshes]
    over = [np.zeros(len(s['indices']), bool) for s in meshes]
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return cull, over
    Ps, Ts, own, loc = [], [], [], []
    off = 0
    for k, s in enumerate(meshes):
        t = s['indices']
        if not len(t):
            continue
        Ps.append(s['position'][:, :3].astype(np.float64)); Ts.append(t + off); off += len(s['position'])
        own.append(np.full(len(t), k)); loc.append(np.arange(len(t)))
    if not Ts:
        return cull, over
    P = np.concatenate(Ps); t = np.concatenate(Ts); own = np.concatenate(own); loc = np.concatenate(loc)
    A, B, C = P[t[:, 0]], P[t[:, 1]], P[t[:, 2]]
    cen = (A + B + C) / 3
    n = np.cross(B - A, C - A)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)
    r = max(float(np.linalg.norm(cen.max(0) - cen.min(0))) / 2, 1e-9)
    ext = np.max(np.linalg.norm(np.stack([A, B, C], 1) - cen[:, None], axis=2), 1)
    kd = cKDTree(cen)
    import itertools
    lst = kd.query_ball_point(cen, ext + 1e-9, workers=-1)
    lens = np.fromiter((len(x) for x in lst), np.int64, len(lst))
    j = np.fromiter(itertools.chain.from_iterable(lst), np.int64, int(lens.sum()))
    i = np.repeat(np.arange(len(lst)), lens)
    # each pair once: from the lower index's search, or from the higher one's when the lower radius missed it
    dd = np.linalg.norm(cen[i] - cen[j], axis=1)
    keep = (j > i) | ((j < i) & (dd > ext[j] + 1e-9))
    i, j = i[keep], j[keep]
    i, j = np.minimum(i, j), np.maximum(i, j)
    if not len(i):
        return cull, over
    keep = np.linalg.norm(cen[i] - cen[j], axis=1) <= np.maximum(ext[i], ext[j])
    i, j = i[keep], j[keep]
    dots = (n[i] * n[j]).sum(1)
    sep = np.abs(((cen[i] - A[j]) * n[j]).sum(1))
    keep = (dots > 0.9) & (sep < rel_tol * r)          # same-facing, (nearly) coplanar
    i, j = i[keep], j[keep]

    def inside(p, a, b, c):
        v0, v1, v2 = b - a, c - a, p - a
        d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
        d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
        den = d00 * d11 - d01 * d01
        ok = den > 1e-20
        den = np.where(ok, den, 1)
        v = (d11 * d20 - d01 * d21) / den; w = (d00 * d21 - d01 * d20) / den
        return ok & (v > 0.02) & (w > 0.02) & (v + w < 0.98)
    hit = inside(cen[i], A[j], B[j], C[j]) | inside(cen[j], A[i], B[i], C[i])
    for x in j[hit]:                      # later triangle of a same-facing pair is the overlay
        over[own[x]][loc[x]] = True
    return cull, over


def skydome_groups(model, size_ratio=2.0, shell=0.6, min_height=0.08):
    """Mesh groups forming sky domes / sky boxes / horizon backdrops around the rest of the model.
    A group counts when it is a hollow shell (>= `shell` of its vertices are at >= 60% of its max radius),
    isn't flat (height >= min_height x width), surrounds the remaining geometry horizontally (open tops and
    bottoms allowed), rises above it (>= 8% of its height) and is >= size_ratio x as large. Several nested domes are all found. Returns a set of
    mesh group ids (empty if none, or if nothing but a few stray triangles would remain)."""
    pts = {}
    for s in model['meshes']:
        if len(s['position']):
            pts.setdefault(s['mesh'], []).append(s['position'][:, :3])
    if len(pts) < 2:
        return set()
    info = {}
    for g, Ps in pts.items():
        P = np.concatenate(Ps)
        lo, hi = P.min(0), P.max(0)
        d = np.linalg.norm(P - (lo + hi) / 2, axis=1)
        ext = hi - lo
        horiz = max(ext[0], ext[2])
        info[g] = dict(lo=lo, hi=hi, diag=float(np.linalg.norm(ext)),
                       shell=float(np.mean(d >= 0.6 * d.max())) if d.max() > 0 else 0.0,
                       tall=ext[1] >= min_height * horiz and min(ext[0], ext[2]) >= 0.3 * horiz)
    cand = {g for g, i in info.items() if i['shell'] >= shell and i['tall']}
    while cand:
        core = [g for g in info if g not in cand]
        # something real must remain inside it, not just a few stray triangles
        if sum(len(P) for g in core for P in pts[g]) < 24:
            return set()
        # extent of the remaining geometry, ignoring the outermost 1% of its vertices on each side
        # (a few stray pieces, e.g. a backdrop strip, shouldn't decide it)
        C = np.concatenate([np.concatenate(pts[g]) for g in core])
        clo, chi = np.percentile(C, 1, axis=0), np.percentile(C, 99, axis=0)
        cdiag = max(float(np.linalg.norm(chi - clo)), 1e-9)
        keep = set()
        for g in cand:
            i = info[g]
            m = 0.02 * (i['hi'] - i['lo']).max()
            around = all(clo[a] >= i['lo'][a] - m and chi[a] <= i['hi'][a] + m for a in (0, 2))
            # a dome rises well above what it encloses (a body around its own head doesn't)
            above = i['hi'][1] - chi[1] >= 0.08 * (i['hi'][1] - i['lo'][1])
            if around and above and i['diag'] >= size_ratio * cdiag:
                keep.add(g)
        if keep == cand:
            return cand
        cand = keep
    return set()


def strip_groups(model, groups):
    if not groups:
        return model
    return dict(model, meshes=[s for s in model['meshes'] if s['mesh'] not in groups])



def _flags_key(model):
    return tuple(id(s) for s in model['meshes'])


def get_face_flags(model):
    """face_flags() for exactly this set of submeshes, cached on the submeshes (per set: a model shown with
    different variants/parts has different overlaps)."""
    if not model or not model['meshes']:
        return [], []
    key = _flags_key(model)
    if not all(key in s.get('_flags', {}) for s in model['meshes']):
        cull, over = face_flags(model['meshes'])
        for s, c, o in zip(model['meshes'], cull, over):
            s.setdefault('_flags', {})[key] = (c, o)
    return [s['_flags'][key][0] for s in model['meshes']], [s['_flags'][key][1] for s in model['meshes']]


def annotate_face_flags(model):
    """Precompute get_face_flags(model) (e.g. off the UI thread)."""
    get_face_flags(model)
    return model


# ------------------------------------------------------------------ anchor refinements
def _skin_nonarm_points(body_model, body_nodes, upper):
    """Body vertices skinned mostly to bones outside the arm subtree that starts at bone 'upper'."""
    ch = _children(body_nodes)
    arm, st = set(), [upper]
    while st:
        x = st.pop(); arm.add(x); st += ch.get(x, [])
    out = []
    for s in body_model['meshes']:
        if 'joints' not in s or not s.get('palette'):
            continue
        pal = np.array(list(s['palette']) + [-1])
        J = pal[np.minimum(s['joints'].astype(int), len(pal) - 1)]
        dom = J[np.arange(len(J)), np.argmax(s['weights'], 1)]
        out.append(s['position'][~np.isin(dom, list(arm)), :3])
    return np.concatenate(out) if out else np.zeros((0, 3))


def hand_or_forearm(att_model, att_nodes, clips, body_nodes, body_model, forearm):
    """Hand weapons normally mount on the forearm bone (Knock Out's fit it exactly). Some are built for the
    hand bone instead (e.g. d12's guns): if mounting on the forearm buries part of the weapon in the torso or
    legs while the hand doesn't, the hand is used."""
    kids = [i for i, nd in enumerate(body_nodes) if nd['parent'] == forearm]
    if not kids or body_model is None:
        return forearm
    hand = kids[0]
    B = _skin_nonarm_points(body_model, body_nodes, body_nodes[forearm]['parent'])
    if not len(B):
        return forearm
    try:
        from scipy.spatial import cKDTree
        near = lambda P, r: cKDTree(B).query(P)[0] < r
    except ImportError:
        Bs = B[:: max(1, len(B) // 4000)]
        near = lambda P, r: np.array([np.min(np.linalg.norm(Bs - p, axis=1)) < r for p in P[:: max(1, len(P) // 2000)]])
    r = 0.03 * float(B[:, 1].max() - B[:, 1].min())
    BW = world(body_nodes)
    frac = {}
    for b in (forearm, hand):
        pm = pose_attachment(att_model, att_nodes, clips, BW[b])
        P = np.concatenate([s['position'][:, :3] for s in pm['meshes']])
        frac[b] = float(np.mean(near(P, r)))
    return hand if frac[forearm] > 0.03 and frac[hand] < 0.01 else forearm


def vehicle_anchor(att_model, att_nodes, clips, body_nodes, body_bbox):
    """Anchor for an attachment with no mount tag and no matching pieces (vehicle weapons): the body's origin
    if that places it sensibly (centred left-right, not below the ground, over the body's length), otherwise the
    first bone, nearest the root first, that does (e.g. the chassis bone of Optimus' truck). None = origin."""
    lo, hi = body_bbox
    W, Hh = float(hi[0] - lo[0]), float(hi[1] - lo[1])
    BW = world(body_nodes)

    def ok(H):
        pm = pose_attachment(att_model, att_nodes, clips, H)
        P = np.concatenate([s['position'][:, :3] for s in pm['meshes']])
        c = (P.min(0) + P.max(0)) / 2
        return abs(c[0] - (lo[0] + hi[0]) / 2) <= 0.25 * W and P[:, 1].min() >= lo[1] - 0.05 * Hh and lo[2] <= c[2] <= hi[2]
    if ok(np.eye(4)):
        return None
    depth = {}
    for i, nd in enumerate(body_nodes):
        depth[i] = 0 if nd['parent'] < 0 else depth.get(nd['parent'], 0) + 1
    for b in sorted(range(len(body_nodes)), key=lambda i: (depth[i], i)):
        if not np.allclose(BW[b], np.eye(4), atol=1e-4) and ok(BW[b]):
            return b
    return None


# ------------------------------------------------------------------ animation playback
class Animator:
    """Poses a (viewed) model with an animation clip.
    skeletons: {merged part index: node list} (a part whose skeleton differs from the body's stays still);
    body: the body skeleton (attachments follow their anchor bone in it, stored as submesh['_anchor']).
    Skinned submeshes deform through their weights; rigid groups follow their node; vertices are taken in
    their bind/placed positions, so pose(None) gives the model unchanged."""

    def __init__(self, model, skeletons, body=None):
        self.body = body
        self.sk = {}
        for k, nodes in skeletons.items():
            if nodes:
                W0 = world(nodes)
                self.sk[k] = (nodes, [np.linalg.pinv(W0[j]) for j in range(len(nodes))])
        if body:
            W0 = world(body)
            self.body_inv = [np.linalg.pinv(W0[j]) for j in range(len(body))]
        self.items = []
        for idx, s in enumerate(model['meshes']):
            if 'position' not in s or not len(s['position']):
                continue
            P = np.c_[s['position'][:, :3].astype(np.float64), np.ones(len(s['position']))]
            N = s['normal'][:, :3].astype(np.float64) if 'normal' in s else None
            if '_anchor' in s:
                if s['_anchor'] is not None and body:
                    self.items.append(('anchor', idx, P, N, s['_anchor']))
                continue
            k = s.get('part', 0)
            if k not in self.sk:
                continue
            nodes = self.sk[k][0]
            if 'joints' in s and s.get('palette'):
                pal = np.array([p if 0 <= p < len(nodes) else -1 for p in s['palette']] + [-1])
                J = pal[np.minimum(s['joints'].astype(int), len(pal) - 1)]
                w = s['weights'].astype(np.float64)
                w = w / np.maximum(w.sum(1, keepdims=True), 1e-9)
                self.items.append(('skin', idx, P, N, (k, J, w)))
            else:
                mid = s['mesh'] % 1000
                node = next((i for i, nd in enumerate(nodes) for g, _ in nd['meshes'] if g == mid), None)
                if node is not None:
                    self.items.append(('rigid', idx, P, N, (k, node)))

    @property
    def animated(self):
        return bool(self.items)

    def pose(self, clip, time):
        """[(submesh index, positions (n,3) float32, normals or None)] for the clip at time (seconds)."""
        mats = {}
        for k, (nodes, inv) in self.sk.items():
            Wp = pose(nodes, clip, time) if clip else world(nodes)
            mats[k] = np.stack([Wp[j] @ inv[j] for j in range(len(nodes))] + [np.eye(4)])
        if self.body:
            Wb = pose(self.body, clip, time) if clip else world(self.body)
            bmats = [Wb[j] @ self.body_inv[j] for j in range(len(self.body))]
        out = []
        for kind, idx, P, N, extra in self.items:
            if kind == 'skin':
                k, J, w = extra
                S = mats[k][J]                                   # (n, 4, 4, 4)
                M = np.einsum('nk,nkij->nij', w, S)
                pos = np.einsum('nij,nj->ni', M, P)[:, :3]
                nrm = np.einsum('nij,nj->ni', M[:, :3, :3], N) if N is not None else None
            else:
                M = mats[extra[0]][extra[1]] if kind == 'rigid' else bmats[extra]
                pos = P @ M.T
                pos = pos[:, :3]
                nrm = N @ M[:3, :3].T if N is not None else None
            if nrm is not None:
                nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
                nrm = nrm.astype(np.float32)
            out.append((idx, pos.astype(np.float32), nrm))
        return out


def clip_channels(nodes, clip, min_scale=1e-4):
    """glTF-style channels of a clip for a skeleton: [(joint index, 'rotation'|'translation'|'scale',
    times (n,), values (n, 4|3))]. Uses the same precedence as pose_locals (constant values win over keys).
    Quaternions are kept sign-continuous; zero scales are clamped like the exported rig."""
    out = []
    names = {'r': 'rotation', 't3': 'translation', 's': 'scale'}
    for j, nd in enumerate(nodes):
        tr = clip['tracks'].get(nd['idx'])
        if not tr:
            continue
        keys, const = tr
        for ch, path in names.items():
            if ch in const:
                times = np.array([0.0], np.float32)
                vals = np.array([const[ch]], np.float32)
            else:
                ks = [k for k in keys if ch in k]
                if not ks:
                    continue
                times = np.array([k['t'] for k in ks], np.float32)
                vals = np.array([k[ch] for k in ks], np.float32)
                order = np.argsort(times, kind='stable')
                times, vals = times[order], vals[order]
                keep = np.r_[np.diff(times) > 1e-6, True]          # strictly increasing times
                times, vals = times[keep], vals[keep]
            if path == 'rotation':
                vals = vals / np.maximum(np.linalg.norm(vals, axis=1, keepdims=True), 1e-9)
                for i in range(1, len(vals)):
                    if np.dot(vals[i - 1], vals[i]) < 0:
                        vals[i] = -vals[i]
            elif path == 'scale':
                vals = np.where(np.abs(vals) < min_scale, np.where(vals >= 0, min_scale, -min_scale), vals)
            out.append((j, path, times.astype(np.float32), vals.astype(np.float32)))
    return out

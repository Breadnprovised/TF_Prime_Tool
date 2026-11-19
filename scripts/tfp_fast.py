"""Optional compiled (numba) kernels. Used automatically when numba is installed; the tool works without it."""
import numpy as np

try:
    import numba
    HAVE_NUMBA = True
except Exception:          # pragma: no cover
    HAVE_NUMBA = False

if HAVE_NUMBA:
    @numba.njit(cache=True)
    def _cp_v1(W, size):
        out = np.zeros(size, np.uint8)
        wi = 0; buf = np.uint64(0); n = 0; o = 0
        while o < size:
            if n < 17:
                buf |= np.uint64(W[wi]) << np.uint64(n); wi += 1; n += 32
            flag = buf & np.uint64(1); buf >>= np.uint64(1); n -= 1
            if flag == 0:
                out[o] = np.uint8(buf & np.uint64(0xff)); buf >>= np.uint64(8); n -= 8; o += 1
            else:
                l = int(buf & np.uint64(15)) + 3; d = int((buf >> np.uint64(4)) & np.uint64(0xfff))
                buf >>= np.uint64(16); n -= 16
                if l > size - o:
                    l = size - o
                s = o - d
                for k in range(l):
                    out[o + k] = out[s + k]
                o += l
        return out

    @numba.njit(cache=True)
    def _cp_v2(W, size):
        out = np.zeros(size, np.uint8)
        wi = 0; buf = np.uint64(0); n = 0; o = 0
        L = np.zeros(1024, np.int32); R = np.zeros(1024, np.int32); V = np.zeros(1024, np.int32)
        stack = np.zeros(1024, np.int32); side = np.zeros(1024, np.int8)
        while o < size:
            # count: 16 bits
            if n < 16:
                buf |= np.uint64(W[wi]) << np.uint64(n); wi += 1; n += 32
            cnt = int(buf & np.uint64(0xffff)); buf >>= np.uint64(16); n -= 16
            # tree (pre-order): bit 1 -> internal node (first child = bit-1 branch), bit 0 -> 9-bit leaf
            nn = 0; sp = 0
            # create root
            V[0] = 0; L[0] = 0; R[0] = 0; nn = 1
            stack[0] = 0; side[0] = 0; sp = 1
            cur = 0
            pending = True
            node = 0
            while pending:
                # read node 'node'
                if n == 0:
                    buf = np.uint64(W[wi]); wi += 1; n = 32
                b = buf & np.uint64(1); buf >>= np.uint64(1); n -= 1
                if b:
                    V[node] = -1
                    # left child first
                    c = nn; nn += 1; V[c] = 0; L[c] = 0; R[c] = 0
                    L[node] = c
                    stack[sp] = node; sp += 1
                    node = c
                    continue
                if n < 9:
                    buf |= np.uint64(W[wi]) << np.uint64(n); wi += 1; n += 32
                V[node] = int(buf & np.uint64(511)); buf >>= np.uint64(9); n -= 9
                # climb to the first ancestor whose right child is still missing
                pending = False
                while sp > 1:
                    sp -= 1
                    par = stack[sp]
                    if R[par] == 0:
                        c = nn; nn += 1; V[c] = 0; L[c] = 0; R[c] = 0
                        R[par] = c
                        stack[sp] = par; sp += 1
                        # mark: right child created; prevent re-entry by using negative marker
                        R[par] = c
                        node = c
                        pending = True
                        break
            while cnt > 0 and o < size:
                nd = 0
                while V[nd] < 0:
                    if n == 0:
                        buf = np.uint64(W[wi]); wi += 1; n = 32
                    b = buf & np.uint64(1); buf >>= np.uint64(1); n -= 1
                    nd = L[nd] if b else R[nd]
                s = V[nd]; cnt -= 1
                if s < 256:
                    out[o] = s; o += 1
                else:
                    if n < 12:
                        buf |= np.uint64(W[wi]) << np.uint64(n); wi += 1; n += 32
                    d = int(buf & np.uint64(0xfff)); buf >>= np.uint64(12); n -= 12
                    l = s - 0xfd
                    if l > size - o:
                        l = size - o
                    cnt -= 1
                    st = o - d
                    for k in range(l):
                        out[o + k] = out[st + k]
                    o += l
        return out


def decompress(words, ver, size):
    """words: big-endian u32 stream as numpy uint32 (plus 2 padding words). Returns bytes or None."""
    if not HAVE_NUMBA:
        return None
    W = np.ascontiguousarray(words, dtype=np.uint32)
    if ver == 1:
        return _cp_v1(W, size).tobytes()
    if ver == 2:
        return _cp_v2(W, size).tobytes()
    return None


if HAVE_NUMBA:
    @numba.njit(cache=True)
    def _bc_color_k(blk, opaque):
        n = blk.shape[0]
        out = np.empty((n, 16, 4), np.int32)
        pal = np.empty((4, 4), np.int32)
        for b in range(n):
            c0 = int(blk[b, 0]) | (int(blk[b, 1]) << 8)
            c1 = int(blk[b, 2]) | (int(blk[b, 3]) << 8)
            r0 = ((c0 >> 11) & 31) * 255 // 31; g0 = ((c0 >> 5) & 63) * 255 // 63; b0 = (c0 & 31) * 255 // 31
            r1 = ((c1 >> 11) & 31) * 255 // 31; g1 = ((c1 >> 5) & 63) * 255 // 63; b1 = (c1 & 31) * 255 // 31
            four = (c0 > c1) or opaque
            pal[0, 0] = r0; pal[0, 1] = g0; pal[0, 2] = b0; pal[0, 3] = 255
            pal[1, 0] = r1; pal[1, 1] = g1; pal[1, 2] = b1; pal[1, 3] = 255
            if four:
                pal[2, 0] = (2 * r0 + r1) // 3; pal[2, 1] = (2 * g0 + g1) // 3; pal[2, 2] = (2 * b0 + b1) // 3
                pal[3, 0] = (r0 + 2 * r1) // 3; pal[3, 1] = (g0 + 2 * g1) // 3; pal[3, 2] = (b0 + 2 * b1) // 3
                pal[3, 3] = 255
            else:
                pal[2, 0] = (r0 + r1) // 2; pal[2, 1] = (g0 + g1) // 2; pal[2, 2] = (b0 + b1) // 2
                pal[3, 0] = 0; pal[3, 1] = 0; pal[3, 2] = 0; pal[3, 3] = 0
            pal[2, 3] = 255
            bits = int(blk[b, 4]) | (int(blk[b, 5]) << 8) | (int(blk[b, 6]) << 16) | (int(blk[b, 7]) << 24)
            for i in range(16):
                ix = (bits >> (2 * i)) & 3
                for c in range(4):
                    out[b, i, c] = pal[ix, c]
        return out

    @numba.njit(cache=True)
    def _bc_alpha_k(blk, signed):
        n = blk.shape[0]
        out = np.empty((n, 16), np.int32)
        pal = np.empty(8, np.int32)
        for b in range(n):
            a0 = int(blk[b, 0]); a1 = int(blk[b, 1])
            if signed:
                a0 = (a0 + 128) & 255; a1 = (a1 + 128) & 255
            pal[0] = a0; pal[1] = a1
            if a0 > a1:
                for i in range(1, 7):
                    pal[1 + i] = ((7 - i) * a0 + i * a1) // 7
            else:
                for i in range(1, 5):
                    pal[1 + i] = ((5 - i) * a0 + i * a1) // 5
                pal[6] = 0; pal[7] = 255
            bits = 0
            for i in range(6):
                bits |= int(blk[b, 2 + i]) << (8 * i)
            for i in range(16):
                out[b, i] = pal[(bits >> (3 * i)) & 7]
        return out


def bc_color(blk, opaque):
    """blk (..., 8) uint8 -> (..., 16, 4) int32, or None without numba."""
    if not HAVE_NUMBA:
        return None
    sh = blk.shape[:-1]
    return _bc_color_k(np.ascontiguousarray(blk.reshape(-1, 8)), bool(opaque)).reshape(sh + (16, 4))


def bc_alpha(blk, signed):
    if not HAVE_NUMBA:
        return None
    sh = blk.shape[:-1]
    return _bc_alpha_k(np.ascontiguousarray(blk.reshape(-1, 8)), bool(signed)).reshape(sh + (16,))

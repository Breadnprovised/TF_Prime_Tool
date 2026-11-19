"""Audio for Transformers Prime (Wii U): the sound archive (content/TRF_SOUND.bfsar) and its streams
(content/stream/NNNNN.dspadpcm.bfstm).

- BFSTM streams and BFWAV waves are decoded (DSP-ADPCM, PCM16, PCM8) to int16 arrays.
- The BFSAR gives every sound its name: stream sounds point at a stream file, wave sounds at a wave inside
  a wave archive (BFWAR, via a wave-sound file BFWSD). Sequence sounds (note data played through an
  instrument bank) can't be rendered; their instrument samples are listed with the wave archives.
- WAV export keeps loop points (a 'smpl' chunk).
"""
import os
import struct
import sys

import numpy as np

try:
    from numba import njit
except ImportError:                        # numba is optional: a pure-Python decoder is used instead
    njit = None


# ---------------------------------------------------------------- DSP-ADPCM

def _dsp_decode_py(data, coefs, h1, h2, n, out):
    for f in range((n + 13) // 14):
        hdr = np.int64(data[f * 8])
        scale = np.int64(1) << (hdr & 0xF)
        ci = (hdr >> 4) & 7
        c1 = np.int64(coefs[ci * 2])
        c2 = np.int64(coefs[ci * 2 + 1])
        for k in range(14):
            i = f * 14 + k
            if i >= n:
                break
            b = np.int64(data[f * 8 + 1 + k // 2])
            if k & 1:
                nib = b & 0xF
            else:
                nib = b >> 4
            if nib >= 8:
                nib -= 16
            s = ((nib * scale) << 11) + 1024 + c1 * h1 + c2 * h2
            s >>= 11
            if s > 32767:
                s = np.int64(32767)
            elif s < -32768:
                s = np.int64(-32768)
            out[i] = s
            h2 = h1
            h1 = s
    return h1, h2


_dsp_decode = njit(cache=True, nogil=True)(_dsp_decode_py) if njit else _dsp_decode_py


def dsp_decode(data, coefs, n, h1=0, h2=0):
    """Decode n samples of one DSP-ADPCM channel (8-byte frames of 14 samples)."""
    out = np.zeros(n, np.int32)
    _dsp_decode(np.frombuffer(data, np.uint8), np.asarray(coefs, np.int64), np.int64(h1), np.int64(h2), n, out)
    return out.astype(np.int16)


def _decode_channel(codec, raw, n, adpcm):
    if codec == 2:
        return dsp_decode(raw, adpcm['coefs'], n, adpcm['h1'], adpcm['h2'])
    if codec == 1:
        return np.frombuffer(raw[:n * 2], '>i2').astype(np.int16)
    if codec == 0:
        return (np.frombuffer(raw[:n], np.int8).astype(np.int16) << 8)
    raise ValueError('unsupported audio codec %d' % codec)


CODECS = {0: 'PCM8', 1: 'PCM16', 2: 'DSP-ADPCM', 3: 'IMA-ADPCM'}


# ---------------------------------------------------------------- binary helpers

class _R:
    def __init__(self, d, base=0):
        self.d = d
        self.base = base

    def u8(self, o): return self.d[o]
    def u16(self, o): return struct.unpack_from('>H', self.d, o)[0]
    def u32(self, o): return struct.unpack_from('>I', self.d, o)[0]
    def s32(self, o): return struct.unpack_from('>i', self.d, o)[0]

    def sections(self, a):
        """Section table of a Cafe sound file at a: {id: (offset, size)}."""
        n = self.u16(a + 0x10)
        return {self.u16(a + 0x14 + 12 * i): (a + self.u32(a + 0x18 + 12 * i), self.u32(a + 0x1c + 12 * i))
                for i in range(n)}

    def ref_table(self, a):
        """Reference table at a: list of absolute addresses (relative to the table)."""
        return [a + self.s32(a + 8 + 8 * i) for i in range(self.u32(a))]

    def adpcm(self, a):
        return dict(coefs=struct.unpack_from('>16h', self.d, a), h1=struct.unpack_from('>h', self.d, a + 0x22)[0],
                    h2=struct.unpack_from('>h', self.d, a + 0x24)[0])


# ---------------------------------------------------------------- BFSTM / BFWAV

def parse_bfstm(d, decode=True):
    """BFSTM stream -> dict(rate, channels, frames, loop, loop_start, codec, tracks, samples[frames, ch])."""
    r = _R(d)
    if d[:4] != b'FSTM':
        raise ValueError('not a BFSTM file')
    sec = r.sections(0)
    info = sec[0x4000][0] + 8
    si = info + r.s32(info + 4)
    codec, loop, nch = d[si], d[si + 1], d[si + 2]
    rate, loop_start, frames, nblk, bsize, bsamp, lsize, lsamp, lpad = struct.unpack_from('>9I', d, si + 4)
    out = dict(kind='stream', rate=rate, channels=nch, frames=frames, loop=bool(loop), loop_start=loop_start,
               codec=CODECS.get(codec, codec), tracks=[])
    tref = r.s32(info + 12)
    if r.u16(info + 8) == 0x0101 and tref not in (0, -1):          # track info: which channels belong together
        t = info + tref
        for ta in r.ref_table(t):
            try:                    # track info: volume, pan, span, flags, then a reference to its channel list
                ct = ta + r.s32(ta + 8)
                n = r.u32(ct)
                chans = list(d[ct + 4:ct + 4 + n])
                if 0 < n <= nch and all(c < nch for c in chans):
                    out['tracks'].append(chans)
            except Exception:
                pass
    if not decode:
        return out
    ct = info + r.s32(info + 20)
    ad = []
    for ca in r.ref_table(ct):
        off = r.s32(ca + 4)
        ad.append(r.adpcm(ca + off) if codec == 2 and off not in (0, -1) else None)
    data = sec[0x4002][0] + 8 + r.s32(si + 52)           # sample data reference, relative to the DATA body
    # de-interleave the blocks: block b holds bsize bytes per channel (last block: lpad bytes)
    chunks = [[] for _ in range(nch)]
    p = data
    for b in range(nblk):
        sz = lpad if b == nblk - 1 else bsize
        for c in range(nch):
            chunks[c].append(d[p:p + sz])
            p += sz
    cols = []
    for c in range(nch):
        cols.append(_decode_channel(codec, b''.join(chunks[c]), frames, ad[c] or {}))
    out['samples'] = np.stack(cols, 1) if cols else np.zeros((0, 0), np.int16)
    return out


def parse_bfwav(d, a=0, decode=True):
    """BFWAV wave at offset a -> same dict shape as parse_bfstm (kind='wave')."""
    r = _R(d)
    if d[a:a + 4] != b'FWAV':
        raise ValueError('not a BFWAV file')
    sec = r.sections(a)
    info = sec[0x7000][0] + 8
    codec, loop = d[info], d[info + 1]
    rate, loop_start, frames = r.u32(info + 4), r.u32(info + 8), r.u32(info + 12)
    chs = r.ref_table(info + 0x14)
    out = dict(kind='wave', rate=rate, channels=len(chs), frames=frames, loop=bool(loop), loop_start=loop_start,
               codec=CODECS.get(codec, codec), tracks=[])
    if not decode:
        return out
    data = sec[0x7001][0] + 8
    cols = []
    for ca in chs:
        sp = data + r.s32(ca + 4)
        ao = r.s32(ca + 12)
        ad = r.adpcm(ca + ao) if codec == 2 and ao not in (0, -1) else {}
        nbytes = (frames + 13) // 14 * 8 if codec == 2 else frames * (2 if codec == 1 else 1)
        cols.append(_decode_channel(codec, d[sp:sp + nbytes], frames, ad))
    out['samples'] = np.stack(cols, 1) if cols else np.zeros((0, 0), np.int16)
    return out


# ---------------------------------------------------------------- WAV

def wav_bytes(samples, rate, loop=False, loop_start=0):
    """16-bit PCM WAV; loops are kept as a 'smpl' chunk (read by most samplers and audio editors)."""
    s = np.ascontiguousarray(samples, '<i2')
    if s.ndim == 1:
        s = s[:, None]
    n, ch = s.shape
    data = s.tobytes()
    fmt = struct.pack('<HHIIHH', 1, ch, rate, rate * ch * 2, ch * 2, 16)
    chunks = b'fmt ' + struct.pack('<I', len(fmt)) + fmt + b'data' + struct.pack('<I', len(data)) + data
    if len(data) & 1:
        chunks += b'\0'
    if loop and n:
        smpl = struct.pack('<9I', 0, 0, int(1e9 / rate), 60, 0, 0, 0, 1, 0)
        smpl += struct.pack('<6I', 0, 0, int(loop_start), n - 1, 0, 0)
        chunks += b'smpl' + struct.pack('<I', len(smpl)) + smpl
    return b'RIFF' + struct.pack('<I', 4 + len(chunks)) + b'WAVE' + chunks


def write_wav(path, audio):
    with open(path, 'wb') as f:
        f.write(wav_bytes(audio['samples'], audio['rate'], audio['loop'], audio['loop_start']))


# ---------------------------------------------------------------- BFSAR

class Sound:
    """One named sound. kind: 'stream' | 'wave' | 'sequence' | 'warc-wave' (a raw wave in an archive)."""
    __slots__ = ('name', 'kind', 'index', 'file', 'warc', 'wave', 'group', 'player')

    def __init__(self, name, kind, index, file=None, warc=None, wave=None, group='', player=''):
        self.name, self.kind, self.index = name, kind, index
        self.file, self.warc, self.wave, self.group, self.player = file, warc, wave, group, player


class SoundArchive:
    """Index of TRF_SOUND.bfsar. sounds: list of Sound. load(sound) -> decoded audio dict."""

    def __init__(self, path, content_dir=None):
        self.path = path
        self.content = content_dir or os.path.dirname(path)
        with open(path, 'rb') as f:
            self.d = d = f.read()
        r = self.r = _R(d)
        if d[:4] != b'FSAR':
            raise ValueError('not a BFSAR file')
        sec = r.sections(0)
        # strings
        sb = sec[0x2000][0] + 8
        st = sb + r.s32(sb + 4)
        self.strings = []
        for i in range(r.u32(st)):
            o = st + 4 + 12 * i
            a = st + r.s32(o + 4)
            self.strings.append(d[a:a + r.u32(o + 8) - 1].decode('ascii', 'replace'))
        ib = sec[0x2001][0] + 8
        tabs = {r.u16(ib + 8 * i): ib + r.s32(ib + 8 * i + 4) for i in range(8)}
        self.file_base = sec[0x2002][0] + 8
        # files: ('ext', path) or ('int', offset, size); files stored only inside groups are found there
        self.files = []
        for fa in r.ref_table(tabs[0x2106]):
            kind = r.u16(fa)
            p = fa + r.s32(fa + 4)
            if kind == 0x220D:
                self.files.append(('ext', d[p:d.index(b'\0', p)].decode('ascii', 'replace')))
            else:
                off, size = r.s32(p + 4), r.u32(p + 8)
                ok = off != -1 and size not in (0, 0xFFFFFFFF)
                self.files.append(('int', self.file_base + off, size) if ok else ('none',))
        for k in range(len(self.files)):
            f = self.files[k]
            if f[0] == 'int' and d[f[1]:f[1] + 4] == b'FGRP':
                gs = r.sections(f[1])
                gi, gf = gs[0x7800][0] + 8, gs[0x7801][0] + 8
                for e in r.ref_table(gi):
                    fid, off, size = r.u32(e), r.s32(e + 8), r.u32(e + 12)
                    if fid < len(self.files) and self.files[fid][0] == 'none' and off != -1 and size not in (0, 0xFFFFFFFF):
                        self.files[fid] = ('int', gf + off, size)
        self.warc_files = [r.u32(a) for a in r.ref_table(tabs[0x2103])]
        players = [self._name(a, 4) for a in r.ref_table(tabs[0x2102])]
        self._wsd = {}
        self.sounds = []
        for i, a in enumerate(r.ref_table(tabs[0x2100])):
            fid, pid = r.u32(a), r.u32(a + 4) & 0xFFFFFF
            det = r.u16(a + 12)
            flags = r.u32(a + 20)
            name = self.strings[r.u32(a + 24)] if flags & 1 else 'sound_%05d' % i
            player = players[pid] if pid < len(players) else ''
            if det == 0x2201:
                self.sounds.append(Sound(name, 'stream', i, file=fid, player=player))
            elif det == 0x2202:
                idx = r.u32(a + r.s32(a + 16))
                self.sounds.append(Sound(name, 'wave', i, file=fid, wave=idx, player=player))
            else:
                self.sounds.append(Sound(name, 'sequence', i, file=fid, player=player))
        self.by_name = {s.name: s for s in self.sounds}
        # raw waves of every wave archive (also the instrument samples sequences play)
        self.warc_names = [self._warc_name(a, k) for k, a in enumerate(r.ref_table(tabs[0x2103]))]
        self.raw_waves = []
        for k, fid in enumerate(self.warc_files):
            for j in range(self._warc_count(fid)):
                self.raw_waves.append(Sound('%s_%03d' % (self.warc_names[k], j), 'warc-wave', len(self.raw_waves),
                                            file=fid, warc=k, wave=j, group=self.warc_names[k]))

    def _name(self, a, at):
        r = self.r
        try:
            flags = r.u32(a + at)
            sid = r.u32(a + at + 4)
            return self.strings[sid] if flags & 1 and sid < len(self.strings) else ''
        except Exception:
            return ''

    def _warc_name(self, a, k):
        n = self._name(a, 8)
        return n or 'WARC_%03d' % k

    def _file(self, fid):
        f = self.files[fid] if fid < len(self.files) else ('none',)
        if f[0] == 'int':
            return f[1], f[2]
        return None

    def _warc_count(self, fid):
        f = self._file(fid)
        if not f or self.d[f[0]:f[0] + 4] != b'FWAR':
            return 0
        info = self.r.sections(f[0])[0x6800][0] + 8
        return self.r.u32(info)

    def _warc_wave(self, fid, j):
        f = self._file(fid)
        if not f:
            raise ValueError('wave archive file %d is not in the sound archive' % fid)
        s = self.r.sections(f[0])
        info, data = s[0x6800][0] + 8, s[0x6801][0] + 8
        return data + self.r.s32(info + 4 + 12 * j + 4)

    def _wsd_waves(self, fid):
        """BFWSD wave-sound file -> per entry: [(warc index, wave index), ...] (one per note)."""
        if fid in self._wsd:
            return self._wsd[fid]
        r, out = self.r, []
        f = self._file(fid)
        if f and self.d[f[0]:f[0] + 4] == b'FWSD':
            b = r.sections(f[0])[0x6800][0] + 8
            idt = b + r.s32(b + 4)
            ids = [(r.u32(idt + 4 + 8 * i) & 0xFFFFFF, r.u32(idt + 8 + 8 * i)) for i in range(r.u32(idt))]
            for e in r.ref_table(b + r.s32(b + 12)):
                nt = e + r.s32(e + 0x14)
                out.append([ids[r.u32(n)] for n in r.ref_table(nt) if r.u32(n) < len(ids)])
        self._wsd[fid] = out
        return out

    def stream_path(self, s):
        f = self.files[s.file] if s.file < len(self.files) else ('none',)
        if f[0] != 'ext':
            return None
        return os.path.join(self.content, *f[1].split('/'))

    def describe(self, s):
        """Short source description for the list ('stream/00001', 'WARC_SE_SYS #3', 'sequence')."""
        if s.kind == 'stream':
            f = self.files[s.file] if s.file < len(self.files) else ('none',)
            return f[1].split('/')[-1].split('.')[0] if f[0] == 'ext' else ''
        if s.kind == 'wave':
            w = self._wsd_waves(s.file)
            if s.wave < len(w) and w[s.wave]:
                a, j = w[s.wave][0]
                return '%s #%d' % (self.warc_names[a] if a < len(self.warc_names) else a, j)
            return ''
        if s.kind == 'warc-wave':
            return '#%d' % s.wave
        return 'sequence'

    def load(self, s, decode=True):
        """Decode a Sound -> audio dict (samples int16 [frames, channels], rate, loop, loop_start, ...)."""
        if s.kind == 'stream':
            p = self.stream_path(s)
            if not p or not os.path.isfile(p):
                raise FileNotFoundError('stream file not found: %s' % p)
            with open(p, 'rb') as f:
                return parse_bfstm(f.read(), decode)
        if s.kind == 'wave':
            w = self._wsd_waves(s.file)
            if s.wave >= len(w) or not w[s.wave]:
                raise ValueError('wave sound %s has no wave' % s.name)
            a, j = w[s.wave][0]
            return parse_bfwav(self.d, self._warc_wave(self.warc_files[a], j), decode)
        if s.kind == 'warc-wave':
            return parse_bfwav(self.d, self._warc_wave(s.file, s.wave), decode)
        raise ValueError('%s is a sequence (note data played through an instrument bank) and can\'t be '
                         'rendered; its instrument samples are listed under the wave archives' % s.name)


def find_sound_archive(game_dir):
    """The sound archive (content/TRF_SOUND.bfsar) under a game folder, or None."""
    best = None
    base = game_dir.rstrip('\\/').count(os.sep)
    for root, dirs, files in os.walk(game_dir):
        if root.count(os.sep) - base >= 3:           # it sits in content/, no need to search deep
            dirs[:] = []
        for n in files:
            if n.lower().endswith('.bfsar'):
                p = os.path.join(root, n)
                if best is None or n.upper() == 'TRF_SOUND.BFSAR':
                    best = p
    return best


# ---------------------------------------------------------------- list entries for the tool

LANGS = {'DU': 'DU', 'FR': 'FR', 'GE': 'GE', 'IT': 'IT', 'SP': 'SP', 'SW': 'SW'}


def sound_group(name):
    """Folder path in the list (and under .exported/audio) for a sound name."""
    p = name.split('_')
    if p[0] in ('BGM', 'JNG'):
        return ('Music',)
    if p[0] == 'SE':
        return ('Sound effects',)
    if p[0] == 'MA':
        return ('MA',)
    if p[0] == 'VO':
        lang = LANGS.get(p[-1], 'English')            # localized copies end in a language code
        return ('Voice', lang, p[1] if len(p) > 2 else 'other')
    return ('Other',)


class AudioEntry:
    """A sound as a list entry (mirrors the fields the tool's archive entries have)."""
    cls = 'audio'
    part = -1
    off = 0
    index = 0
    tset = -1
    attach = ()

    def __init__(self, archive, sound, group):
        self.archive, self.sound, self.group = archive, sound, group
        self.name = sound.name
        self.dat = archive.path
        self.size = 0
        self.key = ('audio', sound.kind, sound.index)

    @property
    def kind(self):
        return self.sound.kind


def index_audio(archive):
    """AudioEntry list: every named sound, then every raw wave of every wave archive."""
    out = []
    for s in archive.sounds:
        e = AudioEntry(archive, s, sound_group(s.name))
        if s.kind == 'stream':
            p = archive.stream_path(s)
            try:
                e.size = os.path.getsize(p) if p else 0
            except OSError:
                e.size = 0
        elif s.kind == 'wave':
            try:
                a = archive.load(s, decode=False)
                e.size = (a['frames'] + 13) // 14 * 8 * a['channels']
            except Exception:
                pass
        out.append(e)
    for s in archive.raw_waves:
        e = AudioEntry(archive, s, ('Wave archives', s.group))
        try:
            a = archive.load(s, decode=False)
            e.size = (a['frames'] + 13) // 14 * 8 * a['channels']
        except Exception:
            pass
        out.append(e)
    return out


def split_tracks(a):
    """A multi-track stream as one audio dict per track (layers the game mixes); otherwise [a]."""
    tr = a.get('tracks') or []
    if len(tr) < 2 or 'samples' not in a:
        return [a]
    return [dict(a, samples=a['samples'][:, t], channels=len(t), tracks=[]) for t in tr]


def preview(a):
    """What the player gets: the first track of a multi-track stream (OS players handle 1-2 channels)."""
    p = split_tracks(a)[0]
    if p['channels'] > 2:
        p = dict(p, samples=p['samples'][:, :2], channels=2)
    return p


def export_audio(e, outdir):
    """Write an AudioEntry as WAV into outdir/<group path>/<name>.wav (multi-track streams: one WAV per
    track, <name>_track<N>.wav). Returns the paths."""
    a = e.archive.load(e.sound)
    d = os.path.join(outdir, *e.group)
    os.makedirs(d, exist_ok=True)
    parts = split_tracks(a)
    out = []
    for k, x in enumerate(parts):
        p = os.path.join(d, e.name + ('_track%d' % k if len(parts) > 1 else '') + '.wav')
        write_wav(p, x)
        out.append(p)
    return out


# ---------------------------------------------------------------- playback (no extra packages)

class Player:
    """Plays int16 audio through the OS: winsound on Windows, afplay on macOS, paplay/aplay/ffplay on Linux."""

    def __init__(self):
        import tempfile
        self.tmp = os.path.join(tempfile.gettempdir(), 'tfp_tool_preview_%d.wav' % os.getpid())
        self.proc = None
        self.start = None
        self.end = 0.0

    def play(self, audio, offset=0.0):
        """Play from offset seconds. Returns False if no player is available."""
        import time
        self.stop()
        s = audio['samples'][int(offset * audio['rate']):]
        with open(self.tmp, 'wb') as f:
            f.write(wav_bytes(s, audio['rate']))
        self.end = offset + len(s) / float(audio['rate'])
        ok = self._launch()
        self.start = time.time() - offset if ok else None
        return ok

    def _launch(self):
        import subprocess
        import shutil
        if sys.platform.startswith('win'):
            import winsound
            winsound.PlaySound(self.tmp, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
            return True
        cmds = [['afplay', self.tmp]] if sys.platform == 'darwin' else [
            ['paplay', self.tmp], ['aplay', '-q', self.tmp], ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', self.tmp]]
        for c in cmds:
            if shutil.which(c[0]):
                self.proc = subprocess.Popen(c, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
        return False

    def position(self):
        """Seconds from the start of the sound, or None when stopped or finished."""
        import time
        if self.start is None:
            return None
        t = time.time() - self.start
        return t if t < self.end else None

    def stop(self):
        if sys.platform.startswith('win'):
            try:
                import winsound
                winsound.PlaySound(None, 0)
            except Exception:
                pass
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass
            self.proc = None
        self.start = None


#!/usr/bin/env python3
"""
Transformers Prime: The Game (Wii U) - model & texture viewer / extractor.

Requires: Python 3.8+, numpy, Pillow   (pip install numpy pillow)

GUI:   python tfp_tool.py
CLI:   python tfp_tool.py extract <game folder> <output folder> [--obj] [--textures] [--raw] [--placeholders]

Select the game folder (the one containing code/ content/ meta/, or the content folder itself).
"""
import os
import sys
import glob
import threading
import queue
import traceback

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tfp_core as core
import tfp_render as rend
import tfp_export as exp

try:  # GPU viewer (Windows/Linux): pip install PyOpenGL pyopengltk
    if sys.platform == 'darwin':
        raise ImportError('pyopengltk does not support macOS')
    import tfp_glview
    HAVE_GL = True
except Exception as _gl_err:
    tfp_glview = None
    HAVE_GL = False

PLACEHOLDER_SIZE = 0xc48  # the game's dummy unit-cube model


TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(TOOL_DIR, '.exported')     # every export goes here


def open_folder_in_os(path):
    import subprocess
    os.makedirs(path, exist_ok=True)
    try:
        if sys.platform.startswith('win'):
            os.startfile(path)
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', path])
        else:
            subprocess.Popen(['xdg-open', path])
    except Exception:
        pass


def find_dats(folder):
    pats = [os.path.join(folder, '**', '*.dat')]
    found = []
    for p in pats:
        found += glob.glob(p, recursive=True)
    dats = [f for f in found if os.path.basename(f).lower().startswith('d') and os.path.getsize(f) > 0x800]
    return sorted(dats, key=lambda p: os.path.basename(p).lower())


def is_placeholder(e):
    return e.cls == 'model' and e.size == PLACEHOLDER_SIZE


# ------------------------------------------------------------------ loading helpers (cached)
class Loader:
    def __init__(self):
        self.cache = {}
        self.order = []
        self.sizes = {}
        self.total = 0
        self.lock = threading.Lock()
        self.overrides = {}
        self.assembly = {}      # (dat, off) -> list of bools: which parts an assembled view uses

    CACHE_BYTES = 1500 << 20        # keep up to ~1.5 GB of decoded data, least recently used dropped first

    @staticmethod
    def _size(v, depth=0, seen=None):
        """Approximate memory of a cached value; shared buffers (e.g. the package bytes every texture info
        points to) are counted once."""
        if seen is None:
            seen = set()
        if isinstance(v, (bytes, bytearray, np.ndarray)):
            if id(v) in seen:
                return 0
            seen.add(id(v))
            return len(v) if not isinstance(v, np.ndarray) else v.nbytes
        if depth > 4:
            return 64
        if isinstance(v, dict):
            return 64 + sum(Loader._size(x, depth + 1, seen) for x in v.values())
        if isinstance(v, (list, tuple)):
            return 64 + sum(Loader._size(x, depth + 1, seen) for x in v)
        return 64

    def _get(self, key, fn):
        with self.lock:
            if key in self.cache:
                try:
                    self.order.remove(key)
                except ValueError:
                    pass
                self.order.append(key)                 # most recently used goes last
                return self.cache[key]
        v = fn()
        with self.lock:
            if key not in self.cache:
                self.cache[key] = v
                self.order.append(key)
                self.sizes[key] = self._size(v)
                self.total += self.sizes[key]
            while self.total > self.CACHE_BYTES and len(self.order) > 1:
                old = self.order.pop(0)
                self.cache.pop(old, None)
                self.total -= self.sizes.pop(old, 0)
        return v

    def data(self, e):
        """Decompressed bytes; for a part entry, just that GEO block."""
        whole = self._get(('d', e.dat, e.off), lambda: core.load_entry(e))
        if e.part < 0:
            return whole
        parts = self._get(('gp', e.dat, e.off), lambda: core.geo_parts(whole))
        if e.part >= len(parts):
            raise ValueError('package has only %d GEO parts' % len(parts))
        a, b = parts[e.part]
        return whole[a:b]

    def whole(self, e):
        return self.data(e.clone(part=-1))

    def model(self, e):
        if e.part == -2:
            return self.assemble(e)[0]
        return self._get(('m',) + e.key, lambda: self._placed(e))

    def raw_model(self, e):
        """Model exactly as stored (mesh groups in their node's local space)."""
        return self._get(('raw',) + e.key, lambda: core.parse_geo(self.data(e)))

    def owner_of(self, e):
        """Package whose attachment e is (needs self.entries), or None."""
        for x in getattr(self, 'entries', None) or ():
            if x.part == -2 and x.dat == e.dat and any(a.key == e.key for a in x.attach):
                return x
        return None

    def _placed(self, e):
        m = self.raw_model(e)
        if e.part == -1:
            pkg = self.owner_of(e)
            if pkg is not None:       # a weapon etc.: show it posed where it sits on the body
                return self.posed_attachment(pkg, e, self.entries)
        try:
            return core.place_by_nodes(m, core.geo_nodes(self.data(e)))
        except Exception:
            return m

    def parts(self, e):
        """The package's own parts, followed by its attachment models (weapons etc.)."""
        return [e.clone(part=k, attach=()) for k in range(e.nparts)] + list(e.attach)

    def assembly_use(self, e):
        use = self.assembly.get((e.dat, e.off))
        if use is None or len(use) != len(self.parts(e)):
            use = self.assembly_use_default(e)
        return use

    def assembly_use_default(self, e):
        own = [self.model(p) for p in self.parts(e)[:e.nparts]]
        return core.default_assembly(own) + [False] * len(e.attach)   # attachments start off

    def anim_files(self, e, entries):
        """Animation files stored after a model / package (up to the next model)."""
        if not entries:
            return []
        plain = [x for x in entries if x.part < 0 and x.dat == e.dat]
        try:
            i = next(k for k, x in enumerate(plain) if x.off == e.off)
        except StopIteration:
            return []
        out = []
        for x in plain[i + 1:]:
            if x.cls == 'model':
                break
            if x.cls == 'other' and x.size > 0x40 and b'ANIMWIIU' in self.data(x)[:0x100]:
                out.append(x)
        return out

    def clips_for(self, e, entries):
        """[(label, clip)] of every animation stored with this model / package."""
        def make():
            out = []
            for x in self.anim_files(e, entries):
                try:
                    for c in core.parse_anim(self.data(x)):
                        out.append(('%04d  %s' % (x.index, c['name']), c))
                except Exception:
                    pass
            return out
        return self._get(('clips',) + e.key, make)

    def animator(self, e, view_model):
        """core.Animator for the model currently shown for entry e."""
        body = None
        if e.part == -2:
            body = self._body_nodes(e)
            allp = self.parts(e)
            used = [k for k, u in enumerate(self.assembly_use(e)) if u]
            sk = {}
            for i, k in enumerate(used):
                if k < e.nparts:
                    nodes = core.geo_nodes(self.data(allp[k]))
                    sk[i] = nodes if len(nodes) == len(body) else None
        else:
            sk = {0: core.geo_nodes(self.data(e))}
        return core.Animator(view_model, sk, body)

    def rig_for(self, e, model, entries, animations=True):
        """Skeletons + per-vertex joints/weights for exporting `model` (the model as exported for entry e).
        Returns None when the model has no node hierarchy. Assembled packages: parts sharing the body skeleton
        bind to it, other parts get their own, and each attachment gets its own skeleton posed as shown and
        parented under the body bone it is mounted on."""
        skels, part_skel = [], {}

        def clamp(locs):
            return [(r, t, tuple(v if abs(v) > 1e-4 else (1e-4 if v >= 0 else -1e-4) for v in s)) for r, t, s in locs]

        def add(nodes, locs, attach=None, prefix='bone'):
            locs = clamp(locs)
            W = core.world(nodes, [core.trs(*l) for l in locs])
            skels.append(dict(nodes=nodes, locals=locs, attach=attach,
                              names=['%s%d' % (prefix, i) for i in range(len(nodes))],
                              inv=[np.linalg.inv(W[j]) for j in range(len(nodes))], world=W))
            return len(skels) - 1

        bind = lambda nodes: [(tuple(nd['r']), tuple(nd['t']), tuple(nd['s'])) for nd in nodes]
        try:
            if e.part == -2:
                body = self._body_nodes(e)
                b = add(body, bind(body))
                allp = self.parts(e)
                used = [k for k, u in enumerate(self.assembly_use(e)) if u]
                anchors = {}
                for s in model['meshes']:
                    if '_anchor' in s:
                        anchors.setdefault(s.get('part', 0), s['_anchor'])
                for i, k in enumerate(used):
                    if k < e.nparts:
                        nodes = core.geo_nodes(self.data(allp[k]))
                        part_skel[i] = b if len(nodes) == len(body) else add(nodes, bind(nodes), prefix='part%d_bone' % k)
                    else:
                        a = allp[k]
                        _, nodes, clips, _, _ = self._attach_info(e, a, entries)
                        clip, tm = core.pick_clip(clips)
                        anc = anchors.get(i)
                        part_skel[i] = add(nodes, core.pose_locals(nodes, clip, tm),
                                           attach=(b, anc) if anc is not None else None,
                                           prefix='att%04d_bone' % a.index)
                        # the posed attachment mesh sits at body bone `anc`: its joints' world = H @ pose
                        if anc is not None:
                            H = skels[b]['world'][anc]
                            sk = skels[-1]
                            sk['world'] = {j: H @ sk['world'][j] for j in sk['world']}
                            sk['inv'] = [np.linalg.inv(sk['world'][j]) for j in range(len(nodes))]
            else:
                pkg = self.owner_of(e) if e.part == -1 else None
                nodes = core.geo_nodes(self.data(e))
                if pkg is not None:                       # a weapon etc. shown on its own, posed on the body
                    _, nodes, clips, _, _ = self._attach_info(pkg, e, entries)
                    clip, tm = core.pick_clip(clips)
                    anc = next((s['_anchor'] for s in model['meshes'] if '_anchor' in s), None)
                    H = core.world(self._body_nodes(pkg))[anc] if anc is not None else np.eye(4)
                    part_skel[0] = add(nodes, core.pose_locals(nodes, clip, tm), attach=('matrix', H))
                    sk = skels[-1]
                    sk['world'] = {j: H @ sk['world'][j] for j in sk['world']}
                    sk['inv'] = [np.linalg.inv(sk['world'][j]) for j in range(len(nodes))]
                else:
                    part_skel[0] = add(nodes, bind(nodes))
        except Exception:
            return None
        # per submesh: joint indices (into its skeleton) and weights
        verts = {}
        for idx, s in enumerate(model['meshes']):
            if 'position' not in s or not len(s['position']):
                continue
            si = part_skel.get(s.get('part', 0))
            if si is None:
                continue
            nodes = skels[si]['nodes']; n = len(s['position'])
            if 'joints' in s and s.get('palette'):
                pal = np.array([p if 0 <= p < len(nodes) else 0 for p in s['palette']] + [0])
                J = pal[np.minimum(s['joints'].astype(int), len(pal) - 1)]
                w = s['weights'].astype(np.float32)
                w = w / np.maximum(w.sum(1, keepdims=True), 1e-9)
                J = np.where(w > 0, J, 0)
            else:
                mid = s['mesh'] % 1000
                node = next((i for i, nd in enumerate(nodes) for g, _ in nd['meshes'] if g == mid), 0)
                J = np.zeros((n, 4), np.int64); J[:, 0] = node
                w = np.zeros((n, 4), np.float32); w[:, 0] = 1
            verts[idx] = (si, J.astype(np.uint16), w.astype(np.float32))
        if not verts:
            return None
        # animations (Blender actions): clips stored with the model drive its main skeleton; each attachment's own
        # clips drive the attachment skeleton
        anims = []
        if animations and skels:
            owner = e.clone(part=-1, attach=()) if e.part >= 0 else e
            main = 0
            if e.part == -1 and self.owner_of(e) is not None:
                clips_main = [(c['name'], c) for c in self._attach_info(self.owner_of(e), e, entries)[2]]
            else:
                clips_main = [(c['name'], c) for _, c in self.clips_for(owner, entries)]
            seen = set()
            for nm, c in clips_main:
                ch = core.clip_channels(skels[main]['nodes'], c)
                if ch and nm not in seen:
                    seen.add(nm)
                    anims.append(dict(name=nm, channels=[(main,) + x for x in ch]))
            if e.part == -2:
                allp = self.parts(e)
                used = [k for k, u in enumerate(self.assembly_use(e)) if u]
                for i, k in enumerate(used):
                    if k >= e.nparts and i in part_skel:
                        for c in self._attach_info(e, allp[k], entries)[2]:
                            ch = core.clip_channels(skels[part_skel[i]]['nodes'], c)
                            if ch:
                                anims.append(dict(name='att%04d_%s' % (allp[k].index, c['name']),
                                                  channels=[(part_skel[i],) + x for x in ch]))
        return dict(skeletons=skels, part_skel=part_skel, verts=verts, animations=anims)

    def anim_for(self, a, entries):
        """The animation file stored after model a (before the next model), or None."""
        if not entries:
            return None
        plain = [x for x in entries if x.part < 0 and x.dat == a.dat]
        try:
            i = next(k for k, x in enumerate(plain) if x.off == a.off)
        except StopIteration:
            return None
        for x in plain[i + 1:i + 8]:
            if x.cls == 'model':
                break
            if x.cls == 'other' and b'ANIMWIIU' in self.data(x)[:0x100]:
                return x
        return None

    def _attach_info(self, pkg, a, entries):
        """(model, nodes, clips, tag, anchor bone) for attachment a of package pkg."""
        def make():
            an = self.anim_for(a, entries)
            clips = core.parse_anim(self.data(an)) if an is not None else []
            nodes = core.geo_nodes(self.data(a))
            body = self._body_nodes(pkg)
            m = self.raw_model(a)
            tag = core.attachment_tag(clips)
            bone = core.attachment_anchor(nodes, m, clips, body)
            if tag in ('RH', 'LH') and bone is not None:
                try:
                    bone = core.hand_or_forearm(m, nodes, clips, body, self.raw_model(pkg.clone(part=0, attach=())), bone)
                except Exception:
                    pass
            return m, nodes, clips, tag, bone
        return self._get(('ainfo', pkg.dat, pkg.off) + a.key, make)

    def _body_nodes(self, pkg):
        return self._get(('bnodes', pkg.dat, pkg.off), lambda: core.geo_nodes(self.data(pkg.clone(part=0, attach=()))))

    def posed_attachment(self, pkg, a, entries):
        """Attachment model moved onto the body and posed as in-game (see core.pose_attachment).
        Attachments with no fitting pieces borrow the bone of a sibling attachment with the same mount tag."""
        def make():
            m = self.raw_model(a)
            try:
                m, nodes, clips, tag, bone = self._attach_info(pkg, a, entries)
                if bone is None and tag:
                    for b in pkg.attach:
                        if b.key != a.key:
                            _, _, _, t2, b2 = self._attach_info(pkg, b, entries)
                            if t2 == tag and b2 is not None:
                                bone = b2
                                break
                body = self._body_nodes(pkg)
                if bone is not None:
                    H = core.world(body)[bone]
                elif body:
                    # nothing to anchor to (vehicle weapons): the body's origin, or the first bone that seats it
                    H = np.eye(4)
                    try:
                        bm = self.model(pkg.clone(part=0, attach=()))
                        P = np.concatenate([s['position'][:, :3] for s in bm['meshes']])
                        vb = core.vehicle_anchor(m, nodes, clips, body, (P.min(0), P.max(0)))
                        if vb is not None:
                            H = core.world(body)[vb]
                            bone = vb
                    except Exception:
                        pass
                else:
                    H = None
                out = core.pose_attachment(m, nodes, clips, H)
                for s in out['meshes']:
                    s['_anchor'] = bone         # body bone it follows during body animations (None: still)
                return out
            except Exception as ex:
                return dict(m, errors=m['errors'] + ['attachment pose failed: %s' % ex])
        return self._get(('pose', pkg.dat, pkg.off) + a.key, make)

    def assemble(self, e, entries=None):
        """Merge the selected parts of a package: returns (model, textures, texinfos, used parts). Cached."""
        use = self.assembly_use(e)
        return self._get(('asm',) + e.key + tuple(bool(u) for u in use), lambda: self._assemble(e, entries, use))

    def _assemble(self, e, entries, use):
        ps = [p for p, u in zip(self.parts(e), use) if u]
        models, offs, texs, infos, seen = [], [], [], [], {}
        for p in ps:
            src = self.texture_set(p, entries)[0]
            k = src.key if src is not None else None
            if k not in seen:
                seen[k] = len(texs)
                texs += self.textures(src); infos += self.texture_infos(src)
            m = self.model(p)
            if p.part == -1 and p.key != e.key:           # an attachment: pose it on the body
                m = self.posed_attachment(e, p, entries)
            models.append(m); offs.append(seen[k])
        return core.merge_models(models, offs), texs, infos, [k for k, u in enumerate(use) if u]

    def _sets(self, e):
        return self._get(('ts', e.dat, e.off), lambda: core.texture_sets(self.whole(e)))

    def _from_package(self, pkg, e):
        """Texture source inside package pkg for model entry e (picks the per-part set)."""
        sets = self._sets(pkg)
        if not sets:
            return None
        if e.part < 0 and len(sets) == 1 and pkg is not e:
            return pkg
        i = 0
        if e.part >= 0:
            per = self._get(('pts', pkg.dat, pkg.off, e.nparts),
                            lambda: core.part_texture_sets(self.whole(pkg), e.nparts))
            if per is not None and e.part < len(per):
                i = per[e.part]                 # one texture entry per part; empty = same as previous
            elif len(sets) == e.nparts:
                i = e.part
        return pkg.clone(part=-1, tset=i, cls='textures')

    def texture_set(self, e, entries=None):
        """Returns (texture source entry or None, how) for a model entry."""
        if e.part == -2:
            return None, 'assembled'
        ov = self.overrides.get(e.key)
        if ov is not None:
            return (ov if ov != 'none' else None), 'manual'
        if e.textures is not None and e.textures.cls == 'textures':
            src = self._from_package(e.textures, e)
            if src is not None:
                return src, 'paired'
        src = self._from_package(e, e)          # textures stored inside the model package itself
        if src is not None:
            return src, 'embedded'
        m = self.model(e)
        need = [l['texture'] for x in m['materials'] for l in x['layers']]
        if not need or not entries:
            return None, 'none'
        want = max(need) + 1
        # a mirrored twin (same animation with L/R swapped, e.g. a left-hand copy of a right-hand weapon)
        # shares its textures
        twin = self._mirror_twin(e, entries)
        if twin is not None:
            src, how = self.texture_set(twin, entries)
            if src is not None and len(self.texture_infos(src)) >= want:
                return src, 'mirror of %04d' % twin.index
        pos = None
        for i, x in enumerate(entries):
            if x.key == e.key:
                pos = i; break
        if pos is None:
            return None, 'none'
        for j in range(pos - 1, max(-1, pos - 40), -1):
            c = entries[j]
            if c.dat != e.dat:
                break
            if c.cls == 'textures' and len(self.texture_infos(c)) >= want:
                return c, 'guessed'
        return None, 'none'

    def _clip_names(self, e, entries):
        def make():
            an = self.anim_for(e, entries)
            try:
                return [c['name'] for c in core.parse_anim(self.data(an))] if an is not None else []
            except Exception:
                return []
        return self._get(('clipnames',) + e.key, make)

    def _mirror_twin(self, e, entries):
        """Model in the same archive whose animation is e's with left/right swapped (has its own textures)."""
        import re
        if e.part != -1 or not entries:
            return None
        names = self._clip_names(e, entries)
        if not names:
            return None
        swap = lambda n: re.sub(r'_([LR])([A-Z])_', lambda m: '_%s%s_' % ('R' if m.group(1) == 'L' else 'L', m.group(2)), n)
        want = swap(names[0])
        if want == names[0]:
            return None
        for x in entries:
            if x.cls == 'model' and x.part == -1 and x.dat == e.dat and x.key != e.key and \
                    x.textures is not None and x.textures.cls == 'textures':
                cn = self._clip_names(x, entries)
                if cn and cn[0] == want:
                    return x
        return None

    def texture_infos(self, e):
        if e is None or e.cls != 'textures':
            return []
        if e.tset >= 0:
            return self._get(('ti',) + e.key, lambda: self._sets(e)[e.tset])
        return self._get(('ti',) + e.key, lambda: core.find_textures(self.whole(e)))

    def textures(self, e):
        if e is None or e.cls != 'textures':
            return []
        return self._get(('t',) + e.key, lambda: [core.decode_texture(t) for t in self.texture_infos(e)])


CACHE_PATH = os.path.join(os.path.expanduser('~'), '.tfp_tool_cache.json')


def index_with_cache(dats, table=None, progress=None):
    """index_game, remembering per-package model counts between runs (they need decompression to find)."""
    import json
    try:
        with open(CACHE_PATH) as f:
            cache = json.load(f)
    except Exception:
        cache = {}
    n0 = len(cache)
    ents = core.index_game(dats, progress=progress, table=table, part_cache=cache)
    if len(cache) != n0:
        try:
            with open(CACHE_PATH, 'w') as f:
                json.dump(cache, f)
        except Exception:
            pass
    return ents


def find_table(folder):
    """Read the archive file table from the game executable (code/*.rpx), if present."""
    for p in glob.glob(os.path.join(folder, '**', '*.rpx'), recursive=True):
        try:
            t = core.read_rpx_file_table(p)
            if t:
                return t
        except Exception:
            pass
    return None


def export_entry(loader, e, outdir, fmt_glb=True, fmt_obj=False, flip_v=False, raw=False, textures_png=True, entries=None,
                 variants=False, model_textures=False, skydome=True, tex_dir=None, skeleton=True, animations=True):
    """Export one entry. Returns list of written files."""
    os.makedirs(outdir, exist_ok=True)
    written = []
    name = e.name
    if raw:
        p = os.path.join(outdir, name + '.bin')
        with open(p, 'wb') as f:
            f.write(loader.data(e))
        written.append(p)
    if e.cls == 'model':
        m = loader.model(e)
        if not skydome:
            m = core.strip_groups(m, core.skydome_groups(m))
        if variants is not True:          # True: keep every variant; else the groups listed replace their original
            m = core.strip_variants(m, keep=variants or ())
        if e.part == -2:
            texs = loader.assemble(e, entries)[1]
            srcs = [loader.texture_set(p, entries)[0] for p, u in zip(loader.parts(e), loader.assembly_use(e)) if u]
        else:
            srcs = [loader.texture_set(e, entries)[0]]
            texs = loader.textures(srcs[0])
        if model_textures:
            for ts in {s.key: s for s in srcs if s is not None}.values():
                written += _write_pngs(loader, ts, outdir)
        part_names = None
        if e.part == -2:            # name each part / attachment object after what it is
            allp = loader.parts(e)
            used = [k for k, u in enumerate(loader.assembly_use(e)) if u]
            part_names = {i: (('part%d' % k) if k < e.nparts else ('attachment%04d' % allp[k].index))
                          for i, k in enumerate(used)}
        if fmt_glb:
            p = os.path.join(outdir, name + '.glb')
            rig = loader.rig_for(e, m, entries, animations=animations) if skeleton else None
            exp.write_glb(p, m, texs, flip_v=flip_v, name=name, part_names=part_names, rig=rig); written.append(p)
        if fmt_obj:
            sub = os.path.join(outdir, name + '_obj'); os.makedirs(sub, exist_ok=True)
            p = os.path.join(sub, name + '.obj')
            exp.write_obj(p, m, texs, flip_v=not flip_v, name=name, part_names=part_names); written.append(p)
    elif e.cls == 'textures' and textures_png:
        # texture sets exported directly all go into one shared folder (no per-set subfolders)
        written += _write_pngs(loader, e, tex_dir or os.path.join(outdir, 'textures'), flat=True)
    return written


def _write_pngs(loader, ts, outdir, flat=False):
    """Write a texture set as PNGs into <outdir>/<set name>_textures (or straight into outdir if flat),
    once per set. Each texture is written at full resolution (mip level 0)."""
    sub = outdir if flat else os.path.join(outdir, ts.name + '_textures')
    os.makedirs(sub, exist_ok=True)
    out = []
    for info, t in zip(loader.texture_infos(ts), loader.textures(ts)):
        p = os.path.join(sub, '%s_%d_%s_%dx%d.png' % (ts.name, info['index'], core.fmt_name(info['format']),
                                                      info['width'], info['height']))
        if not os.path.exists(p):
            Image.fromarray(t).save(p)
        out.append(p)
    return out


# ------------------------------------------------------------------ CLI
def cli_extract(argv):
    import argparse
    ap = argparse.ArgumentParser(prog='tfp_tool.py extract')
    ap.add_argument('game'); ap.add_argument('out', nargs='?', default=EXPORT_DIR,
                                             help='output folder (default: .exported next to this script)')
    ap.add_argument('--obj', action='store_true', help='also write OBJ/MTL/PNG')
    ap.add_argument('--no-glb', action='store_true')
    ap.add_argument('--textures', action='store_true', help='also export every texture set as PNG')
    ap.add_argument('--raw', action='store_true', help='also dump every decompressed file (.bin)')
    ap.add_argument('--placeholders', action='store_true', help='include the dummy cube models')
    ap.add_argument('--flip-v', action='store_true')
    ap.add_argument('--variants', action='store_true', help='keep all variant mesh groups')
    ap.add_argument('--no-skydome', action='store_true', help='drop detected sky domes around levels')
    ap.add_argument('--no-skeleton', action='store_true', help='export static meshes without skeletons/weights')
    ap.add_argument('--no-animations', action='store_true', help='leave animation clips out of GLB files')
    a = ap.parse_args(argv)
    dats = find_dats(a.game)
    if not dats:
        print('No .dat archives found under', a.game); return 1
    print('Indexing %d archives...' % len(dats))
    table = find_table(a.game)
    print('File table from executable:', 'yes' if table else 'no (scanning archives)')
    ents = index_with_cache(dats, table=table)
    loader = Loader()
    loader.entries = ents
    todo = []
    for e in ents:
        if e.cls == 'model' and (a.placeholders or not is_placeholder(e)):
            todo.append(e)
        elif e.cls == 'textures' and a.textures:
            todo.append(e)
        elif a.raw:
            todo.append(e)
    print('%d items to export' % len(todo))
    for i, e in enumerate(todo):
        d = os.path.join(a.out, os.path.splitext(os.path.basename(e.dat))[0])
        try:
            export_entry(loader, e, d, fmt_glb=not a.no_glb, fmt_obj=a.obj, flip_v=a.flip_v, raw=a.raw,
                         textures_png=e.cls == 'textures', entries=ents,
                         variants=a.variants, skydome=not a.no_skydome, tex_dir=os.path.join(a.out, 'textures'),
                         skeleton=not a.no_skeleton, animations=not a.no_animations)
            print('[%d/%d] %s (%s)' % (i + 1, len(todo), e.name, e.cls))
        except Exception as ex:
            print('[%d/%d] %s FAILED: %s' % (i + 1, len(todo), e.name, ex))
    return 0


# ------------------------------------------------------------------ GUI
def run_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from PIL import ImageTk

    root = tk.Tk()
    root.title('TF Prime (Wii U) Model Viewer / Extractor')
    root.geometry('1280x800')
    loader = Loader()
    state = dict(entries=[], items={}, sel=None, model=None, texs=[], texinfo=[], yaw=0.6, pitch=-0.25,
                 zoom=1.0, pan=[0, 0], drag=None, render_job=None, task=0)
    msgq = queue.Queue()

    # ---- toolbar
    bar = ttk.Frame(root, padding=4); bar.pack(side='top', fill='x')
    ttk.Button(bar, text='Open Game Folder', command=lambda: open_folder()).pack(side='left')
    ttk.Label(bar, text='  Show:').pack(side='left')
    filt = tk.StringVar(value='Models')
    cb = ttk.Combobox(bar, textvariable=filt, state='readonly', width=24,
                      values=['Models', 'Textures', 'All'])
    cb.pack(side='left'); cb.bind('<<ComboboxSelected>>', lambda ev: fill_tree())
    hide_dummies = tk.BooleanVar(value=True)
    ttk.Label(bar, text='  List:').pack(side='left')
    ttk.Checkbutton(bar, text='Hide Dummies', variable=hide_dummies,
                    command=lambda: fill_tree()).pack(side='left', padx=(4, 6))
    hide_parts = tk.BooleanVar(value=True)
    ttk.Checkbutton(bar, text='Hide Parts', variable=hide_parts,
                    command=lambda: fill_tree()).pack(side='left')

    class _Off:             # V flip is no longer a GUI option; previews/exports use the game's orientation
        @staticmethod
        def get():
            return False
    flip = _Off()
    textured = tk.BooleanVar(value=True)
    ttk.Separator(bar, orient='vertical').pack(side='left', fill='y', padx=6)
    ttk.Label(bar, text='Viewer:').pack(side='left', padx=(0, 4))
    ttk.Checkbutton(bar, text='Textures', variable=textured, command=lambda: schedule_render()).pack(side='left')
    wire = tk.BooleanVar(value=False)
    variant_keep = {}       # model key -> set of variant mesh groups switched on (Variants section)
    variant_vars = {}       # mesh group -> BooleanVar of its Variants checkbox (current model)
    show_sky = tk.BooleanVar(value=True)       # Visibility > Skydome (on by default); kept across model changes
    ttk.Checkbutton(bar, text='Wireframe', variable=wire, command=lambda: schedule_render()).pack(side='left', padx=6)
    wasd = tk.BooleanVar(value=False)
    wasd_cb = ttk.Checkbutton(bar, text='WASD Camera', variable=wasd, command=lambda: on_wasd())
    wasd_cb.pack(side='left', padx=(6, 0))
    ttk.Separator(bar, orient='vertical').pack(side='left', fill='y', padx=6)
    ttk.Label(bar, text='Export:').pack(side='left')
    exp_skel = tk.BooleanVar(value=True)
    exp_anim = tk.BooleanVar(value=False)
    ttk.Checkbutton(bar, text='Skeleton', variable=exp_skel,
                    command=lambda: anim_cb.state(['!disabled'] if exp_skel.get() else ['disabled'])
                    ).pack(side='left', padx=(4, 0))
    anim_cb = ttk.Checkbutton(bar, text='Animations', variable=exp_anim)   # animations need the skeleton
    anim_cb.pack(side='left', padx=(6, 0))
    ttk.Button(bar, text='Export', command=lambda: export_selected()).pack(side='left', padx=4)
    ttk.Button(bar, text='Exports Folder', command=lambda: open_folder_in_os(EXPORT_DIR)).pack(side='left')

    status = tk.StringVar(value='Open the game folder (the one with code/, content/, meta/).')
    sb = ttk.Frame(root, padding=2); sb.pack(side='bottom', fill='x')
    ttk.Label(sb, textvariable=status).pack(side='left')

    pw = ttk.PanedWindow(root, orient='horizontal'); pw.pack(fill='both', expand=True)
    left = ttk.Frame(pw); pw.add(left, weight=1)
    tree = ttk.Treeview(left, columns=('type', 'size'), selectmode='extended')
    tree.heading('#0', text='File'); tree.heading('type', text='Type'); tree.heading('size', text='Size')
    tree.column('#0', width=220); tree.column('type', width=90); tree.column('size', width=80, anchor='e')
    ys = ttk.Scrollbar(left, orient='vertical', command=tree.yview); tree.configure(yscrollcommand=ys.set)
    tree.pack(side='left', fill='both', expand=True); ys.pack(side='right', fill='y')

    nb = ttk.Notebook(pw); pw.add(nb, weight=3)
    vf = ttk.Frame(nb); nb.add(vf, text='3D preview')
    glview = None
    if HAVE_GL:
        try:
            glview = tfp_glview.GLModelView(vf)
            glview.pack(fill='both', expand=True)
        except Exception as ex:
            print('OpenGL viewer unavailable, using software renderer:', ex)
            glview = None
    canvas = tk.Canvas(vf, bg='#282c34', highlightthickness=0)
    if glview is None:
        canvas.pack(fill='both', expand=True)
        wasd_cb.state(['disabled'])     # first-person camera needs the GPU viewer

    def on_wasd():
        if glview is None:
            return
        glview.set_fly(wasd.get())

    partsbar = ttk.Frame(vf)        # part toggles for assembled packages
    part_vars = []
    # ---- animation player
    animbar = ttk.Frame(vf)
    anim = dict(clips=[], clip=None, t=0.0, playing=False, animator=None, job=None, last=None, scrubbing=False)
    ttk.Label(animbar, text='Animation:').pack(side='left', padx=(4, 2))
    anim_var = tk.StringVar(value='(none)')
    anim_combo = ttk.Combobox(animbar, textvariable=anim_var, state='readonly', width=42)
    anim_combo.pack(side='left')
    play_btn = ttk.Button(animbar, text='Play', width=6, command=lambda: anim_toggle())
    play_btn.pack(side='left', padx=4)
    anim_pos = tk.DoubleVar(value=0.0)
    anim_scale = ttk.Scale(animbar, from_=0.0, to=1.0, variable=anim_pos, command=lambda v: anim_scrub(float(v)))
    anim_scale.pack(side='left', fill='x', expand=True, padx=4)
    anim_time = ttk.Label(animbar, text='0.00 / 0.00 s', width=14)
    anim_time.pack(side='left')
    ttk.Label(animbar, text='Speed').pack(side='left', padx=(6, 2))
    speed_var = tk.StringVar(value='1x')
    ttk.Combobox(animbar, textvariable=speed_var, state='readonly', width=5,
                 values=['0.25x', '0.5x', '1x', '2x']).pack(side='left', padx=(0, 4))
    tf = ttk.Frame(nb); nb.add(tf, text='Textures')
    tleft = ttk.Frame(tf); tleft.pack(side='left', fill='y')
    ttk.Label(tleft, text='Texture set for this model:').pack(anchor='w')
    tsvar = tk.StringVar()
    tscombo = ttk.Combobox(tleft, textvariable=tsvar, state='readonly', width=36)
    tscombo.pack(fill='x')
    tlist = tk.Listbox(tleft, width=38); tlist.pack(fill='both', expand=True)
    tcanvas = tk.Canvas(tf, bg='#3a3f4b', highlightthickness=0); tcanvas.pack(side='left', fill='both', expand=True)
    inf = ttk.Frame(nb); nb.add(inf, text='Info')
    info = tk.Text(inf, wrap='none', font=('Courier', 10)); info.pack(fill='both', expand=True)

    # ---- helpers
    def set_status(s, p=None):
        status.set(s)

    def pump():
        try:
            while True:
                fn = msgq.get_nowait(); fn()
        except queue.Empty:
            pass
        root.after(50, pump)

    def ui(fn):
        msgq.put(fn)

    task_local = threading.local()

    def cancelled():
        """True inside a background task that a newer task (or Cancel) has replaced."""
        return getattr(task_local, 'id', state['task']) != state['task']

    class _Cancelled(Exception):
        pass

    def index_progress(f):
        if cancelled():
            raise _Cancelled()              # stop indexing early; the result would be discarded anyway
        ui(lambda: set_status('Indexing archives… %d%%' % round(f * 100)))

    def bg(work, done=None, label='Working…'):
        """Run work() off the UI thread, then done(result) on it. Starting a task cancels the running one:
        it stops at its next check (exports stop between files) and its result is discarded."""
        state['task'] += 1
        tid = state['task']
        set_status(label)

        def run():
            task_local.id = tid
            try:
                r = work()
                ui(lambda: done(r) if done and tid == state['task'] else None)
            except Exception as ex:
                msg = '%s\n\n%s' % (ex, traceback.format_exc()[-1500:]); short = 'Error: %s' % ex
                ui(lambda: (messagebox.showerror('Error', msg), set_status(short)) if tid == state['task'] else None)
        threading.Thread(target=run, daemon=True).start()

    def open_folder():
        d = filedialog.askdirectory(title='Select the Transformers Prime game folder')
        if not d:
            return
        dats = find_dats(d)
        if not dats:
            messagebox.showerror('Not found', 'No .dat archives found under:\n' + d); return

        def work():
            table = find_table(d)
            return index_with_cache(dats, table=table, progress=index_progress)

        def done(ents):
            state['entries'] = ents
            loader.entries = ents
            loader.cache.clear(); loader.order.clear(); loader.sizes.clear(); loader.total = 0
            n = sum(1 for e in ents if e.cls == 'model' and not is_placeholder(e))
            set_status('%d archives, %d files, %d models, %d texture sets' % (
                len(dats), len(ents), n, sum(e.cls == 'textures' for e in ents)), 1)
            fill_tree()
        bg(work, done, 'Indexing archives…')

    def shown(e):
        if hide_dummies.get() and (is_placeholder(e) or e.cls == 'empty'):
            return False
        if hide_parts.get() and e.part >= 0:     # individual parts of a package with an assembled entry
            return False
        f = filt.get()
        if f == 'Models':
            return e.cls == 'model'
        if f == 'Textures':
            return e.cls == 'textures'
        return True

    def fill_tree():
        tree.delete(*tree.get_children())
        state['items'] = {}
        groups = {}
        for e in state['entries']:
            if not shown(e):
                continue
            dat = os.path.basename(e.dat)
            if dat not in groups:
                groups[dat] = tree.insert('', 'end', text=dat, open=False, values=('archive', ''))
            lbl = '%04d  @%08x' % (e.index, e.off) + ('  part %d' % e.part if e.part >= 0 else
                                                      '  assembled' if e.part == -2 else '')
            iid = tree.insert(groups[dat], 'end', text=lbl,
                              values=(e.cls + (' (dummy)' if is_placeholder(e) else ''), _sz(e.size)))
            state['items'][iid] = e
        if len(groups) == 1:
            for g in groups.values():
                tree.item(g, open=True)

    def _sz(n):
        return '%.1f MB' % (n / 1048576) if n >= 1048576 else '%.1f KB' % (n / 1024)

    def selected_entries():
        out = []
        for iid in tree.selection():
            if iid in state['items']:
                out.append(state['items'][iid])
            else:
                out += [state['items'][c] for c in tree.get_children(iid) if c in state['items']]
        return out

    def fill_partsbar(r):
        """Toggles for an assembled package, in three sections:
        Parts (real model pieces), Attachments, Variants (0 = original, then alternates), Visibility."""
        for w in partsbar.winfo_children():
            w.destroy()
        part_vars.clear()           # list of (BooleanVar, [part indices it controls])
        e = r['e']
        has_sky = bool(r.get('sky'))
        vgroups = r.get('variants') or []
        keep = variant_keep.setdefault(e.key, set())

        variant_vars.clear()

        def add_variants():
            # per original piece: 0 = the original, then its variants numbered 1..n; they act like radio buttons
            vmap = state.get('vmap') or {}
            bases = sorted(set(vmap.values()))
            for bi, base in enumerate(bases):
                if bi:
                    ttk.Label(partsbar, text='|').pack(side='left', padx=2)
                members = [base] + sorted(g for g in vgroups if vmap.get(g) == base)
                for n, g in enumerate(members):
                    on = (g in keep) if g != base else not any(vmap.get(x) == base for x in keep)
                    v = tk.BooleanVar(value=on)
                    variant_vars[g] = v
                    ttk.Checkbutton(partsbar, text=str(n), variable=v,
                                    command=lambda g=g: on_variant(g)).pack(side='left')

        def add_visibility():
            if has_sky:
                ttk.Checkbutton(partsbar, text='Skydome', variable=show_sky, command=on_sky).pack(side='left')

        def section(title, first=False):
            if not first:
                ttk.Separator(partsbar, orient='vertical').pack(side='left', fill='y', padx=4)
            ttk.Label(partsbar, text=title).pack(side='left', padx=(4 if first else 0, 2))

        if 'used' not in r:
            if not has_sky and not vgroups:
                partsbar.pack_forget(); return
            first = True
            if vgroups:
                section('Variants:', first); add_variants(); first = False
            if has_sky:
                section('Visibility:', first); add_visibility()
            partsbar.pack(side='top', fill='x', before=animbar if animbar.winfo_ismapped() else (glview if glview is not None else canvas))
            return
        allp = loader.parts(e)
        kinds = r['proxy']
        state['part_kinds'] = kinds
        real = [k for k in range(e.nparts) if not kinds[k]]
        groups = [(name, [k for k in range(e.nparts) if kinds[k] == kind]) for name, kind in (('Effects', 'effect'),)]
        groups = [(n, ks) for n, ks in groups if ks]
        proxies = [k for k in range(e.nparts) if kinds[k] == 'proxy']
        att = list(range(e.nparts, len(allp)))

        def add(label, idxs, proxy=False):
            v = tk.BooleanVar(value=any(k in r['used'] for k in idxs))
            part_vars.append((v, idxs))
            ttk.Checkbutton(partsbar, text=label, variable=v,
                            command=lambda: on_parts(proxy_toggled=proxy if proxy else False,
                                                     other_on=not proxy and v.get())).pack(side='left')

        section('Parts:', first=True)
        for k in real:
            add(str(k), [k])
        if att:
            section('Attachments:')
            for k in att:
                add('%04d' % allp[k].index, [k])
        if vgroups:
            section('Variants:')
            add_variants()
        if groups or has_sky:
            section('Visibility:')
            add_visibility()
            for name, ks in groups:
                add(name, ks)
        if proxies:
            ttk.Separator(partsbar, orient='vertical').pack(side='left', fill='y', padx=4)
            add('Proxy', proxies, proxy=True)
        ttk.Button(partsbar, text='Default', command=lambda: on_parts(default=True)).pack(side='left', padx=6)
        partsbar.pack(side='top', fill='x', before=animbar if animbar.winfo_ismapped() else (glview if glview is not None else canvas))

    vmap_known = {}             # assembled package key -> every variant group seen in it ({group: base})
    proxy_saved = {}            # package -> what was shown before Proxy was switched on (restored when it goes off)

    def on_parts(default=False, proxy_toggled=False, other_on=False):
        """Apply the part toggles. Proxy is exclusive: switching it on hides everything else (and remembers
        what was shown), switching it off brings that back, and switching anything else on turns Proxy off."""
        e = state['sel']
        if e is None or e.part != -2:
            return
        key = (e.dat, e.off)
        kinds = state.get('part_kinds') or []
        is_proxy = [k < e.nparts and k < len(kinds) and kinds[k] == 'proxy' for k in range(e.nparts + len(e.attach))]
        if default:
            loader.assembly.pop(key, None); proxy_saved.pop(key, None)
        else:
            use = [False] * (e.nparts + len(e.attach))
            for v, idxs in part_vars:
                for k in idxs:
                    use[k] = v.get()
            if proxy_toggled:
                if any(use[k] for k in range(len(use)) if is_proxy[k]):        # Proxy on: only the proxy
                    proxy_saved[key] = [u and not p for u, p in zip(use, is_proxy)]
                    use = [p for p in is_proxy]
                else:                                                           # Proxy off: restore
                    prev = proxy_saved.pop(key, None) or loader.assembly_use_default(e)
                    use = [u and not p for u, p in zip(prev, is_proxy)]
            elif other_on and any(use[k] for k in range(len(use)) if is_proxy[k]):
                prev = proxy_saved.pop(key, None) or [False] * len(use)       # something else on: Proxy off,
                use = [(u or s) and not p for u, s, p in zip(use, prev, is_proxy)]  # earlier view back + new item
            loader.assembly[key] = use
        state['sel'] = None; state['keep_view'] = True
        on_select()

    def on_select(ev=None):
        sel = [state['items'][i] for i in tree.selection() if i in state['items']]
        if not sel or sel[0] is state['sel']:
            return
        e = sel[0]; state['sel'] = e

        def work():
            res = dict(e=e)
            if e.cls == 'model' and e.part == -2:
                m, texs, infos, used = loader.assemble(e, state['entries'])
                res.update(model=m, texs=texs, texinfo=infos, texset=(None, 'assembled'), used=used,
                           proxy=[('proxy' if not core.is_textured(loader.model(p)) else
                                   'effect' if k < e.nparts and core.is_effect(loader.model(p)) else '')
                                  for k, p in enumerate(loader.parts(e))])
            elif e.cls == 'model':
                res['model'] = loader.model(e)
                ts, how = loader.texture_set(e, state['entries'])
                res['texset'] = (ts, how)
                res['texinfo'] = loader.texture_infos(ts)
                res['texs'] = loader.textures(ts)
            elif e.cls == 'textures':
                res['texinfo'] = loader.texture_infos(e); res['texs'] = loader.textures(e)
            else:
                res['head'] = loader.data(e)[:4096]
            if res.get('model') is not None:
                try:   # anti-flicker data for the default view (variants hidden), off the UI thread
                    core.annotate_face_flags(core.strip_variants(res['model']))
                except Exception:
                    pass
                try:
                    res['clips'] = loader.clips_for(e.clone(part=-1, attach=()) if e.part >= 0 else e,
                                                    state['entries'])
                except Exception:
                    res['clips'] = []
            return res

        def done(r):
            if state['sel'] is not r['e']:
                return
            state['full_model'] = r.get('model')
            state['sky'] = core.skydome_groups(r['model']) if r.get('model') else set()
            vmap = core.variant_map(r['model']) if r.get('model') else {}
            if r['e'].part == -2:       # remember a package's variants, so the section stays while their part is
                known = vmap_known.setdefault(r['e'].key, {})       # hidden (e.g. with Proxy on)
                known.update(vmap)
                vmap = dict(known)
            state['vmap'] = vmap
            r['variants'] = sorted(state['vmap'])
            r['sky'] = state['sky']
            state['model'] = view_model(r.get('model')); state['texs'] = r.get('texs', []); state['texinfo'] = r.get('texinfo', [])
            keep = state.pop('keep_view', False)
            if not keep:
                state.update(yaw=0.6, pitch=-0.25, zoom=1.0, pan=[0, 0])
            show_info(r)
            fill_partsbar(r)
            fill_texset_combo(r)
            fill_textures()
            if state['model'] is not None:
                nb.select(0)
                if glview is not None:
                    glview.set_model(state['model'], state['texs'], flip_v=flip.get(), keep_view=keep)
                    glview.set_options(use_textures=textured.get(), wire=wire.get())
                else:
                    schedule_render()
                fill_animbar(r)
                anim_rebuild()
            elif state['texs']:
                anim_stop(); animbar.pack_forget()
                nb.select(1); canvas.delete('all')
                if glview is not None:
                    glview.set_model(None, [])
            else:
                anim_stop(); animbar.pack_forget()
                nb.select(2); canvas.delete('all')
                if glview is not None:
                    glview.set_model(None, [])
            set_status('%s  (%s, %s)' % (e.name, e.cls, _sz(e.size)), 1)
        bg(work, done, 'Loading %s…' % e.name)

    # ---------------------------------------------------------------- animation playback
    def fill_animbar(r):
        clips = r.get('clips') or []
        anim['clips'] = clips
        labels = [l for l, _ in clips]
        anim_combo['values'] = ['(none)'] + labels
        if anim_var.get() not in labels:            # keep the clip when the same model is rebuilt
            anim_stop(); anim_var.set('(none)'); anim['clip'] = None; anim['t'] = 0.0
        if clips and state.get('model') is not None:
            animbar.pack(side='top', fill='x', before=glview if glview is not None else canvas)
        else:
            animbar.pack_forget()

    def anim_rebuild():
        """(Re)create the animator for the model on screen and re-apply the current pose."""
        anim['animator'] = None
        state.pop('anim_model', None)
        e = state.get('sel')
        if e is None or state.get('model') is None or not anim['clips']:
            return
        try:
            anim['animator'] = loader.animator(e, state['model'])
        except Exception as ex:
            set_status('Animation unavailable: %s' % ex)
            return
        if anim['clip'] is not None:
            anim_apply()

    def anim_select(ev=None):
        lab = anim_var.get()
        c = dict(anim['clips']).get(lab)
        anim['clip'] = c; anim['t'] = 0.0
        anim_scale.configure(to=max(c['duration'], 1e-3) if c else 1.0)
        if c is None:
            anim_stop()
        anim_apply()

    def anim_apply():
        a, c = anim['animator'], anim['clip']
        if a is None:
            return
        upd = a.pose(c, anim['t'])
        dur = c['duration'] if c else 0.0
        anim['scrubbing'] = True; anim_pos.set(anim['t']); anim['scrubbing'] = False
        anim_time.config(text='%.2f / %.2f s' % (anim['t'], dur))
        if glview is not None:
            glview.update_geometry(upd)
        else:
            m = state['model']
            meshes = list(m['meshes'])
            for idx, pos, nrm in upd:
                s = dict(meshes[idx]); P = s['position'].copy(); P[:, :3] = pos; s['position'] = P
                if nrm is not None and 'normal' in s:
                    N = s['normal'].copy(); N[:, :3] = nrm; s['normal'] = N
                meshes[idx] = s
            state['anim_model'] = dict(m, meshes=meshes)
            schedule_render(fast=anim['playing'])

    def anim_scrub(v):
        if anim['scrubbing'] or anim['clip'] is None:
            return
        anim['t'] = v
        anim_apply()

    def anim_toggle():
        if anim['playing']:
            anim_stop(); return
        if anim['clip'] is None:
            if not anim['clips']:
                return
            anim_var.set(anim['clips'][0][0]); anim_select()
        anim['playing'] = True; play_btn.config(text='Pause'); anim['last'] = None
        anim_tick()

    def anim_stop():
        anim['playing'] = False; play_btn.config(text='Play')
        if anim['job']:
            root.after_cancel(anim['job']); anim['job'] = None

    def anim_tick():
        import time as _t
        anim['job'] = None
        if not anim['playing'] or anim['clip'] is None:
            return
        now = _t.perf_counter()
        dt = 0.0 if anim['last'] is None else min(0.1, now - anim['last'])
        anim['last'] = now
        try:
            sp = float(speed_var.get().rstrip('x'))
        except ValueError:
            sp = 1.0
        dur = max(anim['clip']['duration'], 1e-3)
        anim['t'] = (anim['t'] + dt * sp) % dur
        anim_apply()
        anim['job'] = root.after(16 if glview is not None else 60, anim_tick)

    anim_combo.bind('<<ComboboxSelected>>', anim_select)

    def view_model(m):
        if m is None:
            return m
        if not show_sky.get():
            m = core.strip_groups(m, state.get('sky') or set())
        e = state.get('sel')
        vmap = state.get('vmap') or {}
        m = core.strip_variants(m, hidden=set(vmap), vmap=vmap,
                                keep=variant_keep.get(e.key, set()) if e is not None else ())
        return m

    def on_sky():
        on_variants(keep=True)

    def on_variant(g):
        """Select variant g (or the original piece, 'variant 0'); the others of the same piece are deselected."""
        e = state.get('sel')
        if e is None:
            return
        ks = variant_keep.setdefault(e.key, set())
        vmap = state.get('vmap') or {}
        base = vmap.get(g, g)
        ks.difference_update({x for x in ks if vmap.get(x) == base})
        if g != base:
            ks.add(g)
        for gg, var in variant_vars.items():
            b = vmap.get(gg, gg)
            var.set((gg in ks) if gg != b else not any(vmap.get(x) == b for x in ks))
        on_variants(keep=True)

    def on_variants(keep=False):
        m = state.get('full_model')
        if m is None:
            return
        state['model'] = view_model(m)
        if glview is not None:
            glview.set_model(state['model'], state['texs'], flip_v=flip.get(), keep_view=keep)
            glview.set_options(use_textures=textured.get(), wire=wire.get())
        else:
            schedule_render()
        anim_rebuild()

    def show_info(r):
        info.delete('1.0', 'end')
        e = r['e']
        L = ['File:      %s' % e.name, 'Archive:   %s @ 0x%x (%s)' % (e.dat, e.off, e.kind),
             'Size:      %d bytes' % e.size, 'Class:     %s' % e.cls, '']
        m = r.get('model')
        if m:
            L.append('GEO version %d, %d materials, %d submeshes' % (m['version'], len(m['materials']), len(m['meshes'])))
            for s in m['meshes']:
                L.append('  mesh %d.%d  material %d  %6d verts  %6d tris  attrs: %s' % (
                    s['mesh'], s['sub'], s['material'], s['vcount'], len(s['indices']),
                    ','.join(k for k in ('position', 'normal', 'uv', 'uv2', 'color', 'joints', 'weights') if k in s)))
            for mt in m['materials']:
                L.append('  material %d: layers -> textures %s' % (mt['index'], [l['texture'] for l in mt['layers']]))
            hid = core.variant_groups(m)
            if hid:
                L.append('  variant mesh groups %s duplicate the body; selectable under Variants'
                         % sorted(hid))
            for er in m['errors']:
                L.append('  ! ' + er)
            ts, how = r.get('texset', (None, 'none'))
            if 'used' in r:
                allp = loader.parts(e)
                kinds = r.get('proxy', [])
                L.append('Assembled from %s (toggle above the 3D view)' % ', '.join(
                    ('part %d%s' % (k, ' (%s)' % kinds[k] if k < len(kinds) and kinds[k] else ''))
                    if k < e.nparts else ('attachment %04d' % allp[k].index) for k in r['used']))
            elif ts is not None:
                L.append('Texture set: %s (%s)' % (ts.name, how))
            elif not any(x['layers'] for x in m['materials']):
                L.append('Texture set: none - this model has no texture layers (vertex colours only)')
            else:
                L.append('Texture set: not found - pick one on the Textures tab')
        for t in r.get('texinfo', []):
            L.append('  tex %d: %dx%d %s mips=%d tile=%d' % (t['index'], t['width'], t['height'],
                                                           core.fmt_name(t['format']), t['mips'], t['tileMode']))
        if 'head' in r:
            h = r['head']
            for i in range(0, min(len(h), 512), 16):
                L.append('%06x  %-48s %s' % (i, h[i:i + 16].hex(' '),
                                             ''.join(chr(c) if 32 <= c < 127 else '.' for c in h[i:i + 16])))
        info.insert('1.0', '\n'.join(L))

    def fill_texset_combo(r):
        e = r['e']
        if e.cls != 'model' or e.part == -2:
            tscombo['values'] = []; tsvar.set('(per part - select a part to change its set)' if e.part == -2 else '')
            return
        # same archive first, then every other archive (shared sets can live elsewhere)
        sets = [x for x in state['entries'] if x.cls == 'textures' and x.dat == e.dat] + \
               [x for x in state['entries'] if x.cls == 'textures' and x.dat != e.dat]
        state['texset_choices'] = [None, 'none'] + sets
        labels = ['Auto', '(no textures)'] + [x.name for x in sets]
        tscombo['values'] = labels
        ov = loader.overrides.get(e.key)
        if ov is None:
            ts, how = r.get('texset', (None, 'none'))
            tsvar.set('Auto' + (' -> %s (%s)' % (ts.name, how) if ts is not None else ' -> none'))
        elif ov == 'none':
            tsvar.set('(no textures)')
        else:
            tsvar.set(ov.name)

    def on_texset(ev=None):
        e = state['sel']
        if e is None or e.cls != 'model':
            return
        i = tscombo.current()
        ch = state['texset_choices'][i]
        key = e.key
        if ch is None:
            loader.overrides.pop(key, None)
        else:
            loader.overrides[key] = ch
        state['sel'] = None
        on_select()

    def fill_textures():
        tlist.delete(0, 'end'); tcanvas.delete('all')
        for t in state['texinfo']:
            tlist.insert('end', '%d: %dx%d %s' % (t['index'], t['width'], t['height'], core.fmt_name(t['format'])))
        if state['texinfo']:
            tlist.selection_set(0); show_texture()

    def show_texture(ev=None):
        s = tlist.curselection()
        if not s or s[0] >= len(state['texs']):
            return
        img = Image.fromarray(state['texs'][s[0]])
        cw, ch = max(tcanvas.winfo_width(), 64), max(tcanvas.winfo_height(), 64)
        sc = min(cw / img.width, ch / img.height, 4.0)
        img = img.resize((max(1, int(img.width * sc)), max(1, int(img.height * sc))), Image.LANCZOS)
        bgimg = Image.new('RGBA', img.size, (90, 90, 90, 255))
        tile = 16
        a = np.zeros((img.height, img.width), bool)
        yy, xx = np.mgrid[0:img.height, 0:img.width]
        a = ((yy // tile + xx // tile) % 2).astype(bool)
        bga = np.array(bgimg); bga[a] = (120, 120, 120, 255)
        comp = Image.alpha_composite(Image.fromarray(bga), img.convert('RGBA'))
        state['tkimg_t'] = ImageTk.PhotoImage(comp)
        tcanvas.delete('all'); tcanvas.create_image(cw // 2, ch // 2, image=state['tkimg_t'])

    def schedule_render(fast=False):
        if glview is not None:
            glview.set_options(flip_v=flip.get(), use_textures=textured.get(), wire=wire.get())
            return
        if state['render_job']:
            root.after_cancel(state['render_job'])
        state['render_job'] = root.after(15 if fast else 30, lambda: do_render(fast))

    def do_render(fast=False):
        state['render_job'] = None
        m = state.get('anim_model') or state['model']
        if m is None:
            return
        cw, ch = max(canvas.winfo_width(), 64), max(canvas.winfo_height(), 64)
        sc = 0.5 if fast else 1.0
        size = (max(32, int(cw * sc)), max(32, int(ch * sc)))
        pan = (state['pan'][0] * sc, state['pan'][1] * sc)
        if fast or wire.get():
            tc = None
            if textured.get() and state['texs']:
                tc = rend.texture_sampler(state['texs'], m['materials'], flip_v=flip.get())
            img = rend.render(m['meshes'], size, yaw=state['yaw'], pitch=state['pitch'], zoom=state['zoom'],
                              pan=pan, tex_colors=tc, wire=wire.get())
        else:
            img = rend.render_hq(m['meshes'], size, yaw=state['yaw'], pitch=state['pitch'], zoom=state['zoom'],
                                 pan=pan, textures=state['texs'], materials=m['materials'], flip_v=flip.get(),
                                 use_textures=textured.get())
        if fast:
            img = img.resize((cw, ch), Image.NEAREST)
        state['tkimg'] = ImageTk.PhotoImage(img)
        canvas.delete('all'); canvas.create_image(0, 0, anchor='nw', image=state['tkimg'])
        if fast and not state['drag']:
            schedule_render(False)

    def press(ev, mode):
        state['drag'] = (mode, ev.x, ev.y)

    def motion(ev):
        if not state['drag']:
            return
        mode, x0, y0 = state['drag']
        dx, dy = ev.x - x0, ev.y - y0
        state['drag'] = (mode, ev.x, ev.y)
        if mode == 'rot':
            state['yaw'] += dx * 0.01; state['pitch'] = max(-1.55, min(1.55, state['pitch'] - dy * 0.01))
        else:
            state['pan'][0] += dx; state['pan'][1] += dy
        schedule_render(True)

    def wheel(ev):
        d = ev.delta if ev.delta else (120 if ev.num == 4 else -120)
        state['zoom'] *= 1.1 if d > 0 else 1 / 1.1
        schedule_render(True)

    def reset(ev=None):
        state.update(yaw=0.6, pitch=-0.25, zoom=1.0, pan=[0, 0]); schedule_render()

    canvas.bind('<ButtonPress-1>', lambda ev: press(ev, 'rot'))
    canvas.bind('<ButtonPress-2>', lambda ev: press(ev, 'pan'))
    canvas.bind('<ButtonPress-3>', lambda ev: press(ev, 'pan'))
    for b in ('<B1-Motion>', '<B2-Motion>', '<B3-Motion>'):
        canvas.bind(b, motion)
    canvas.bind('<ButtonRelease>', lambda ev: (state.update(drag=None), schedule_render(False)))
    canvas.bind('<MouseWheel>', wheel); canvas.bind('<Button-4>', wheel); canvas.bind('<Button-5>', wheel)
    canvas.bind('<Double-Button-1>', reset)
    canvas.bind('<Configure>', lambda ev: schedule_render())
    tlist.bind('<<ListboxSelect>>', show_texture)
    tscombo.bind('<<ComboboxSelected>>', on_texset)
    tcanvas.bind('<Configure>', show_texture)
    tree.bind('<<TreeviewSelect>>', on_select)

    def export_list(ents, outdir, model_textures=False):
        n = len(ents); fails = []
        for i, e in enumerate(ents):
            if cancelled():
                break
            d = os.path.join(outdir, os.path.splitext(os.path.basename(e.dat))[0])
            try:
                export_entry(loader, e, d, fmt_glb=True, fmt_obj=False, flip_v=flip.get(),
                             raw=e.cls not in ('model', 'textures'),         # other files: their raw data
                             entries=state['entries'], variants=variant_keep.get(e.key, set()), skydome=show_sky.get(),
                             tex_dir=os.path.join(outdir, 'textures'), model_textures=model_textures,
                             skeleton=exp_skel.get(), animations=exp_skel.get() and exp_anim.get())
            except Exception as ex:
                fails.append('%s: %s' % (e.name, ex))
            if not cancelled():
                ui(lambda i=i, e=e: set_status('Exporting %d/%d  %s' % (i + 1, n, e.name)))
        return fails, n

    def export_done(r):
        fails, n = r
        set_status('Export finished: %d items, %d failed%s  ->  %s' % (n, len(fails), '',
                                                                       EXPORT_DIR), 1)
        if fails:
            messagebox.showwarning('Some exports failed', '\n'.join(fails[:30]))

    def export_selected():
        ents = selected_entries()
        if not ents:
            messagebox.showinfo('Nothing selected', 'Select one or more files in the list.'); return
        bg(lambda: export_list(ents, EXPORT_DIR, model_textures=True), export_done, 'Exporting…')

    pump()
    root.mainloop()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'extract':
        sys.exit(cli_extract(sys.argv[2:]))
    run_gui()

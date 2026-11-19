"""GPU (OpenGL) model viewer widget for Tkinter.  Requires: pip install PyOpenGL pyopengltk"""
import ctypes
import math
import numpy as np
from OpenGL import GL as gl
from pyopengltk import OpenGLFrame


class GLModelView(OpenGLFrame):
    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.animate = 0
        self.width, self.height = 1, 1
        self._redraw_job = None
        self._uploaded_once = False
        self.model = None
        self.textures = []
        self.tex_ids = []
        self.flip_v = False
        self.use_textures = True
        self.wire = False
        self.yaw, self.pitch, self.dist_scale = 0.6, -0.25, 1.0
        self.pan = [0.0, 0.0]
        self._drag = None
        self._gl_ready = False
        self._pending_upload = False
        self.bind('<ButtonPress-1>', lambda e: self._press(e, 'rot'))
        self.bind('<ButtonPress-2>', lambda e: self._press(e, 'pan'))
        self.bind('<ButtonPress-3>', lambda e: self._press(e, 'pan'))
        for b in ('<B1-Motion>', '<B2-Motion>', '<B3-Motion>'):
            self.bind(b, self._motion)
        self.bind('<ButtonRelease>', lambda e: setattr(self, '_drag', None))
        self.bind('<MouseWheel>', self._wheel)
        self.bind('<Button-4>', self._wheel)
        self.bind('<Button-5>', self._wheel)
        self.bind('<Double-Button-1>', lambda e: self.reset_view())
        # first-person ("WASD Camera") mode
        self.fly = False
        self.fly_pos = np.zeros(3)
        self.fly_yaw = self.fly_pitch = 0.0
        self.fly_speed = 1.0              # multiplier, changed with the mouse wheel
        self._keys = set()
        self._key_release_jobs = {}
        self._fly_job = None
        self._fly_last = None
        self.bind('<KeyPress>', self._key_down)
        self.bind('<KeyRelease>', self._key_up)
        self.bind('<FocusOut>', lambda e: self._keys.clear())

    # ---------------------------------------------------------------- public
    def set_model(self, model, textures, flip_v=False, keep_view=False):
        self.model = model
        self.textures = textures or []
        self.flip_v = flip_v
        self._prepare()
        self._pending_upload = True
        self._pending_buffers = True
        if keep_view:
            self.refresh()
        else:
            self.reset_view()

    def set_options(self, flip_v=None, use_textures=None, wire=None):
        if flip_v is not None and flip_v != self.flip_v:
            self.flip_v = flip_v
            self._prepare()
            self._pending_buffers = True
        if use_textures is not None:
            self.use_textures = use_textures
        if wire is not None:
            self.wire = wire
        self.refresh()

    def reset_view(self):
        self.yaw, self.pitch, self.dist_scale, self.pan = 0.6, -0.25, 1.0, [0.0, 0.0]
        if self.fly:
            self._fly_from_orbit()
        self.refresh()

    def set_fly(self, on):
        """Toggle first-person camera: WASD to move, Q/E down/up, Shift faster,
        right-drag to look around, wheel to change speed."""
        on = bool(on)
        if on and not self.fly:
            self._fly_from_orbit()
        self.fly = on
        self._keys.clear()
        if on:
            self.focus_set()
        self.refresh()

    # orbit camera as a matrix, so fly mode can start exactly where the orbit view is
    @staticmethod
    def _rx(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    @staticmethod
    def _ry(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def _fly_rot(self):
        return self._rx(-self.fly_pitch) @ self._ry(self.fly_yaw)

    def _fly_from_orbit(self):
        r = getattr(self, 'radius', 1.0)
        center = getattr(self, 'center', np.zeros(3))
        dist = r * 2.6 / self.dist_scale
        R = self._rx(-self.pitch) @ self._ry(self.yaw)
        t = np.array([self.pan[0] * r, self.pan[1] * r, -dist])
        # eye = R @ (world - center) + t  ->  camera (eye origin) in world:
        self.fly_pos = center - R.T @ t
        self.fly_yaw, self.fly_pitch = self.yaw, self.pitch

    def refresh(self):
        # coalesce bursts of mouse events into one redraw
        if self._redraw_job is None:
            self._redraw_job = self.after_idle(self._do_refresh)

    def _do_refresh(self):
        self._redraw_job = None
        if not self.context_created:
            return
        try:
            self._display()
        except Exception as ex:  # keep the UI alive if the driver complains
            print('GL redraw failed:', ex)

    # ---------------------------------------------------------------- data
    def _prepare(self):
        self.draw = []
        m = self.model
        if not m:
            self.center = np.zeros(3); self.radius = 1.0
            return
        ps = [s['position'][:, :3] for s in m['meshes'] if 'position' in s and len(s['indices'])]
        if ps:
            P = np.concatenate(ps)
            lo, hi = P.min(0), P.max(0)
            self.center = (lo + hi) / 2
            self.radius = max(float(np.linalg.norm(hi - lo)) / 2, 1e-4)
        else:
            self.center = np.zeros(3); self.radius = 1.0
        try:
            import tfp_core
            overs = tfp_core.get_face_flags(m)[1]          # cached per visible mesh set (often precomputed)
        except Exception:
            overs = [np.zeros(len(s['indices']), bool) for s in m['meshes']]
        for si, (s, ov) in enumerate(zip(m['meshes'], overs)):
            if 'position' not in s or not len(s['indices']):
                continue
            # everything is drawn back-face culled (as in-game); decal-like overlays (same-facing, coplanar with
            # an earlier triangle) go last, drawn with a depth bias
            tri = s['indices'].reshape(-1, 3)
            rest = ~ov
            item = dict(src=si,
                        pos=np.ascontiguousarray(s['position'][:, :3], np.float32),
                        idx=np.ascontiguousarray(np.concatenate([tri[rest], tri[ov]]).reshape(-1), np.uint32),
                        nover=int(ov.sum()) * 3)
            if 'normal' in s:
                n = s['normal'][:, :3].astype(np.float32)
                ln = np.linalg.norm(n, axis=1, keepdims=True); ln[ln == 0] = 1
                item['nrm'] = np.ascontiguousarray(n / ln)
            if 'uv' in s:
                uv = s['uv'][:, :2].astype(np.float32).copy()
                if self.flip_v:
                    uv[:, 1] = 1 - uv[:, 1]
                item['uv'] = np.ascontiguousarray(uv)
            if 'color' in s:
                item['col'] = np.ascontiguousarray(s['color'][:, :4], np.uint8)
            mi = s.get('material', 0)
            mats = m['materials']
            item['tex'] = mats[mi]['layers'][0]['texture'] if mi < len(mats) and mats[mi]['layers'] else None
            hue = (mi * 0.37) % 1.0
            import colorsys
            item['flat'] = colorsys.hsv_to_rgb(hue, 0.25, 0.9)
            self.draw.append(item)

    def _upload_buffers(self):
        for it in getattr(self, '_old_bufs', []):
            try:
                gl.glDeleteBuffers(len(it), it)
            except Exception:
                pass
        self._old_bufs = []
        for it in self.draw:
            bufs = {}
            for k in ('pos', 'nrm', 'uv', 'col'):
                if k in it:
                    b = gl.glGenBuffers(1)
                    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, b)
                    gl.glBufferData(gl.GL_ARRAY_BUFFER, it[k].nbytes, it[k], gl.GL_STATIC_DRAW)
                    bufs[k] = b
            b = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, b)
            gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, it['idx'].nbytes, it['idx'], gl.GL_STATIC_DRAW)
            bufs['idx'] = b
            it['vbo'] = bufs
            self._old_bufs.append(list(bufs.values()))
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
        self._pending_buffers = False

    def update_geometry(self, updates):
        """Replace vertex positions/normals: updates = [(submesh index, positions, normals or None)].
        Used for animation playback; the triangles stay the same."""
        by_src = {it.get('src'): it for it in getattr(self, 'draw', [])}
        pending = []
        for idx, pos, nrm in updates:
            it = by_src.get(idx)
            if it is None or len(pos) != len(it['pos']):
                continue
            it['pos'] = np.ascontiguousarray(pos, np.float32)
            if nrm is not None and 'nrm' in it:
                it['nrm'] = np.ascontiguousarray(nrm, np.float32)
            pending.append(it)
        for it in pending:
            it['_dirty'] = True                     # uploaded at the start of the next redraw (context current)
        self.refresh()

    def _flush_dirty(self):
        for it in getattr(self, 'draw', []):
            if it.pop('_dirty', False):
                vb = it.get('vbo', {})
                for k in ('pos', 'nrm'):
                    if k in vb and k in it:
                        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vb[k])
                        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, it[k].nbytes, it[k])
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    def _upload_textures(self):
        if self.tex_ids:
            gl.glDeleteTextures(self.tex_ids)
        self.tex_ids = []
        for t in self.textures:
            tid = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, tid)
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
            mip = bool(getattr(gl, 'glGenerateMipmap', None))
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER,
                               gl.GL_LINEAR_MIPMAP_LINEAR if mip else gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
            img = np.ascontiguousarray(t, np.uint8)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, img.shape[1], img.shape[0], 0,
                            gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, img)
            if mip:
                try:
                    gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
                except Exception:
                    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            self.tex_ids.append(tid)
        self._pending_upload = False

    # ---------------------------------------------------------------- GL callbacks
    def initgl(self):
        gl.glClearColor(40 / 255, 44 / 255, 52 / 255, 1)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)
        gl.glEnable(gl.GL_NORMALIZE)
        gl.glLightModeli(gl.GL_LIGHT_MODEL_TWO_SIDE, gl.GL_TRUE)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, (0.35, 0.35, 0.35, 1))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, (0.75, 0.75, 0.75, 1))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_SPECULAR, (0, 0, 0, 1))
        gl.glEnable(gl.GL_COLOR_MATERIAL)
        gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)
        gl.glAlphaFunc(gl.GL_GREATER, 0.5)
        self._gl_ready = True
        if not self._uploaded_once:  # initgl is also called on every resize
            self._pending_upload = True
            self._uploaded_once = True

    def redraw(self):
        w, h = max(self.width, 1), max(self.height, 1)
        gl.glViewport(0, 0, w, h)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        if not self.model:
            return
        try:
            gl.glBindVertexArray(0)
        except Exception:
            pass
        if self._pending_upload:
            self._upload_textures()
        if getattr(self, '_pending_buffers', True):
            self._upload_buffers()
            for it in self.draw:
                it.pop('_dirty', None)
        else:
            self._flush_dirty()
        r = self.radius
        dist = r * 2.6 / self.dist_scale
        if self.fly:
            dist = float(np.linalg.norm(self.fly_pos - self.center))
        gl.glMatrixMode(gl.GL_PROJECTION); gl.glLoadIdentity()
        # keep the depth range tight around the model: Tk GL contexts often have a 16-bit depth buffer,
        # and a loose near plane makes nearby surfaces (e.g. chest trim) z-fight
        near, far = max(dist - r * 1.25, dist * 0.05), dist + r * 1.25
        if self.fly:            # the camera can be inside the model
            near, far = max(dist - r * 1.25, r * 0.005), dist + r * 1.25
        fh = math.tan(math.radians(20.0)) * near
        gl.glFrustum(-fh * w / h, fh * w / h, -fh, fh, near, far)
        gl.glMatrixMode(gl.GL_MODELVIEW); gl.glLoadIdentity()
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, (0.3, 0.5, 0.8, 0.0))  # light fixed to camera
        if self.fly:
            gl.glRotatef(-math.degrees(self.fly_pitch), 1, 0, 0)
            gl.glRotatef(math.degrees(self.fly_yaw), 0, 1, 0)
            gl.glTranslatef(*(-self.fly_pos).tolist())
        else:
            gl.glTranslatef(self.pan[0] * r, self.pan[1] * r, -dist)
            gl.glRotatef(-math.degrees(self.pitch), 1, 0, 0)
            gl.glRotatef(math.degrees(self.yaw), 0, 1, 0)
            gl.glTranslatef(*(-self.center).tolist())
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE if self.wire else gl.GL_FILL)
        if self.wire:                       # wireframe shows every edge
            gl.glDisable(gl.GL_CULL_FACE)
        else:                               # back faces are culled, as in-game
            gl.glEnable(gl.GL_CULL_FACE); gl.glFrontFace(gl.GL_CCW); gl.glCullFace(gl.GL_BACK)
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        for it in self.draw:
            vb = it['vbo']
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vb['pos']); gl.glVertexPointer(3, gl.GL_FLOAT, 0, None)
            if 'nrm' in vb:
                gl.glEnableClientState(gl.GL_NORMAL_ARRAY)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vb['nrm']); gl.glNormalPointer(gl.GL_FLOAT, 0, None)
            tex = it['tex']
            textured = (self.use_textures and not self.wire and tex is not None and tex < len(self.tex_ids)
                        and 'uv' in vb)
            if textured:
                gl.glEnable(gl.GL_TEXTURE_2D); gl.glEnable(gl.GL_ALPHA_TEST)
                gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex_ids[tex])
                gl.glEnableClientState(gl.GL_TEXTURE_COORD_ARRAY)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vb['uv']); gl.glTexCoordPointer(2, gl.GL_FLOAT, 0, None)
                gl.glColor4f(1, 1, 1, 1)
            elif 'col' in vb and not self.wire:
                gl.glEnableClientState(gl.GL_COLOR_ARRAY)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vb['col']); gl.glColorPointer(4, gl.GL_UNSIGNED_BYTE, 0, None)
            else:
                gl.glColor4f(*it['flat'], 1)
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, vb['idx'])
            nt, no = len(it['idx']), it.get('nover', 0)
            if nt > no:
                gl.glDrawElements(gl.GL_TRIANGLES, nt - no, gl.GL_UNSIGNED_INT, None)
            if no:
                gl.glEnable(gl.GL_POLYGON_OFFSET_FILL); gl.glPolygonOffset(-1.0, -4.0)
                gl.glDrawElements(gl.GL_TRIANGLES, no, gl.GL_UNSIGNED_INT, ctypes.c_void_p((nt - no) * 4))
                gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
            gl.glDisableClientState(gl.GL_NORMAL_ARRAY)
            gl.glDisableClientState(gl.GL_TEXTURE_COORD_ARRAY)
            gl.glDisableClientState(gl.GL_COLOR_ARRAY)
            gl.glDisable(gl.GL_TEXTURE_2D); gl.glDisable(gl.GL_ALPHA_TEST)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)

    # ---------------------------------------------------------------- input
    def _press(self, e, mode):
        self.focus_set()
        if self.fly:
            mode = 'look' if e.num == 3 else None
        self._drag = (mode, e.x, e.y) if mode else None

    def _motion(self, e):
        if not self._drag:
            return
        mode, x0, y0 = self._drag
        dx, dy = e.x - x0, e.y - y0
        self._drag = (mode, e.x, e.y)
        if mode == 'look':
            self.fly_yaw += dx * 0.005
            self.fly_pitch = max(-1.55, min(1.55, self.fly_pitch - dy * 0.005))
        elif mode == 'rot':
            self.yaw += dx * 0.01
            self.pitch = max(-1.55, min(1.55, self.pitch - dy * 0.01))
        else:
            k = 2.0 / max(self.height, 1) / self.dist_scale
            self.pan[0] += dx * k
            self.pan[1] -= dy * k
        self.refresh()

    def _wheel(self, e):
        d = e.delta if e.delta else (120 if e.num == 4 else -120)
        if self.fly:
            self.fly_speed = min(50.0, max(0.02, self.fly_speed * (1.25 if d > 0 else 0.8)))
            return
        self.dist_scale *= 1.1 if d > 0 else 1 / 1.1
        self.refresh()

    # ---------------------------------------------------------------- WASD movement
    MOVE_KEYS = {'w', 's', 'a', 'd', 'q', 'e', 'shift_l', 'shift_r'}

    def _key_down(self, e):
        k = e.keysym.lower()
        if k not in self.MOVE_KEYS:
            return
        job = self._key_release_jobs.pop(k, None)     # key auto-repeat sends release+press pairs
        if job:
            self.after_cancel(job)
        self._keys.add(k)
        if self.fly and self._fly_job is None:
            self._fly_last = None
            self._fly_job = self.after(16, self._fly_tick)

    def _key_up(self, e):
        k = e.keysym.lower()
        if k in self._keys:
            self._key_release_jobs[k] = self.after(30, lambda k=k: (self._keys.discard(k),
                                                                     self._key_release_jobs.pop(k, None)))

    def _fly_tick(self):
        import time
        self._fly_job = None
        if not self.fly or not (self._keys - {'shift_l', 'shift_r'}):
            return
        now = time.perf_counter()
        dt = min(0.1, now - self._fly_last) if self._fly_last else 0.016
        self._fly_last = now
        R = self._fly_rot()
        fwd, right, up = R.T @ np.array([0, 0, -1.0]), R.T @ np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
        v = np.zeros(3)
        k = self._keys
        if 'w' in k: v += fwd
        if 's' in k: v -= fwd
        if 'd' in k: v += right
        if 'a' in k: v -= right
        if 'e' in k: v += up
        if 'q' in k: v -= up
        n = np.linalg.norm(v)
        if n > 0:
            speed = getattr(self, 'radius', 1.0) * 1.2 * self.fly_speed
            if 'shift_l' in k or 'shift_r' in k:
                speed *= 3
            self.fly_pos = self.fly_pos + v / n * speed * dt
            self.refresh()
        self._fly_job = self.after(16, self._fly_tick)

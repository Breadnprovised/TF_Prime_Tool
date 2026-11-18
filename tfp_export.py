"""Exporters: glTF binary (.glb) with embedded PNG textures, and OBJ/MTL/PNG."""
import io
import json
import os
import struct
import numpy as np
from PIL import Image


def _png_bytes(rgba):
    b = io.BytesIO()
    Image.fromarray(rgba).save(b, 'PNG')
    return b.getvalue()


def _mesh_material(m, materials):
    """returns (material index, texture index or None)"""
    mi = m.get('material', 0)
    if mi < len(materials) and materials[mi]['layers']:
        return mi, materials[mi]['layers'][0]['texture']
    return mi, None


def alpha_mode(tex):
    """glTF alpha mode for a base-colour texture, from its alpha channel: OPAQUE (no transparency), MASK (cut-out
    shapes such as decals and grilles: hard or mostly hard edges) or BLEND (see-through surfaces such as glass
    and holograms: a large share of partially transparent pixels)."""
    if tex is None or tex.ndim < 3 or tex.shape[2] < 4:
        return 'OPAQUE'
    a = tex[..., 3]
    if a.min() >= 250:
        return 'OPAQUE'
    partial = float(np.mean((a > 16) & (a < 240)))
    return 'BLEND' if partial > 0.2 else 'MASK'


def write_glb(path, model, textures, flip_v=False, name='model', part_names=None, rig=None):
    """model: parse_geo() result. textures: list of RGBA numpy arrays (may be empty).
    rig (optional, see Loader.rig_for): skeletons and per-vertex joints/weights -> glTF skins (Blender armatures)."""
    src_idx = [i for i, m in enumerate(model['meshes']) if 'position' in m and len(m['indices'])]
    meshes = [model['meshes'][i] for i in src_idx]
    bin_parts, views, accessors = [], [], []
    offset = 0

    def add_view(data, target=None):
        nonlocal offset
        pad = (-len(data)) % 4
        views.append(dict(buffer=0, byteOffset=offset, byteLength=len(data), **({'target': target} if target else {})))
        bin_parts.append(data + b'\0' * pad)
        offset += len(data) + pad
        return len(views) - 1

    def add_acc(arr, comp, typ, target=None, minmax=False, normalized=False):
        v = add_view(arr.tobytes(), target)
        a = dict(bufferView=v, componentType=comp, count=int(arr.shape[0]), type=typ)
        if normalized:
            a['normalized'] = True
        if minmax:
            a['min'] = arr.min(0).tolist(); a['max'] = arr.max(0).tolist()
        accessors.append(a)
        return len(accessors) - 1

    # textures / images
    images, gl_textures = [], []
    for i, t in enumerate(textures):
        v = add_view(_png_bytes(t))
        images.append(dict(bufferView=v, mimeType='image/png', name='%s_tex%d' % (name, i)))
        gl_textures.append(dict(source=i, sampler=0))
    gl_materials = []
    for m in model['materials']:
        pbr = dict(metallicFactor=0.0, roughnessFactor=0.8)
        gm = dict(name='mat%d' % m['index'], pbrMetallicRoughness=pbr, doubleSided=True)
        if m['layers'] and m['layers'][0]['texture'] < len(textures):
            ti = m['layers'][0]['texture']
            pbr['baseColorTexture'] = dict(index=ti)
            mode = alpha_mode(textures[ti])
            if mode != 'OPAQUE':
                gm['alphaMode'] = mode
                if mode == 'MASK':
                    gm['alphaCutoff'] = 0.5
        gl_materials.append(gm)
    if not gl_materials:
        gl_materials.append(dict(name='default', pbrMetallicRoughness=dict(metallicFactor=0.0), doubleSided=True))
    # one-sided copies, used for triangles that have a coincident opposite-facing twin: the game shows only the
    # side facing the camera, so drawing both sides would z-fight
    n_mat = len(gl_materials)
    onesided = {}                    # material -> its one-sided copy, created only for materials that need one

    def onesided_of(mi):
        if mi not in onesided:
            gl_materials.append(dict(gl_materials[mi], name=gl_materials[mi]['name'] + '_onesided', doubleSided=False))
            onesided[mi] = len(gl_materials) - 1
        return onesided[mi]
    try:
        import tfp_core
        twins = tfp_core.get_face_flags(dict(model, meshes=meshes))[0]
    except Exception:
        twins = [np.zeros(len(m['indices']), bool) for m in meshes]

    by_part = {}                     # merged-part index -> primitives (one object per part / attachment)
    rig_verts = (rig or {}).get('verts', {})
    part_skel = (rig or {}).get('part_skel', {})
    for si_, m, tw in zip(src_idx, meshes, twins):
        attrs = {'POSITION': add_acc(np.ascontiguousarray(m['position'][:, :3], np.float32), 5126, 'VEC3', 34962, True)}
        if 'normal' in m:
            n = m['normal'][:, :3].astype(np.float32)
            ln = np.linalg.norm(n, axis=1, keepdims=True)
            bad = (ln[:, 0] < 1e-6) | ~np.isfinite(ln[:, 0])
            n = np.where(bad[:, None], np.array([0, 1, 0], np.float32), n / np.where(bad[:, None], 1, ln))
            attrs['NORMAL'] = add_acc(np.ascontiguousarray(n, np.float32), 5126, 'VEC3', 34962)
        if 'uv' in m:
            uv = m['uv'][:, :2].astype(np.float32).copy()
            if flip_v:
                uv[:, 1] = 1 - uv[:, 1]
            attrs['TEXCOORD_0'] = add_acc(uv, 5126, 'VEC2', 34962)
        if 'color' in m:
            attrs['COLOR_0'] = add_acc(np.ascontiguousarray(m['color'], np.uint8), 5121, 'VEC4', 34962, normalized=True)
        if m.get('part', 0) in part_skel:          # skinned object: every primitive needs joints + weights
            if si_ in rig_verts:
                _, J, W = rig_verts[si_]
            else:
                J = np.zeros((len(m['position']), 4), np.uint16)
                W = np.zeros((len(m['position']), 4), np.float32); W[:, 0] = 1
            attrs['JOINTS_0'] = add_acc(np.ascontiguousarray(J, np.uint16), 5123, 'VEC4', 34962)
            attrs['WEIGHTS_0'] = add_acc(np.ascontiguousarray(W, np.float32), 5126, 'VEC4', 34962)
        mat = m.get('material', 0)
        mat = mat if mat < n_mat else 0
        tri = m['indices'].reshape(-1, 3)
        prims = []
        for sel, mi in ((~tw, mat), (tw, None)):
            if sel.any():
                if mi is None:
                    mi = onesided_of(mat)
                idx = add_acc(np.ascontiguousarray(tri[sel].reshape(-1), np.uint32), 5125, 'SCALAR', 34963)
                prims.append(dict(attributes=attrs, indices=idx, material=mi))
        by_part.setdefault(m.get('part', 0), []).extend(prims)
    # one mesh (= one object in Blender) per part / attachment; its pieces are primitives with their own materials
    gl_meshes, nodes, skins = [], [], []
    top = []                         # children of the scene root node
    # skeletons -> joint nodes + skins
    skel_nodes = []
    for sk in (rig or {}).get('skeletons', []):
        base = len(nodes)
        for (r, t, s), nm in zip(sk['locals'], sk['names']):
            nodes.append(dict(name=nm, rotation=[float(x) for x in r], translation=[float(x) for x in t],
                              scale=[float(x) for x in s]))
        for j, nd in enumerate(sk['nodes']):
            if nd['parent'] >= 0:
                nodes[base + nd['parent']].setdefault('children', []).append(base + j)
        skel_nodes.append(base)
    for si, sk in enumerate((rig or {}).get('skeletons', [])):
        base = skel_nodes[si]
        roots = [base + j for j, nd in enumerate(sk['nodes']) if nd['parent'] < 0]
        att = sk.get('attach')
        if att and att[0] == 'matrix':
            M = np.asarray(att[1], np.float64)
            nodes.append(dict(name=sk['names'][0].rstrip('0123456789') + 'anchor',
                              matrix=[float(x) for x in M.T.reshape(-1)], children=roots))
            top.append(len(nodes) - 1)
        elif att:
            nodes[skel_nodes[att[0]] + att[1]].setdefault('children', []).extend(roots)
        else:
            top.extend(roots)
        ibm = np.stack([np.asarray(M, np.float32).T.reshape(-1) for M in sk['inv']])
        skin = dict(name=sk['names'][0].rstrip('0123456789') + 'skin',
                    joints=list(range(base, base + len(sk['nodes']))),
                    inverseBindMatrices=add_acc(np.ascontiguousarray(ibm, np.float32), 5126, 'MAT4'))
        if len(roots) == 1 and not att:
            skin['skeleton'] = roots[0]                 # optional; only valid when it's the common root
        skins.append(skin)
    scene_nodes = []                 # skinned mesh nodes must sit at the scene root
    for k in sorted(by_part):
        pn = (part_names or {}).get(k) or (name if len(by_part) == 1 else '%s_part%d' % (name, k))
        gl_meshes.append(dict(name=pn, primitives=by_part[k]))
        nd = dict(name=pn, mesh=len(gl_meshes) - 1)
        nodes.append(nd)
        if k in part_skel:
            nd['skin'] = part_skel[k]
            scene_nodes.append(len(nodes) - 1)
        else:
            top.append(len(nodes) - 1)
    if len(top) == 1 and not skins:
        scene_nodes.insert(0, top[0])
    elif top:
        nodes.append(dict(name=name if not skins else name + '_skeleton', children=top))
        scene_nodes.insert(0, len(nodes) - 1)
    if not nodes:
        nodes, scene_nodes = [dict(name=name)], [0]
    gl_anims = []
    for an in (rig or {}).get('animations', []):
        samplers, channels = [], []
        for skel, joint, prop, times, vals in an['channels']:
            inp = add_acc(np.ascontiguousarray(times.reshape(-1, 1), np.float32), 5126, 'SCALAR', None, True)
            out = add_acc(np.ascontiguousarray(vals, np.float32), 5126, 'VEC4' if prop == 'rotation' else 'VEC3')
            samplers.append(dict(input=inp, output=out, interpolation='LINEAR' if len(times) > 1 else 'STEP'))
            channels.append(dict(sampler=len(samplers) - 1, target=dict(node=skel_nodes[skel] + joint, path=prop)))
        if channels:
            gl_anims.append(dict(name=an['name'], samplers=samplers, channels=channels))
    gltf = dict(asset=dict(version='2.0', generator='tfp_tool'), scene=0,
                scenes=[dict(nodes=scene_nodes)], nodes=nodes, meshes=gl_meshes,
                materials=gl_materials, accessors=accessors, bufferViews=views,
                buffers=[dict(byteLength=offset)])
    if skins:
        gltf['skins'] = skins
    if gl_anims:
        gltf['animations'] = gl_anims
    if images:
        gltf['images'] = images; gltf['textures'] = gl_textures
        gltf['samplers'] = [dict(magFilter=9729, minFilter=9987, wrapS=10497, wrapT=10497)]
    js = json.dumps(gltf, separators=(',', ':')).encode()
    js += b' ' * ((-len(js)) % 4)
    binb = b''.join(bin_parts)
    with open(path, 'wb') as f:
        f.write(struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(binb)))
        f.write(struct.pack('<II', len(js), 0x4E4F534A)); f.write(js)
        f.write(struct.pack('<II', len(binb), 0x004E4942)); f.write(binb)


def write_obj(path, model, textures, flip_v=True, name='model', part_names=None):
    """OBJ uses bottom-left UV origin, so V is flipped by default."""
    base = os.path.splitext(path)[0]
    d = os.path.dirname(path)
    tex_files = []
    for i, t in enumerate(textures):
        fn = '%s_tex%d.png' % (os.path.basename(base), i)
        Image.fromarray(t).save(os.path.join(d, fn))
        tex_files.append(fn)
    with open(base + '.mtl', 'w') as f:
        for m in model['materials']:
            f.write('newmtl mat%d\nKd 1 1 1\n' % m['index'])
            if m['layers'] and m['layers'][0]['texture'] < len(tex_files):
                f.write('map_Kd %s\n' % tex_files[m['layers'][0]['texture']])
            f.write('\n')
    with open(path, 'w') as f:
        f.write('mtllib %s.mtl\n' % os.path.basename(base))
        vo = 1
        parts = sorted({m.get('part', 0) for m in model['meshes']})
        cur = None
        for m in sorted(model['meshes'], key=lambda s: s.get('part', 0)):
            if 'position' not in m or not len(m['indices']):
                continue
            k = m.get('part', 0)
            if k != cur:                 # one object per part / attachment; its pieces are groups
                cur = k
                pn = (part_names or {}).get(k) or (name if len(parts) == 1 else '%s_part%d' % (name, k))
                f.write('o %s\n' % pn)
            f.write('g mesh%d_%d\nusemtl mat%d\n' % (m['mesh'], m['sub'], m.get('material', 0)))
            np.savetxt(f, m['position'][:, :3], fmt='v %.6f %.6f %.6f')
            has_uv, has_n = 'uv' in m, 'normal' in m
            if has_uv:
                uv = m['uv'][:, :2].copy()
                if flip_v:
                    uv[:, 1] = 1 - uv[:, 1]
                np.savetxt(f, uv, fmt='vt %.6f %.6f')
            if has_n:
                np.savetxt(f, m['normal'][:, :3], fmt='vn %.6f %.6f %.6f')
            t = m['indices'].astype(np.int64) + vo
            if has_uv and has_n:
                fmt = 'f %d/%d/%d %d/%d/%d %d/%d/%d'; arr = np.repeat(t, 3, axis=1)
            elif has_uv:
                fmt = 'f %d/%d %d/%d %d/%d'; arr = np.repeat(t, 2, axis=1)
            elif has_n:
                fmt = 'f %d//%d %d//%d %d//%d'; arr = np.repeat(t, 2, axis=1)
            else:
                fmt = 'f %d %d %d'; arr = t
            np.savetxt(f, arr, fmt=fmt)
            vo += len(m['position'])

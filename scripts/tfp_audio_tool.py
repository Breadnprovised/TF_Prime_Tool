#!/usr/bin/env python3
"""
Transformers Prime: The Game (Wii U) - audio viewer / extractor.

Requires: Python 3.8+, numpy  (numba optional: faster decoding)

GUI:   python scripts/tfp_audio_tool.py   (or "Open Audio Viewer.bat")
CLI:   python scripts/tfp_audio_tool.py extract <game folder> [output folder, default .exported]

Select the game folder (the one containing code/ content/ meta/, or the content folder itself). The tool reads
content/TRF_SOUND.bfsar (names, sound effects, instrument samples) and content/stream/*.bfstm (music, voice).
"""
import os
import sys
import threading
import queue
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tfp_audio as aud

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# the tool's main folder: the scripts live in its 'scripts' subfolder
TOOL_DIR = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR).lower() == 'scripts' else SCRIPT_DIR
EXPORT_DIR = os.path.join(TOOL_DIR, '.exported')     # exports go to .exported/audio


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


def _sz(n):
    return '%.1f MB' % (n / 1048576) if n >= 1048576 else '%.1f KB' % (n / 1024)


# ------------------------------------------------------------------ CLI
def cli_extract(argv):
    import argparse
    ap = argparse.ArgumentParser(prog='tfp_audio_tool.py extract')
    ap.add_argument('game')
    ap.add_argument('out', nargs='?', default=EXPORT_DIR)
    ap.add_argument('--filter', default='', help='only sounds whose name contains this text (e.g. BGM_ or _SP)')
    a = ap.parse_args(argv)
    sar = aud.find_sound_archive(a.game)
    if not sar:
        print('No sound archive (.bfsar) found under', a.game)
        return 1
    ents = [e for e in aud.index_audio(aud.SoundArchive(sar))
            if e.sound.kind != 'sequence' and a.filter.lower() in e.name.lower()]
    print('Exporting %d sounds to %s' % (len(ents), os.path.join(a.out, 'audio')))
    fails = 0
    for i, e in enumerate(ents):
        try:
            aud.export_audio(e, os.path.join(a.out, 'audio'))
        except Exception as ex:
            fails += 1
            print('[%d/%d] %s FAILED: %s' % (i + 1, len(ents), e.name, ex))
            continue
        if (i + 1) % 250 == 0 or i + 1 == len(ents):
            print('[%d/%d] %s' % (i + 1, len(ents), e.name))
    print('Done: %d exported, %d failed' % (len(ents) - fails, fails))
    return 0


# ------------------------------------------------------------------ GUI
def run_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    root = tk.Tk()
    root.title('TF Prime (Wii U) Audio Viewer / Extractor')
    root.geometry('1200x760')
    state = dict(entries=[], items={}, sel=None, task=0)
    msgq = queue.Queue()

    # ---- toolbar
    bar = ttk.Frame(root, padding=4); bar.pack(side='top', fill='x')
    ttk.Button(bar, text='Open Game Folder', command=lambda: open_folder()).pack(side='left')
    ttk.Label(bar, text='  Search:').pack(side='left')
    search = tk.StringVar()
    sentry = ttk.Entry(bar, textvariable=search, width=32); sentry.pack(side='left', padx=(2, 0))
    ttk.Separator(bar, orient='vertical').pack(side='left', fill='y', padx=6)
    ttk.Label(bar, text='List:').pack(side='left')
    show_raw = tk.BooleanVar(value=True)
    ttk.Checkbutton(bar, text='Wave Archives', variable=show_raw, command=lambda: fill_tree()).pack(side='left', padx=4)
    show_seq = tk.BooleanVar(value=True)
    ttk.Checkbutton(bar, text='Sequences', variable=show_seq, command=lambda: fill_tree()).pack(side='left', padx=4)
    ttk.Separator(bar, orient='vertical').pack(side='left', fill='y', padx=6)
    ttk.Button(bar, text='Export', command=lambda: export_selected()).pack(side='left', padx=4)
    ttk.Button(bar, text='Open Exports Folder',
               command=lambda: open_folder_in_os(os.path.join(EXPORT_DIR, 'audio'))).pack(side='left')

    status = tk.StringVar(value='Open the game folder (the one with code/, content/, meta/).')
    sb = ttk.Frame(root, padding=2); sb.pack(side='bottom', fill='x')
    ttk.Label(sb, textvariable=status, anchor='w').pack(side='left', fill='x', expand=True)

    pw = ttk.PanedWindow(root, orient='horizontal'); pw.pack(fill='both', expand=True)
    left = ttk.Frame(pw); pw.add(left, weight=1)
    tree = ttk.Treeview(left, columns=('type', 'size'), selectmode='extended')
    tree.heading('#0', text='Sound'); tree.heading('type', text='Type'); tree.heading('size', text='Size')
    tree.column('#0', width=300); tree.column('type', width=70); tree.column('size', width=80, anchor='e')
    ys = ttk.Scrollbar(left, orient='vertical', command=tree.yview); tree.configure(yscrollcommand=ys.set)
    tree.pack(side='left', fill='both', expand=True); ys.pack(side='right', fill='y')

    right = ttk.Frame(pw); pw.add(right, weight=3)
    abar = ttk.Frame(right, padding=4); abar.pack(side='top', fill='x')
    play_btn = ttk.Button(abar, text='Play', width=6, command=lambda: audio_toggle())
    play_btn.pack(side='left')
    play_btn.state(['disabled'])
    atime = ttk.Label(abar, text='', width=16); atime.pack(side='left', padx=6)
    ainfo = ttk.Label(abar, text=''); ainfo.pack(side='left', padx=6)
    acanvas = tk.Canvas(right, bg='#282c34', highlightthickness=0, height=260)
    acanvas.pack(side='top', fill='both', expand=True)
    info = tk.Text(right, wrap='word', height=11, font=('Courier', 10)); info.pack(side='bottom', fill='x')

    player = aud.Player()
    audio_state = dict(audio=None, entry=None, job=None, pos=0.0)

    # ---- helpers
    def set_status(s):
        status.set(s)

    def pump():
        try:
            while True:
                msgq.get_nowait()()
        except queue.Empty:
            pass
        root.after(50, pump)

    def ui(fn):
        msgq.put(fn)

    task_local = threading.local()

    def cancelled():
        return getattr(task_local, 'id', state['task']) != state['task']

    def bg(work, done=None, label='Working…'):
        """Run work() off the UI thread, then done(result) on it. A new task replaces the running one."""
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

    # ---- loading and the list
    def open_folder():
        d = filedialog.askdirectory(title='Select the Transformers Prime game folder')
        if not d:
            return
        sar = aud.find_sound_archive(d)
        if not sar:
            messagebox.showerror('Not found', 'No sound archive (TRF_SOUND.bfsar) found under:\n' + d); return

        def work():
            return aud.index_audio(aud.SoundArchive(sar))

        def done(ents):
            audio_stop(); clear_view()
            state['entries'] = ents
            n = sum(e.sound.kind != 'warc-wave' for e in ents)
            missing = sum(1 for e in ents if e.sound.kind == 'stream' and not e.size)
            set_status('%d sounds, %d raw waves%s' % (
                n, len(ents) - n, ', %d stream files missing' % missing if missing else ''))
            fill_tree()
        bg(work, done, 'Indexing audio…')

    def shown(e):
        if e.sound.kind == 'warc-wave' and not show_raw.get():
            return False
        if e.sound.kind == 'sequence' and not show_seq.get():
            return False
        q = search.get().strip().lower()
        return not q or q in e.name.lower()

    def fill_tree():
        tree.delete(*tree.get_children())
        state['items'] = {}
        folders = {}
        for e in state['entries']:
            if not shown(e):
                continue
            parent = ''
            for k in range(1, len(e.group) + 1):              # Voice / English / f12 ...
                path = e.group[:k]
                if path not in folders:
                    folders[path] = tree.insert(parent, 'end', text=path[-1], open=False, values=('folder', ''))
                parent = folders[path]
            iid = tree.insert(parent, 'end', text=e.name, values=(e.sound.kind, _sz(e.size) if e.size else ''))
            state['items'][iid] = e
        if search.get().strip() and len(state['items']) <= 400:     # few search hits: show them all
            for f in folders.values():
                tree.item(f, open=True)

    search_job = [None]

    def on_search(*a):
        if search_job[0] is not None:
            root.after_cancel(search_job[0])
        search_job[0] = root.after(300, fill_tree)
    search.trace_add('write', on_search)

    def selected_entries():
        out, seen = [], set()

        def walk(iid):
            if iid in seen:
                return
            seen.add(iid)
            if iid in state['items']:
                out.append(state['items'][iid])
            for c in tree.get_children(iid):
                walk(c)
        for iid in tree.selection():
            walk(iid)
        return out

    # ---- viewer
    def clear_view(text=''):
        audio_state.update(audio=None, entry=None, pos=0.0)
        acanvas.delete('all'); atime.config(text=''); ainfo.config(text='')
        play_btn.state(['disabled'])
        info.delete('1.0', 'end')
        if text:
            acanvas.create_text(20, 20, anchor='nw', fill='#c8ccd4', width=600, text=text)

    def on_select(ev=None):
        sel = [state['items'][i] for i in tree.selection() if i in state['items']]
        if not sel or sel[0] is state['sel']:
            return
        e = sel[0]; state['sel'] = e
        audio_stop()
        if e.sound.kind == 'sequence':
            clear_view('This is a sequence: note data that the game plays through an instrument bank, not recorded '
                       'audio, so it can\'t be previewed or exported as a WAV.\n\nIts instrument samples are in '
                       'Wave archives.')
            ainfo.config(text='%s   sequence' % e.name)
            set_status('%s  (sequence)' % e.name)
            return

        def work():
            try:
                return e, e.archive.load(e.sound), None
            except FileNotFoundError as ex:          # e.g. a stream file missing from content/stream
                return e, None, str(ex)

        def done(r):
            e2, a, err = r
            if state['sel'] is not e2:
                return
            if a is None:
                clear_view('Can\'t load this sound:\n%s' % err)
                ainfo.config(text='%s   %s' % (e2.name, e2.sound.kind))
                set_status(err)
                return
            audio_state.update(audio=a, entry=e2, pos=0.0)
            play_btn.state(['!disabled'])
            src = e2.archive.describe(e2.sound)
            dur = a['frames'] / float(a['rate'])
            ainfo.config(text='%s   %s, %d Hz, %d ch%s' % (
                e2.name, a['codec'], a['rate'], a['channels'],
                ', loops from %.2f s' % (a['loop_start'] / a['rate']) if a['loop'] else ''))
            info.delete('1.0', 'end')
            info.insert('end', '\n'.join([
                'Sound:      %s' % e2.name,
                'Type:       %s' % e2.sound.kind + (' (%s)' % src if src else ''),
                'Folder:     %s' % ' / '.join(e2.group),
                'Player:     %s' % (e2.sound.player or '-'),
                'Codec:      %s' % a['codec'],
                'Rate:       %d Hz' % a['rate'],
                'Channels:   %d' % a['channels'] + ('  (%d tracks %s: Play uses the first, Export writes one WAV '
                                                   'each)' % (len(a['tracks']), a['tracks'])
                                                   if len(a['tracks']) > 1 else ''),
                'Length:     %.3f s (%d samples)' % (dur, a['frames']),
                'Loop:       %s' % ('from %d (%.3f s) to the end' % (a['loop_start'], a['loop_start'] / a['rate'])
                                    if a['loop'] else 'no'),
                '', 'Export writes a 16-bit WAV (loop points kept in a smpl chunk).']))
            draw_wave()
            set_status('%s  (%s, %.2f s)' % (e2.name, e2.sound.kind, dur))
        bg(work, done, 'Loading %s…' % e.name)

    def draw_wave():
        acanvas.delete('all')
        a = audio_state['audio']
        if a is None:
            return
        w = max(acanvas.winfo_width(), 200); h = max(acanvas.winfo_height(), 120)
        s = a['samples']
        n, ch = s.shape
        if not n:
            return
        lane = h / float(ch)
        cols = min(w, n)
        edges = np.linspace(0, n, cols + 1).astype(np.int64)
        for c in range(ch):
            x = s[:, c].astype(np.int32)
            mx = np.maximum.reduceat(x, edges[:-1]); mn = np.minimum.reduceat(x, edges[:-1])
            mid = lane * (c + 0.5); amp = lane * 0.45 / 32768.0
            acanvas.create_line(0, mid, w, mid, fill='#3e4451')
            for i in range(cols):
                xx = i * w / float(cols)
                acanvas.create_line(xx, mid - mx[i] * amp, xx, mid - mn[i] * amp + 1, fill='#61afef')
        if a['loop']:
            lx = a['loop_start'] / float(n) * w
            acanvas.create_line(lx, 0, lx, h, fill='#e5c07b', dash=(4, 3))
            acanvas.create_text(lx + 4, 4, anchor='nw', fill='#e5c07b', text='loop')
        audio_cursor()

    def audio_cursor():
        a = audio_state['audio']
        acanvas.delete('cursor')
        if a is None:
            return
        w = max(acanvas.winfo_width(), 200); h = max(acanvas.winfo_height(), 120)
        dur = a['frames'] / float(a['rate'])
        x = audio_state['pos'] / dur * w if dur else 0
        acanvas.create_line(x, 0, x, h, fill='#e06c75', width=2, tags='cursor')
        atime.config(text='%.2f / %.2f s' % (audio_state['pos'], dur))

    def audio_tick():
        audio_state['job'] = None
        p = player.position()
        if p is None:                                   # finished
            audio_stop(rewind=True)
            return
        audio_state['pos'] = p
        audio_cursor()
        audio_state['job'] = root.after(40, audio_tick)

    def audio_toggle(at=None):
        a = audio_state['audio']
        if a is None:
            return
        if player.start is not None and at is None:
            audio_stop(); return
        start = audio_state['pos'] if at is None else at
        if start >= a['frames'] / float(a['rate']) - 0.01:
            start = 0.0
        if not player.play(aud.preview(a), start):
            messagebox.showinfo('Audio', 'No audio player found. On Linux, install pulseaudio-utils (paplay), '
                                         'alsa-utils (aplay) or ffmpeg (ffplay). Export still works.')
            return
        audio_state['pos'] = start
        play_btn.config(text='Stop')
        if audio_state['job'] is None:
            audio_state['job'] = root.after(40, audio_tick)

    def audio_stop(rewind=False):
        player.stop()
        if audio_state['job'] is not None:
            root.after_cancel(audio_state['job']); audio_state['job'] = None
        if rewind:
            audio_state['pos'] = 0.0
        play_btn.config(text='Play')
        audio_cursor()

    def audio_seek(ev):
        a = audio_state['audio']
        if a is None:
            return
        w = max(acanvas.winfo_width(), 200)
        t = max(0.0, min(1.0, ev.x / float(w))) * a['frames'] / float(a['rate'])
        if player.start is not None:
            audio_toggle(at=t)
        else:
            audio_state['pos'] = t; audio_cursor()

    acanvas.bind('<Button-1>', audio_seek)
    acanvas.bind('<Configure>', lambda ev: draw_wave())
    tree.bind('<<TreeviewSelect>>', on_select)
    root.bind('<space>', lambda ev: None if root.focus_get() is sentry else audio_toggle())

    # ---- export
    def export_selected():
        ents = [e for e in selected_entries() if e.sound.kind != 'sequence']
        if not ents:
            messagebox.showinfo('Nothing to export', 'Select one or more sounds or folders in the list.'); return
        outdir = os.path.join(EXPORT_DIR, 'audio')

        def work():
            fails = []
            for i, e in enumerate(ents):
                if cancelled():
                    break
                try:
                    aud.export_audio(e, outdir)
                except Exception as ex:
                    fails.append('%s: %s' % (e.name, ex))
                if not cancelled() and (i % 20 == 0 or i + 1 == len(ents)):
                    ui(lambda i=i, e=e: set_status('Exporting %d/%d  %s' % (i + 1, len(ents), e.name)))
            return fails

        def done(fails):
            set_status('Export finished: %d sounds, %d failed  ->  %s' % (len(ents), len(fails), outdir))
            if fails:
                messagebox.showwarning('Some exports failed', '\n'.join(fails[:30]))
        bg(work, done, 'Exporting…')

    def on_close():
        player.stop()
        root.destroy()
    root.protocol('WM_DELETE_WINDOW', on_close)

    pump()
    root.mainloop()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'extract':
        sys.exit(cli_extract(sys.argv[2:]))
    run_gui()

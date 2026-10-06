"""eegkit: general EEG/GSR/resp helpers with NO per-file hardcoding.

Everything the old ``qc/research.py`` did via hand-curated bad-channel lists is
*derived* here:
  * dead (disconnected) channels  -> detected as perfectly flat lines near the
    amplifier rail (peak-to-peak below a threshold), both whole-file (static) and
    mid-session spans (dynamic; died / revived during recording).
  * crash tails (mass flat at end) -> detected and trimmed, not hardcoded.
  * stage boundaries -> from LSL event annotations when present; otherwise a
    data-driven estimate (eyes-closed alpha + breathing surge + muscle), or an
    explicit override supplied by ``dataset.STAGE_OVERRIDES``.
Only EEG channel *labels* (10-20 names) are fixed constants; they are filtered to
whatever actually exists in a file, so any 10-20 montage works.
"""
from __future__ import annotations

import re
import numpy as np
import mne
from scipy.signal import butter, sosfilt, welch, hilbert, find_peaks, savgol_filter

mne.set_log_level("ERROR")

# ---- constants ---------------------------------------------------------------
AUX_HINT = ("pesp", "resp", "gsr", "ecg", "eog", "emg", "breath")
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (13, 30)}
FIXED_DUR = 180.0            # protocol fixed stages, s
FLAT_PTP_UV = 1.0           # ptp below this (per window / whole file) => disconnected
MIN_RUN_S = 4.0             # min length of a mid-session dead/alive span
WIN_S = 2.0                 # window used for span & segmentation
MIN_ALIVE_S = 30.0          # a real live stretch this long => the channel WAS connected
WINDOW_LIVE_FRAC = 0.7      # live over [t0,t1) if >70% of the 2 s sub-windows are live

# standard 10-20 labels (filtered to what a file actually contains)
OCCIP = ("Oz", "O1", "O2")
FRONTAL_MIDLINE = ("Fz", "FCz", "Cz")
MUSCLE_HB = ("C3", "C4", "Cz", "T3", "T4", "T5", "T6", "Pz")
REGIONS = {
    "Broca_L": ("F7", "F3", "FC3"), "Broca_R": ("F8", "F4", "FC4"),
    "Wern_L": ("T3", "T5"), "Wern_R": ("T4", "T6"),
}

STAGE_DEFS = [
    ("rest_open1", "open_raw1_start", "open_raw1_finish"),
    ("rest_close1", "open_raw1_finish", "close_raw1_finish"),
    ("hypervent", "hyper_start", "hyper_finish"),
    ("rest_open2", "hyper_finish", "open_raw2_finish"),
    ("reading", "reading_start", "reading_finish"),
    ("rest_open3", "open_raw3_start", "open_raw3_finish"),
    ("speech", "speech_start", "speech_finish"),
    ("rest_open4", "open_raw4_start", "open_raw4_finish"),
    ("rest_close4", "open_raw4_finish", "close_raw4_finish"),
]

# per-raw computed properties are cached by id(raw)
_CACHE: dict[int, dict] = {}


# ---- loading & channel typing ----------------------------------------------
def load(path, preload=True) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_edf(path, preload=preload, infer_types=True)
    # mark belt/GSR aux channels as misc so they never enter the EEG average-ref
    aux_map = {ch: "misc" for ch in raw.ch_names
               if any(h in ch.lower() for h in ("pesp", "resp2", "gsr"))}
    if aux_map:
        raw.set_channel_types(aux_map)
    return raw


def eeg_channels(raw) -> list[str]:
    types = raw.get_channel_types()
    return [c for c, t in zip(raw.ch_names, types)
            if t == "eeg" and not any(h in c.lower() for h in AUX_HINT)]


def aux_channels(raw) -> list[str]:
    types = raw.get_channel_types()
    return [c for c, t in zip(raw.ch_names, types)
            if t != "eeg" or any(h in c.lower() for h in ("pesp", "resp2", "gsr"))]


# ---- dead-channel & crash-tail detection (cached) ---------------------------
def _stats(raw):
    """Compute full-file ptp, 2s-window ptp matrix, static-dead, dynamic spans, trim."""
    key = id(raw)
    if key in _CACHE:
        return _CACHE[key]
    fs = raw.info["sfreq"]
    eeg = eeg_channels(raw)
    idx = [raw.ch_names.index(c) for c in eeg]
    x = raw.get_data()[idx] * 1e6                       # uV, EEG only
    W = int(WIN_S * fs)
    nw = x.shape[1] // W
    win = x[:, :nw * W].reshape(len(eeg), nw, W)
    win_ptp = win.max(axis=2) - win.min(axis=2)         # (nch, nw)
    dead_win = win_ptp < FLAT_PTP_UV
    live_win = ~dead_win
    full_ptp = x.max(axis=1) - x.min(axis=1)
    live_frac = live_win.mean(axis=1)                    # fraction live overall

    # longest contiguous run of LIVE 2 s windows per channel (s).  A channel never
    # connected this session shows only the amp-settle ramp (< MIN_ALIVE_S), so it
    # is *static*; one that lived a real stretch then died/revived (POz, FC3) is
    # *dynamic*.  Cannot use 'alive at t=0' because of that first ramp.
    def _longest_live(b):
        mx = j = 0
        n = len(b)
        while j < n:
            if b[j]:
                k = j
                while k < n and b[k]:
                    k += 1
                mx = max(mx, k - j)
                j = k
            else:
                j += 1
        return mx * WIN_S
    max_live = np.array([_longest_live(live_win[i]) for i in range(len(eeg))])
    static_dead = [eeg[i] for i in range(len(eeg)) if max_live[i] < MIN_ALIVE_S]

    # dynamic (died / revived) spans = dead runs >= MIN_RUN_S inside non-static channels
    spans = []        # (channel, t0, t1) in seconds
    for i, ch in enumerate(eeg):
        if ch in static_dead:
            continue
        b = dead_win[i]
        j = 0
        while j < nw:
            if b[j]:
                k = j
                while k < nw and b[k]:
                    k += 1
                if (k - j) * WIN_S >= MIN_RUN_S and j > 0:      # ignore a span at t=0
                    spans.append((ch, j * WIN_S, k * WIN_S))
                j = k
            else:
                j += 1

    # crash / mass-flat tail: first late window where >=50% EEG channels go flat at once
    dead_frac = dead_win.mean(axis=0)
    trim = None
    for w in range(nw):
        if dead_frac[w] >= 0.5 and w > nw * 0.5:                # only consider late half
            trim = w * WIN_S
            break

    _CACHE[key] = dict(fs=fs, eeg=eeg, static_dead=static_dead, spans=spans,
                       dead_frac=dead_frac, trim=trim, full_ptp=full_ptp,
                       live_frac=live_frac, dead_win=dead_win, max_live=max_live)
    return _CACHE[key]


def static_dead(raw) -> list[str]:
    return _stats(raw)["static_dead"]


def dynamic_spans(raw) -> list[tuple]:
    """(channel, t0, t1) spans that died/revived mid-session (excluding static dead)."""
    st = _stats(raw)
    return [s for s in st["spans"] if s[2] - s[1] >= MIN_RUN_S]


def channel_status(raw) -> dict:
    """{channel: 'active' | 'dead' | 'part'} — for the per-session electrode map."""
    st = _stats(raw)
    status = {}
    for i, c in enumerate(st["eeg"]):
        if c in st["static_dead"]:
            status[c] = "dead"
        elif st["live_frac"][i] >= 0.95:
            status[c] = "active"
        else:
            status[c] = "part"
    return status


def trim_time(raw) -> float | None:
    """End-time of a detected crash/mass-flat tail (drop data after this), else None."""
    return _stats(raw)["trim"]


def live_channels(raw, t0=None, t1=None) -> list[str]:
    """EEG channels live overall (no window), or live over [t0,t1).
    Over a window a channel is 'live' if >WINDOW_LIVE_FRAC of its 2 s sub-windows
    in [t0,t1) are live (majority test keeps flapping channels with brief dropouts,
    drops mostly-dead ones)."""
    st = _stats(raw)
    base = [c for c in st["eeg"] if c not in set(st["static_dead"])]
    if t0 is None or t1 is None:
        return base
    w0 = int(t0 / WIN_S)
    w1 = max(w0 + 1, int(np.ceil(t1 / WIN_S)))
    out = []
    for c in base:
        i = st["eeg"].index(c)
        dw = st["dead_win"][i][w0:w1]
        frac = 1.0 - dw.mean() if len(dw) else 0.0
        if frac >= WINDOW_LIVE_FRAC:
            out.append(c)
    return out


def get(raw, t0=None, t1=None):
    """Average-referenced EEG in uV restricted to live channels, sliced to [t0,t1)."""
    fs = _stats(raw)["fs"]
    live = live_channels(raw, t0, t1)
    idx = [raw.ch_names.index(c) for c in live]
    x = raw.get_data()[idx] * 1e6
    x = x - x.mean(axis=0, keepdims=True)
    if t0 is not None:
        x = x[:, int(t0 * fs):int(t1 * fs)]
    return x, live


# ---- spectral helpers -------------------------------------------------------
def band_pow(seg, fs, lo, hi, nperseg=None):
    nperseg = nperseg or min(seg.shape[-1], int(8 * fs))
    f, p = welch(seg, fs=fs, nperseg=nperseg, axis=-1)
    p = np.moveaxis(p, -1, 0)          # (nfreq, ...)
    m = (f >= lo) & (f <= hi)
    return np.trapezoid(p[m], f[m], axis=0)


def bp(x, fs, lo, hi, order=3):
    sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfilt(sos, x, axis=-1)


def iaf(raw, live, ec_t0, ec_t1):
    """Individual alpha frequency from occipital avg during eyes-closed rest."""
    fs = _stats(raw)["fs"]
    occ = [raw.ch_names.index(c) for c in OCCIP if c in raw.ch_names]
    x = raw.get_data()[occ][:, int(ec_t0 * fs):int(ec_t1 * fs)] * 1e6
    f, p = welch(x.mean(0), fs=fs, nperseg=8 * fs)
    ps = savgol_filter(p, 21, 3)
    m = (f >= 8) & (f <= 13)
    return float(f[m][np.argmax(ps[m])])


def resp_channel(raw, t0, t1):
    """Pick the aux belt channel that actually carries respiration (0.05-0.5 Hz)."""
    fs = _stats(raw)["fs"]
    best, bv = None, 0.0
    for c in aux_channels(raw):
        if "gsr" in c.lower():
            continue
        d = raw.get_data()[raw.ch_names.index(c)][int(t0 * fs):int(t1 * fs)]
        d = bp(d, fs, 0.05, 0.5)
        if d.std() < 0.05:
            continue
        f, p = welch(d, fs=fs, nperseg=min(len(d), int(20 * fs)))
        m = (f >= 0.08) & (f <= 0.5)
        v = np.trapezoid(p[m], f[m])
        if v > bv:
            best, bv = c, v
    return best, (np.sqrt(bv) if best else 0.0)


def blink_rate(fp_sig, fs, thr_uV=None):
    """Blinks/min from a frontal channel; adaptive threshold when thr not given."""
    d = bp(fp_sig, fs, 1, 10)
    if thr_uV is None:
        thr_uV = max(25.0, 0.8 * np.percentile(np.abs(d), 99.5))
    pk, _ = find_peaks(d, distance=int(0.4 * fs), prominence=thr_uV)
    return len(pk) / (len(d) / fs) * 60.0


def resp_phase_power(resp, eeg_sig, fs, band=(8, 12), nbins=8):
    """Respiratory-phase amplitude coupling (PAC) of an EEG band, in phase bins."""
    r = bp(resp, fs, 0.05, 0.5)
    phase = np.angle(hilbert(r))
    e = bp(eeg_sig, fs, *band)
    pw = e ** 2
    bins = np.floor((phase + np.pi) / (2 * np.pi) * nbins).astype(int) % nbins
    return np.array([pw[bins == b].mean() if (bins == b).any() else np.nan
                     for b in range(nbins)])


# ---- stages -----------------------------------------------------------------
def event_times(raw) -> dict:
    ev = {}
    for a in raw.annotations:
        d = str(a["description"])
        m = re.search(r"LSL\s+(\S+)", d)
        name = m.group(1) if m else d
        if name.endswith("_start") or name.endswith("_finish"):
            ev.setdefault(name, float(a["onset"]))
    return ev


def stages_from_markers(raw) -> list[tuple]:
    ev = event_times(raw)
    t_end = raw.n_times / _stats(raw)["fs"]
    out = []
    for name, s, e in STAGE_DEFS:
        if s in ev and e in ev and ev[e] > ev[s]:
            t0, t1 = max(0.0, ev[s]), min(t_end, ev[e])
            if t1 - t0 > 1.0:
                out.append((name, t0, t1))
    return out


def estimate_stages(raw) -> list[tuple]:
    """Data-driven boundary estimate for recordings that lack stage markers.
    Anchors: eyes-closed alpha block (rest_close1), breathing surge (hypervent),
    high-muscle engagement (speech).  Fixed protocol stages propagate at FIXED_DUR."""
    st = _stats(raw)
    fs = st["fs"]
    live = live_channels(raw)
    occ = [raw.ch_names.index(c) for c in OCCIP if c in live]
    mus = [raw.ch_names.index(c) for c in MUSCLE_HB if c in live]
    W = int(fs)                     # 1 s windows
    nw = raw.n_times // W
    x = raw.get_data() * 1e6        # uV, all channels
    alpha = np.array([band_pow(x[occ][:, w * W:(w + 1) * W], fs, 8, 12).mean()
                      if occ else 0.0 for w in range(nw)])
    muscle = np.array([band_pow(x[mus][:, w * W:(w + 1) * W], fs, 30, 80).mean()
                       if mus else 0.0 for w in range(nw)])
    rc, _ = resp_channel(raw, 0, nw * W / fs)
    if rc:
        rd = bp(x[raw.ch_names.index(rc)], fs, 0.05, 0.5)
        ramp = np.array([rd[w * W:(w + 1) * W].std() for w in range(nw)])
    else:
        ramp = np.zeros(nw)

    def runs(mask, min_s):
        out = []
        j = 0
        while j < nw:
            if mask[j]:
                k = j
                while k < nw and mask[k]:
                    k += 1
                if (k - j) >= min_s:
                    out.append((j * W / fs, k * W / fs))
                j = k
            else:
                j += 1
        return out

    a_med = np.median(alpha[alpha > 0]) if np.any(alpha > 0) else 0.0
    m_med = np.median(muscle) or 1e-6
    r_med = np.median(ramp) or 1e-6
    ec_runs = runs((alpha > 1.8 * a_med) & (muscle < 0.9 * m_med), 30)
    hv_runs = runs(ramp > 1.7 * r_med, 20)
    sp_runs = runs(muscle > 1.6 * m_med, 10)

    if not ec_runs or not hv_runs:
        return []
    rc0 = ec_runs[0][0]
    hv0 = next((t for t in hv_runs if t[0] >= rc0), hv_runs[0][0])
    seq = [("rest_open1", 0.0, rc0), ("rest_close1", rc0, hv0),
           ("hypervent", hv0, hv0 + FIXED_DUR),
           ("rest_open2", hv0 + FIXED_DUR, hv0 + 2 * FIXED_DUR)]
    read0 = next((t[0] for t in sp_runs if t[0] >= seq[-1][1]), seq[-1][1])
    seq.append(("reading", read0, read0 + FIXED_DUR))
    open3 = read0 + FIXED_DUR
    seq.append(("rest_open3", open3, open3 + FIXED_DUR))
    sp0 = next((t[0] for t in sp_runs if t[0] >= open3), open3)
    seq.append(("speech", sp0, sp0 + FIXED_DUR))
    o4 = sp0 + FIXED_DUR
    seq.append(("rest_open4", o4, min(o4 + FIXED_DUR, nw * W / fs)))
    return seq


def get_stages(raw, key: str | None = None):
    """Stages for a file: markers -> override (dataset metadata) -> estimate."""
    from .dataset import STAGE_OVERRIDES
    st = stages_from_markers(raw)
    if len(st) >= 3:
        return st
    if key and key in STAGE_OVERRIDES:
        return STAGE_OVERRIDES[key]
    est = estimate_stages(raw)
    return est if est else st


# ---- speaking / listening segmentation (muscle proxy) -----------------------
def speak_listen_mask(raw, t0, t1):
    """Classify 2 s windows of an interview into speaking / listening.
    Returns (tw, z, cls) with cls: +1 speak, -1 listen, 0 ambiguous."""
    st = _stats(raw)
    fs = st["fs"]
    x, live = get(raw, t0, t1)
    hbi = [live.index(c) for c in MUSCLE_HB if c in live]
    W, S = int(WIN_S * fs), int(1.0 * fs)
    hb = np.array([np.log(band_pow(x[hbi][:, a:a + W], fs, 30, 80).mean(0) + 1e-9)
                   for a in range(0, x.shape[1] - W, S)])
    tw = np.arange(len(hb)) * S / fs + t0
    z = (hb - np.median(hb)) / (1.4826 * np.median(np.abs(hb - np.median(hb))) + 1e-9)
    cls = np.where(z > 0.5, 1, np.where(z < 0.0, -1, 0))
    # drop runs shorter than MIN_RUN_S
    i = 0
    while i < len(cls):
        j = i
        while j < len(cls) and cls[j] == cls[i]:
            j += 1
        if cls[i] != 0 and (j - i) * S / fs < MIN_RUN_S:
            cls[i:j] = 0
        i = j
    return tw, z, cls

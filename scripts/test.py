import argparse
import csv
import functools
import json
import math
import os
import sys
import time
from fractions import Fraction

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import wavfile
from scipy.signal import lfilter, resample_poly


FS = 1024
N_FFT, HOP = 256, 64
FPS = FS // HOP
T0 = N_FFT / FS
N_LO, N_HI, N_T = 32, 16, 160
LO_BAND = (20.0, 460.0)
HI_BAND = (460.0, 3680.0)
EPS = 1e-10
GRID_S = 0.5
LABEL_HZ = 10
DURATION_HZ = 4
N_AVG = 10
SLOW_TAUS = (30.0, 60.0, 120.0)
R2_MIN = 0.95
GUARD_S = 1.0
RESAMPLE = {11378: (9, 100), 3012: (17, 50), 1024: (1, 1)}
STREAMS = {11378: (18, 25, 8192), 3012: (17, 25, 2048)}
N_Q = 7
Z_RANGE = 3.0
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
TAUS = np.array(QUANTILES)
MED = 3
ALPHA = BETA = 0.1
SURFACES = ["PH0", "PH10", "PH12", "Polished MC", "Polished_Copper"]
CHF_REF = {"PH0": 183.87, "PH10": 218.56, "PH12": 202.07, "Polished MC": 203.48, "Polished_Copper": 102.63}
CHF_CATEGORIES = ("step_chf", "ramp_chf")
NEAR_CHF = 0.7
FOLDS = [
    [
        "SS_PH0_150", "SS_PH0_195", "SS_PH0_75", "SS_PH0_CHF", "SS_PH10_170", "SS_PH10_70", "SS_PH12_150", "SS_PH12_50",
        "SS_Polished MC_105", "SS_Polished MC_180", "SS_Polished MC_30", "SS_Polished_Copper_15",
        "SS_Polished_Copper_90", "TR_PH0_CHF",
    ],
    [
        "SS_PH0_15", "SS_PH0_165", "SS_PH0_90", "SS_PH10_190", "SS_PH10_210", "SS_PH10_90", "SS_PH10_CHF",
        "SS_PH12_170", "SS_PH12_70", "SS_Polished MC_120", "SS_Polished MC_195", "SS_Polished MC_45",
        "SS_Polished_Copper_30", "TR_PH10_CHF",
    ],
    [
        "SS_PH0_105", "SS_PH0_180", "SS_PH0_30", "SS_PH10_110", "SS_PH10_15", "SS_PH12_190", "SS_PH12_90",
        "SS_PH12_CHF", "SS_Polished MC_135", "SS_Polished MC_60", "SS_Polished_Copper_45", "TR_PH12_CHF",
    ],
    [
        "SS_PH0_120", "SS_PH0_45", "SS_PH10_130", "SS_PH10_30", "SS_PH12_110", "SS_PH12_15", "SS_Polished MC_150",
        "SS_Polished MC_214", "SS_Polished MC_75", "SS_Polished MC_CHF", "SS_Polished_Copper_60", "TR_Polished MC_CHF",
    ],
    [
        "SS_PH0_135", "SS_PH0_60", "SS_PH10_150", "SS_PH10_50", "SS_PH12_130", "SS_PH12_30", "SS_Polished MC_15",
        "SS_Polished MC_165", "SS_Polished MC_90", "SS_Polished_Copper_105", "SS_Polished_Copper_75",
        "SS_Polished_Copper_CHF", "TR_Polished_Copper_CHF",
    ],
]
EXPERIMENTS = [key for fold in FOLDS for key in fold]
BATCH = 512
DELTA_MAX_DB = 15.0
DELTA_TOL_DB = 0.01
MAPE_MIN = 5.0
HIGH_LOAD = 0.7
HIGH_LOAD_MIN_WINDOWS = 5


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_edges(f_lo, f_hi, n_mels):
    return mel_to_hz(np.linspace(hz_to_mel(f_lo), hz_to_mel(f_hi), n_mels + 2))


HI_EDGES = mel_edges(*HI_BAND, N_HI)


def mel_filterbank(freqs, edges, bands):
    fb = np.zeros((len(edges) - 2, len(freqs)))
    for i in bands:
        lo, ce, hi = edges[i], edges[i + 1], edges[i + 2]
        fb[i] = np.clip(np.minimum((freqs - lo) / (ce - lo), (hi - freqs) / (hi - ce)), 0.0, None)
        fb[i] /= fb[i].sum()
    return fb


def resample(x, up, down):
    return x if up == down else resample_poly(x, up, down)


def frame_features(x):
    x = np.asarray(x, dtype=np.float64)
    K = (len(x) - N_FFT) // HOP + 1
    if K < 1:
        raise ValueError("recording shorter than one STFT window")
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(K)[:, None]
    w = np.hanning(N_FFT)
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / FS)
    fb = mel_filterbank(freqs, mel_edges(*LO_BAND, N_LO), range(N_LO))
    band = (freqs >= LO_BAND[0]) & (freqs <= LO_BAND[1])
    scale = 2.0 / (FS * (w ** 2).sum())
    logmel = np.empty((N_LO, K), dtype=np.float32)
    power = np.empty(K, dtype=np.float64)
    df = freqs[1] - freqs[0]
    for s in range(0, K, 4096):
        seg = x[idx[s:s + 4096]]
        seg = seg - seg.mean(axis=1, keepdims=True)
        P = (np.abs(np.fft.rfft(seg * w, axis=1)) ** 2) * scale
        logmel[:, s:s + 4096] = (10.0 * np.log10(P @ fb.T + EPS)).T.astype(np.float32)
        power[s:s + 4096] = P[:, band].sum(axis=1) * df
    level = (10.0 * np.log10(power + EPS)).astype(np.float32)
    return logmel, level, power


def band_features(x, fs, n_frames):
    n_fft, hop = fs // 4, fs // 16
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
    avail = HI_EDGES[2:] <= 0.9 * fs / 2.0
    fb = mel_filterbank(freqs, HI_EDGES, np.flatnonzero(avail))
    K = min(n_frames, (len(x) - n_fft) // hop + 1)
    w = np.hanning(n_fft)
    scale = 2.0 / (fs * (w ** 2).sum())
    out = np.full((N_HI, n_frames), np.nan, dtype=np.float32)
    for s in range(0, K, 512):
        k = np.arange(s, min(K, s + 512))
        seg = x[np.arange(n_fft)[None, :] + hop * k[:, None]]
        seg = seg - seg.mean(axis=1, keepdims=True)
        P = (np.abs(np.fft.rfft(seg * w, axis=1)) ** 2) * scale
        out[:, k] = (10.0 * np.log10(P @ fb.T + EPS)).T.astype(np.float32)
    if K < n_frames:
        out[:, K:] = out[:, K - 1:K]
    out[~avail] = np.nan
    return out, avail


def stream_plan(fs):
    cls = int(round(fs))
    if cls in RESAMPLE:
        low = RESAMPLE[cls]
    elif fs < FS:
        raise ValueError(f"sampling rate {fs:g} Hz is below {FS} Hz")
    else:
        ratio = Fraction(FS / fs).limit_denominator(1000)
        low = (ratio.numerator, ratio.denominator)
    if cls in STREAMS:
        return low, STREAMS[cls]
    stream = 8192 if fs >= 8192 else 2048 if fs >= 2048 else None
    if stream is None:
        return low, None
    ratio = Fraction(stream / fs).limit_denominator(1000)
    return low, (ratio.numerator, ratio.denominator, stream)


def record_features(x, fs, duration=None):
    x = np.asarray(x, dtype=np.float64)
    duration = len(x) / fs if duration is None else duration
    low, hi = stream_plan(fs)
    wav = resample(x, *low)[:int(np.floor(duration * FS + 1e-6))].astype(np.float32)
    logmel, level, power = frame_features(wav)
    record = dict(logmel=logmel, level=level, power=power)
    if hi is not None:
        up, down, fs_hi = hi
        stream = resample(x, up, down)[:int(np.floor(duration * fs_hi + 1e-6))]
        record["hi"], record["hi_avail"] = band_features(stream, fs_hi, logmel.shape[1])
    return record


def slow_level(power, tau):
    a = np.exp(-1.0 / (FPS * tau))
    num = lfilter([1.0], [1.0, -a], power)
    den = lfilter([1.0], [1.0, -a], np.ones_like(power))
    return 10.0 * np.log10(num / den + EPS)


def read_features(path):
    z = np.load(path, allow_pickle=False)
    names = set(z.files)
    records = []
    for meta in json.loads(str(z["index"])):
        record = dict(meta)
        for name in ("logmel", "level", "power", "hi", "hi_avail", "q", "r2"):
            if f"{meta['key']}__{name}" in names:
                record[name] = z[f"{meta['key']}__{name}"]
        records.append(record)
    return records


class WindowBank:
    def __init__(self, records, device="cpu"):
        self.device = torch.device(device)
        self.meta = [{k: v for k, v in r.items() if not isinstance(v, np.ndarray)} for r in records]
        self.keys = [m["key"] for m in self.meta]
        self.index = {k: i for i, k in enumerate(self.keys)}
        lm, lv, ps, hi, hi_ok, ls, offset = [], [], [], [], [], [], 0
        tab = {n: [] for n in ("e", "t", "k_end", "y", "r", "valid")}
        for e, r in enumerate(records):
            K = r["logmel"].shape[1]
            lm.append(r["logmel"])
            lv.append(r["level"])
            ps.append(r["power"])
            if "hi" in r:
                hi.append(np.nan_to_num(r["hi"], nan=0.0))
                hi_ok.append(r["hi_avail"].astype(bool))
            else:
                hi.append(np.zeros((N_HI, K), np.float32))
                hi_ok.append(np.zeros(N_HI, bool))
            ls.append(np.stack([slow_level(r["power"].astype(np.float64), tau) for tau in SLOW_TAUS]))
            t_first = np.ceil((T0 + (N_T - 1) / FPS) / GRID_S) * GRID_S
            t_last = (K - 1) / FPS + T0
            if "duration_s" in r:
                t_last = min(t_last, np.floor(r["duration_s"] * DURATION_HZ) / DURATION_HZ)
            if "q" in r:
                t_last = min(t_last, (len(r["q"]) - 1) / LABEL_HZ)
            t = np.arange(t_first, t_last + 1e-9, GRID_S)
            if "q" in r:
                q, r2 = r["q"].astype(np.float64), r["r2"].astype(np.float64)
                j_end = np.rint(t * LABEL_HZ).astype(np.int64)
                jj = j_end[:, None] - np.arange(N_AVG)[None, :]
                y, r2min = q[jj].mean(axis=1), r2[jj].min(axis=1)
                valid = np.ones(len(t), bool)
                if r["category"] in CHF_CATEGORIES:
                    valid = (t <= r["t_exc"] - GUARD_S + 1e-9) & (r2min >= R2_MIN)
            else:
                y, valid = np.full(len(t), np.nan), np.ones(len(t), bool)
            chf = CHF_REF.get(r.get("surface"), np.nan)
            for name, v in (("e", np.full(len(t), e)), ("t", t), ("k_end", np.rint((t - T0) * FPS).astype(np.int64) + offset),
                            ("y", y), ("r", y / chf), ("valid", valid)):
                tab[name].append(v)
            offset += K
        for name, v in tab.items():
            setattr(self, name, np.concatenate(v))
        dev = self.device
        self.LM = torch.from_numpy(np.concatenate(lm, axis=1)).to(dev)
        self.LV = torch.from_numpy(np.concatenate(lv)).to(dev)
        self.PS = torch.from_numpy(np.concatenate(ps).astype(np.float32)).to(dev)
        self.HI = torch.from_numpy(np.concatenate(hi, axis=1).astype(np.float32)).to(dev)
        self.HI_OK = torch.from_numpy(np.stack(hi_ok)).to(dev)
        self.LS = torch.from_numpy(np.concatenate(ls, axis=1).astype(np.float32)).to(dev)
        self.e_t = torch.from_numpy(self.e).to(dev)
        self.k_end_t = torch.from_numpy(self.k_end).to(dev)
        self.k_range = torch.arange(-(N_T - 1), 1, device=dev)

    def windows(self, keys):
        e = [self.index[k] for k in keys if k in self.index]
        return np.flatnonzero(np.isin(self.e, e) & self.valid)

    def batch(self, idx, gain=0.0):
        idx = torch.as_tensor(idx, device=self.device)
        ke = self.k_end_t[idx]
        kk = ke[:, None] + self.k_range[None, :]
        logmel = self.LM[:, kk].permute(1, 0, 2) + gain
        level = self.LV[kk] + gain
        power = self.PS[kk].mean(dim=1) * (10.0 ** (gain / 10.0))
        lbar = 10.0 * torch.log10(power + EPS)
        ok = self.HI_OK[self.e_t[idx]]
        hi = (self.HI[:, kk].permute(1, 0, 2) + gain) * ok[:, :, None]
        ctx = (self.LS[:, ke].T + gain) - lbar[:, None]
        return {"logmel": logmel, "level": level, "lbar": lbar, "hi": hi, "hi_mask": ok, "ctx": ctx}


class Stats(nn.Module):
    def __init__(self, stats):
        super().__init__()
        for name, value in stats.items():
            self.register_buffer(name, value.detach().clone().float())


class LinTailLaw(nn.Module):
    def __init__(self, n_knots):
        super().__init__()
        self.register_buffer("c", torch.linspace(-Z_RANGE, Z_RANGE, n_knots))
        self.h = 2.0 * Z_RANGE / n_knots
        self.a = nn.Parameter(torch.tensor([-1.6, -1.2, -0.6, 0.0, 0.6, 1.2, 1.6]))
        self.w0 = nn.Parameter(torch.full((N_Q,), -1.0))
        self.u = nn.Parameter(torch.full((N_Q, 2 * n_knots), -4.0))
        self.n_raw = 2 + 2 * n_knots

    def basis(self, z):
        t = (z[:, None] - self.c[None, :]) / self.h
        return torch.cat([self.h * F.softplus(t), -self.h * F.softplus(-t)], dim=1)

    def raw(self):
        return torch.cat([self.a[:, None], self.w0[:, None], self.u], dim=1)

    def unsorted(self, z, raw=None):
        r = self.raw()[None] if raw is None else raw
        return r[..., 0] + F.softplus(r[..., 1]) * z[:, None] + (F.softplus(r[..., 2:]) * self.basis(z)[:, None, :]).sum(-1)

    def quantiles(self, z, raw=None):
        return torch.sort(self.unsorted(z, raw), dim=1).values


def window_statistics(x, n_data):
    data = x[:, :n_data]
    f = [data.mean(dim=3).flatten(1), data.std(dim=3).flatten(1)]
    if x.shape[1] > n_data:
        f.append(x[:, -1, :, 0])
    return torch.cat(f, dim=1)


class MLPNet(nn.Module):
    def __init__(self, n_data, n_rows, use_hi, width, layers, dropout):
        super().__init__()
        self.n_data = n_data
        d = 2 * n_data * n_rows + (n_rows if use_hi else 0)
        blocks = []
        for _ in range(layers):
            blocks += [nn.Linear(d, width), nn.GELU(), nn.Dropout(dropout)]
            d = width
        self.net = nn.Sequential(*blocks)
        self.out_dim = width

    def forward(self, x):
        return self.net(window_statistics(x, self.n_data))


class RFFNet(nn.Module):
    def __init__(self, n_data, n_rows, use_hi, dim, gamma, dropout):
        super().__init__()
        self.n_data = n_data
        d = 2 * n_data * n_rows + (n_rows if use_hi else 0)
        g = torch.Generator().manual_seed(0)
        self.register_buffer("W", torch.randn(d, dim, generator=g) * math.sqrt(2.0 * gamma))
        self.register_buffer("b", torch.rand(dim, generator=g) * 2.0 * math.pi)
        self.scale = math.sqrt(2.0 / dim)
        self.drop = nn.Dropout(dropout)
        self.out_dim = dim

    def forward(self, x):
        return self.drop(self.scale * torch.cos(window_statistics(x, self.n_data) @ self.W + self.b))


class MonoSpec(nn.Module):
    def __init__(self, stats, p):
        super().__init__()
        self.s = Stats(stats)
        self.use_hi, self.use_ctx = bool(p["use_hi"]), bool(p["use_ctx"])
        self.n_rows = N_LO + (N_HI if self.use_hi else 0)
        self.n_side = 3 if self.use_ctx else 0
        self.base = LinTailLaw(int(p["knots"]))
        encoder, dropout = p["encoder"], float(p["dropout"])
        self.backbone = RFFNet(2, self.n_rows, self.use_hi, int(p["rff_dim"]), float(p["rff_gamma"]), dropout) \
            if encoder.startswith("rff") else None
        self.stat = MLPNet(2, self.n_rows, self.use_hi, int(p["stat_width"]), int(p["stat_layers"]), dropout) \
            if encoder.endswith("stat") else None
        d = sum(m.out_dim for m in (self.backbone, self.stat) if m is not None) + self.n_side
        self.norm = nn.LayerNorm(d)
        self.hyper = nn.Linear(d + 2 * self.n_rows + 1 + self.n_side, N_Q * self.base.n_raw)
        nn.init.zeros_(self.hyper.weight)
        nn.init.zeros_(self.hyper.bias)

    def spec(self, b):
        s = self.s
        B = b["logmel"].shape[0]
        ref = b["level"][:, None, :]
        raw = b["logmel"] - ref
        mu, sd = s.shape_mu, s.shape_sd
        full = None
        if self.use_hi:
            raw = torch.cat([raw, b["hi"] - ref], dim=1)
            mu, sd = torch.cat([mu, s.hishape_mu]), torch.cat([sd, s.hishape_sd])
            full = torch.cat([torch.ones(B, N_LO, dtype=torch.bool, device=raw.device), b["hi_mask"]], dim=1)
        x = (raw - mu[None, :, None]) / sd[None, :, None]
        if full is not None:
            x = x * full[:, :, None]
        dlev = ((b["level"] - b["lbar"][:, None]) - s.dlev_mu) / s.dlev_sd
        x = torch.stack([x, dlev[:, None, :].expand_as(x)], dim=1)
        if full is not None:
            x = torch.cat([x, full[:, None, :, None].expand(B, 1, self.n_rows, x.shape[3]).to(x.dtype)], dim=1)
        return x

    def side(self, b):
        return (b["ctx"] - self.s.ctx_mu) / self.s.ctx_sd if self.use_ctx else None

    def z(self, b):
        return (b["lbar"] - self.s.lbar_mu) / self.s.lbar_sd

    def raw(self, b):
        x, v = self.spec(b), self.side(b)
        h = [m(x) for m in (self.backbone, self.stat) if m is not None] + ([v] if v is not None else [])
        sh, fl = x[:, 0], x[:, 1, 0]
        skip = [sh.mean(2), sh.std(2), fl.std(1, keepdim=True)] + ([v] if v is not None else [])
        out = self.hyper(torch.cat([self.norm(torch.cat(h, dim=1))] + skip, dim=1))
        return out.view(-1, N_Q, self.base.n_raw)

    def forward(self, b):
        return self.base.quantiles(self.z(b), self.raw(b)) * self.s.y_sd + self.s.y_mu


@torch.no_grad()
def predict_quantiles(models, bank, idx, gain=0.0):
    out = [[] for _ in models]
    for s in range(0, len(idx), BATCH):
        b = bank.batch(idx[s:s + BATCH], gain)
        for chunk, model in zip(out, models):
            chunk.append(model(b).float().cpu().numpy())
    return np.stack([np.concatenate(c) if c else np.zeros((0, N_Q), np.float32) for c in out])


def load_ensemble(models_dir, config, protocol, fold, device):
    path = os.path.join(models_dir, f"protocol{protocol}_fold{fold}.pt")
    if not os.path.isfile(path):
        fail(f"checkpoint {path} not found")
    settings = next((f["members"] for f in config.get(f"protocol{protocol}", {}).get("folds", []) if f["fold"] == fold), None)
    if settings is None:
        fail(f"{os.path.join(models_dir, 'config.json')} has no settings for protocol {protocol}, fold {fold}")
    try:
        states = torch.load(path, map_location=device, weights_only=True)["members"]
    except Exception as error:
        fail(f"cannot read {path}: {error}")
    models = []
    for state, member in zip(states, settings):
        model = MonoSpec({k[2:]: v for k, v in state.items() if k.startswith("s.")}, member["params"]).to(device)
        model.load_state_dict(state)
        models.append(model.eval())
    return models


def pinball(q, y):
    d = y[:, None] - q
    return np.maximum(TAUS[None, :] * d, (TAUS[None, :] - 1.0) * d)


def crps_windows(q, y):
    return 2.0 * pinball(q, y).mean(axis=1)


def experiment_metrics(q, y, e, r=None, lo=None, hi=None):
    lo = q[:, 0] if lo is None else lo
    hi = q[:, -1] if hi is None else hi
    err = q[:, MED] - y
    crps = crps_windows(q, y)
    ape = np.where(y >= MAPE_MIN, np.abs(err) / np.maximum(y, MAPE_MIN), np.nan)
    rows = {}
    for k in np.unique(e):
        m = e == k
        row = dict(windows=int(m.sum()), rmse=float(np.sqrt(np.mean(err[m] ** 2))), mae=float(np.mean(np.abs(err[m]))),
                   mape=float(100.0 * np.nanmean(ape[m])) if np.isfinite(ape[m]).any() else np.nan,
                   crps=float(crps[m].mean()), coverage=float(np.mean((y[m] >= lo[m]) & (y[m] <= hi[m]))),
                   width=float(np.mean(hi[m] - lo[m])), rmse_high=np.nan)
        if r is not None:
            h = m & (r >= HIGH_LOAD)
            if h.sum() >= HIGH_LOAD_MIN_WINDOWS:
                row["rmse_high"] = float(np.sqrt(np.mean(err[h] ** 2)))
        rows[int(k)] = row
    return rows


def mean_over_experiments(rows, key, subset=None):
    values = [v[key] for k, v in rows.items() if subset is None or k in subset]
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def experiment_scores(q, y, e):
    s = np.maximum(q[:, 0] - y, y - q[:, -1])
    return {int(k): float(np.quantile(s[e == k], 1.0 - BETA)) for k in np.unique(e)}


def conformal_margin(scores):
    v = np.sort(np.asarray(list(scores), dtype=np.float64))
    k = int(np.ceil((len(v) + 1) * (1.0 - ALPHA)))
    return float("inf") if len(v) == 0 or k > len(v) else float(v[k - 1])


def solve_offset(models, bank, idx, gain, target, strict=False):
    def error(delta):
        return float(np.mean(predict_quantiles(models, bank, idx, gain + delta).mean(0, dtype=np.float64)[:, MED] - target))

    lo, hi = -DELTA_MAX_DB, DELTA_MAX_DB
    f_lo, f_hi = error(lo), error(hi)
    if f_lo >= 0.0 or f_hi <= 0.0:
        return None if strict else lo if abs(f_lo) < abs(f_hi) else hi
    while hi - lo > DELTA_TOL_DB:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if error(mid) < 0.0 else (lo, mid)
    return 0.5 * (lo + hi)


def anchor_key(surface, keys):
    steady = [k for k in keys if k.startswith(f"SS_{surface}_") and not k.endswith("_CHF")]
    return min(steady, key=lambda k: float(k.rsplit("_", 1)[1])) if steady else None


def note(message):
    print("note:", message)


def fmt(v):
    return v if isinstance(v, str) else f"{v:d}" if isinstance(v, (int, np.integer)) else f"{v:.3f}"


def show(title, header, rows):
    cells = [header] + [[fmt(v) for v in row] for row in rows]
    widths = [max(len(c[i]) for c in cells) for i in range(len(header))]
    print(f"\n{title}")
    for i, c in enumerate(cells):
        print("  " + "  ".join(v.rjust(w) for v, w in zip(c, widths)))
        if i == 0:
            print("  " + "  ".join("-" * w for w in widths))


def write_csv(out, name, header, rows):
    if out is None:
        return
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, name), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def present(bank, keys):
    return [k for k in keys if k in bank.index]


def evaluate_protocol1(bank, load, config, folds, out):
    parts = []
    for k in folds:
        keys = present(bank, FOLDS[k])
        if not keys:
            note(f"protocol 1, fold {k}: no experiments in the data file, skipped")
            continue
        idx = bank.windows(keys)
        parts.append((k, idx, predict_quantiles(load(1, k), bank, idx).astype(np.float64)))
    if not parts:
        return
    n_members = min(len(p[2]) for p in parts)
    idx = np.concatenate([p[1] for p in parts])
    fold = np.concatenate([np.full(len(p[1]), p[0]) for p in parts])
    members = np.concatenate([p[2][:n_members] for p in parts], axis=1)
    q = np.concatenate([p[2].mean(0) for p in parts])
    stored = "c_hat" in config["protocol1"]
    if not stored:
        note(NO_MARGINS)
    c_hat = np.array(config["protocol1"]["c_hat"]) if stored else np.zeros(len(FOLDS))
    e, y, r = bank.e[idx], bank.y[idx], bank.r[idx]
    rows = experiment_metrics(q, y, e, r, q[:, 0] - c_hat[fold], q[:, -1] + c_hat[fold])
    category = {k: bank.meta[k]["category"] for k in rows}
    steady = {k for k, c in category.items() if c == "steady"}
    single = [mean_over_experiments(experiment_metrics(members[m], y, e), "rmse") for m in range(n_members)]
    summary = [("experiments", len(rows)), ("windows", len(idx)),
               ("RMSE (W/cm²)", mean_over_experiments(rows, "rmse")), ("MAE (W/cm²)", mean_over_experiments(rows, "mae")),
               ("MAPE (%, y >= 5 W/cm²)", mean_over_experiments(rows, "mape")), ("CRPS (W/cm²)", mean_over_experiments(rows, "crps")),
               ("coverage of the 90% interval", mean_over_experiments(rows, "coverage")),
               ("width of the 90% interval (W/cm²)", mean_over_experiments(rows, "width")),
               ("RMSE, steady runs", mean_over_experiments(rows, "rmse", steady)),
               ("RMSE, CHF records", mean_over_experiments(rows, "rmse", set(rows) - steady)),
               ("RMSE, windows with y/q_CHF >= 0.7", mean_over_experiments(rows, "rmse_high")),
               ("RMSE, single member (mean over members)", float(np.mean(single)))]
    show(f"Protocol 1: ensemble of {n_members} members, calibrated 90% interval, means over experiments", ["metric", "value"], summary)
    scores = experiment_scores(q, y, e)
    fold_of = {int(k): int(fold[e == k][0]) for k in scores}
    if stored and len(scores) == sum(map(len, FOLDS)):
        margins = [conformal_margin([s for k, s in scores.items() if fold_of[k] != f]) for f in range(len(FOLDS))]
        show("Conformal margins (W/cm²)", ["fold", "stored", "recomputed"], [[f, c_hat[f], margins[f]] for f in range(len(FOLDS))])
    write_csv(out, "protocol1_summary.csv", ["metric", "value"], summary)
    metrics = ["windows", "rmse", "mae", "mape", "crps", "coverage", "width", "rmse_high"]
    write_csv(out, "protocol1_experiments.csv", ["key", "surface", "category", "fold"] + metrics,
              [[bank.keys[k], bank.meta[k]["surface"], category[k], fold_of[k]] + [v[n] for n in metrics] for k, v in rows.items()])


def evaluate_gain_errors(bank, load, folds, gains, out):
    surface_of = np.array([m["surface"] for m in bank.meta])
    acc = {g: dict(e=[], y=[], plain=[], recalibrated=[]) for g in gains}
    offsets = []
    for k in folds:
        keys = present(bank, FOLDS[k])
        if not keys:
            continue
        idx = bank.windows(keys)
        models = load(1, k)
        outside = [x for j, f in enumerate(FOLDS) if j != k for x in f]
        for surface in SURFACES:
            test = idx[surface_of[bank.e[idx]] == surface]
            if not len(test):
                continue
            anchor = anchor_key(surface, outside)
            if anchor not in bank.index:
                note(f"gain errors, fold {k}, {surface}: anchor run {anchor} is not in the data file, skipped")
                continue
            a_idx = bank.windows([anchor])
            d0 = solve_offset(models, bank, a_idx, 0.0, bank.y[a_idx])
            for g in gains:
                d = solve_offset(models, bank, a_idx, g, bank.y[a_idx])
                for name, gain in (("plain", g), ("recalibrated", g + d)):
                    acc[g][name].append(predict_quantiles(models, bank, test, gain).mean(0, dtype=np.float64))
                acc[g]["e"].append(bank.e[test])
                acc[g]["y"].append(bank.y[test])
                offsets.append([k, surface, anchor, g, d, abs(d + g - d0)])
    if not offsets:
        return
    table = []
    for g in gains:
        e, y = np.concatenate(acc[g]["e"]), np.concatenate(acc[g]["y"])
        rmse = [mean_over_experiments(experiment_metrics(np.concatenate(acc[g][n]), y, e), "rmse") for n in ("plain", "recalibrated")]
        dev = [o[5] for o in offsets if o[3] == g]
        table.append([f"{g:+g}"] + rmse + [float(np.mean(dev)), float(np.max(dev))])
    header = ["gain error (dB)", "RMSE as trained", "RMSE recalibrated", "offset deviation mean (dB)", "offset deviation max (dB)"]
    show("Protocol 1 test windows with a gain error, one-record recalibration (W/cm²)", header, table)
    write_csv(out, "gain_errors.csv", header, table)
    write_csv(out, "gain_offsets.csv", ["fold", "surface", "anchor", "gain_db", "delta_db", "deviation_db"], offsets)


def evaluate_protocol2(bank, load, folds, out):
    table = []
    for k in folds:
        surface = SURFACES[k]
        anchor = anchor_key(surface, EXPERIMENTS)
        scored = present(bank, [x for x in EXPERIMENTS if x.startswith(f"SS_{surface}_") and not x.endswith("_CHF") and x != anchor])
        if anchor not in bank.index or not scored:
            note(f"protocol 2, {surface}: anchor run {anchor} or the scored runs are not in the data file, skipped")
            continue
        models = load(2, k)
        a_idx, idx = bank.windows([anchor]), bank.windows(scored)
        delta = solve_offset(models, bank, a_idx, 0.0, bank.y[a_idx])
        rows = experiment_metrics(predict_quantiles(models, bank, idx, delta).mean(0, dtype=np.float64), bank.y[idx], bank.e[idx])
        table.append([surface, anchor, len(scored), len(idx)] + [mean_over_experiments(rows, n) for n in ("rmse", "mae", "crps")])
    if not table:
        return
    mean = [float(np.mean([r[i] for r in table])) for i in range(4, 7)]
    runs, windows = sum(r[2] for r in table), sum(r[3] for r in table)
    show("Protocol 2: held-out surfaces, steady runs, one-record recalibration (W/cm², means over runs, then over surfaces)",
         ["surfaces", "runs", "windows", "RMSE", "MAE", "CRPS"], [[len(table), runs, windows] + mean])
    write_csv(out, "protocol2_surfaces.csv", ["surface", "anchor", "runs", "windows", "rmse", "mae", "crps"], table + [["mean", "", runs, windows] + mean])


def pick_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(name)
        torch.zeros(1, device=device)
    except (RuntimeError, AssertionError) as error:
        fail(f"device {name} is not available: {str(error).splitlines()[0]}")
    return device


def run_evaluate(args):
    start = time.time()
    if not os.path.isfile(args.data):
        fail(f"{args.data} not found; build it with 'python scripts/train.py prepare --raw <folder with the Dataverse files>' or use data/sample.npz")
    if not all(math.isfinite(g) for g in args.gains):
        fail("--gains must be finite numbers")
    device = pick_device(args.device)
    config = read_config(args.models)
    try:
        bank = WindowBank(read_features(args.data), device)
    except (OSError, ValueError, KeyError) as error:
        fail(f"{args.data} is not a MonoSpec feature file: {error}")
    load = functools.lru_cache(maxsize=None)(lambda protocol, fold: load_ensemble(args.models, config, protocol, fold, device))
    print(f"MonoSpec evaluation: {len(bank.keys)} experiments in {args.data}, device {device}")
    if "1" in args.protocol:
        evaluate_protocol1(bank, load, config, args.folds, args.out)
    if "gain" in args.protocol:
        evaluate_gain_errors(bank, load, args.folds, args.gains, args.out)
    if "2" in args.protocol:
        evaluate_protocol2(bank, load, args.folds, args.out)
    print(f"\nfinished in {time.time() - start:.1f} s")


LEVEL_RANGE = (5.367, 26.055)
CALIBRATION_MAX_WINDOWS = 2000
MAX_AMPLITUDE = 1e30
OUTPUT_COLUMNS = ["time_s", "window_level_db", "z", "q05", "q10", "q25", "q50", "q75", "q90", "q95", "median_w_cm2", "lower90_w_cm2",
                  "upper90_w_cm2", "interval_width_w_cm2", "sensitivity_w_cm2_per_db", "gain_offset_db"]
RISK_COLUMNS = ["q_over_qchf", "risk_level"]
TRUTH_COLUMNS = ["measured_w_cm2", "error_w_cm2", "abs_error_w_cm2", "covered"]
CSV_DECIMALS = {"time_s": 4, "window_level_db": 4, "z": 5, "gain_offset_db": 5, "q_over_qchf": 5}
SEPARATORS = str.maketrans({",": " ", ";": " ", "\t": " ", "\r": " "})
LEVEL_ESTIMATED = "no frame-level column was given; the frame level is estimated from the 32 mel bands"
INTEGER_WAV = "integer WAV samples are scaled to the range -1 to 1"
NO_MARGINS = "config.json has no conformal margins; the interval is the quantile range from 5% to 95%"
RECALIBRATION_FAILED = ("recalibration failed: no gain offset within ±15 dB makes the median match the heat flux of the recalibration record; "
                        "check the units of the recording and the known heat flux")


def fail(message):
    raise SystemExit(f"error: {message}")


def read_config(models):
    path = os.path.join(models, "config.json")
    if not os.path.isfile(path):
        fail(f"{path} not found")
    try:
        with open(path) as fh:
            return json.load(fh)
    except ValueError as error:
        fail(f"{path} is not valid JSON: {error}")


def data_line(text, pos):
    while pos < len(text):
        end = text.find("\n", pos)
        end = len(text) if end < 0 else end
        tokens = text[pos:end].translate(SEPARATORS).split()
        if tokens and not text[pos:end].lstrip().startswith("#"):
            return tokens, pos, end + 1
        pos = end + 1
    return None, pos, pos


def read_table(path):
    with open(path, encoding="utf-8-sig") as fh:
        text = fh.read()
    tokens, start, nxt = data_line(text, 0)
    try:
        [float(v) for v in tokens or []]
    except ValueError:
        tokens, start, nxt = data_line(text, nxt)
    if tokens is None:
        fail(f"{path} contains no numeric row")
    try:
        values = np.array(text[start:].translate(SEPARATORS).split(), dtype=np.float64)
    except ValueError:
        fail(f"{path} contains a value that is not a number")
    if len(values) % len(tokens):
        fail(f"{path}: the number of values is not a multiple of the {len(tokens)} columns of the first row")
    return values.reshape(-1, len(tokens))


def logmel_record(table):
    if table.shape[1] not in (N_LO, N_LO + 1, N_LO + N_HI, N_LO + N_HI + 1):
        fail(f"a log-mel file needs {N_LO}, {N_LO + 1}, {N_LO + N_HI} or {N_LO + N_HI + 1} columns, found {table.shape[1]}")
    if len(table) < N_T:
        fail(f"a window needs {N_T} frames, found {len(table)}")
    logmel = table[:, :N_LO].T.astype(np.float32)
    if not np.isfinite(logmel).all():
        fail(f"the {N_LO} low-band columns must be finite")
    if table.shape[1] in (N_LO + 1, N_LO + N_HI + 1):
        level = table[:, -1].astype(np.float32)
        if not np.isfinite(level).all():
            fail("the level column must be finite")
        power = np.maximum(10.0 ** (level.astype(np.float64) / 10.0) - EPS, 0.0)
    else:
        note(LEVEL_ESTIMATED)
        power = (LO_BAND[1] - LO_BAND[0]) / N_LO * (10.0 ** (logmel.astype(np.float64) / 10.0)).sum(axis=0)
        level = (10.0 * np.log10(power + EPS)).astype(np.float32)
    record = dict(logmel=logmel, level=level, power=power)
    if table.shape[1] >= N_LO + N_HI:
        hi = table[:, N_LO:N_LO + N_HI].T.astype(np.float32)
        avail = np.isfinite(hi).all(axis=1)
        if avail.any():
            record["hi"], record["hi_avail"] = np.where(avail[:, None], hi, np.nan), avail
    return record


def wav_samples(path, column):
    fs, data = wavfile.read(path)
    if data.size == 0:
        fail(f"{path} contains no samples")
    channels = 1 if data.ndim == 1 else data.shape[1]
    if (column or 0) >= channels:
        fail(f"{path} has {channels} channel{'s' * (channels > 1)}; --column {column} is out of range")
    x = data if data.ndim == 1 else data[:, column or 0]
    if x.dtype.kind in "iu":
        half = 128.0 if x.dtype == np.uint8 else 0.0
        x = (x.astype(np.float64) - half) / (half or float(np.iinfo(x.dtype).max) + 1.0)
        note(INTEGER_WAV)
    return float(fs), x.astype(np.float64)


def waveform_record(x, rate, scale):
    x = x * scale
    if not np.isfinite(x).all():
        fail("the waveform contains NaN or infinite samples")
    if np.abs(x).max() > MAX_AMPLITUDE:
        fail("the waveform amplitude is too large")
    return record_features(x, rate)


def read_recording(path, fmt, rate, column, scale):
    if fmt == "wav":
        fs, x = wav_samples(path, column)
        return waveform_record(x, rate or fs, scale)
    table = read_table(path)
    if fmt == "logmel":
        return logmel_record(table)
    n = table.shape[1]
    if n > 2 and column is None:
        fail("a waveform CSV with more than two columns needs --column")
    if column is not None and column >= n:
        fail(f"{path} has {n} columns; --column {column} is out of range")
    if n == 2:
        if len(table) < 2 or not (np.diff(table[:, 0]) > 0).all():
            fail("the first column of a two-column waveform must be increasing time in seconds")
        if rate is None:
            rate = (len(table) - 1) / (table[-1, 0] - table[0, 0])
            note(f"the sampling rate was taken from the time column: {rate:.3f} Hz")
    if rate is None:
        fail("a waveform CSV without a time column needs --rate")
    return waveform_record(table[:, n - 1 if column is None else column], rate, scale)


def make_bank(path, key, args, device):
    fmt = args.format or ("wav" if path.lower().endswith(".wav") else "csv")
    try:
        record = read_recording(path, fmt, args.rate, args.column, args.scale)
    except (ValueError, OSError, EOFError) as error:
        fail(f"{path}: {error}")
    bank = WindowBank([dict(key=key, **record)], device)
    idx = bank.windows([key])
    if not len(idx):
        fail(f"{path} is shorter than {np.ceil((T0 + (N_T - 1) / FPS) / GRID_S) * GRID_S:g} s")
    return bank, idx


@torch.no_grad()
def window_details(models, bank, idx, gain):
    q, z, s, level = [], [], [], []
    for start in range(0, len(idx), BATCH):
        b = bank.batch(idx[start:start + BATCH], gain)
        level.append(b["lbar"].cpu().numpy())
        q.append(np.stack([m(b).float().cpu().numpy() for m in models]).mean(0, dtype=np.float64))
        z.append(np.stack([m.z(b).cpu().numpy() for m in models]).mean(0, dtype=np.float64))
        s.append(np.stack([median_sensitivity(m, b).cpu().numpy() for m in models]).mean(0, dtype=np.float64))
    return np.concatenate(level).astype(np.float64), np.concatenate(z), np.concatenate(q), np.concatenate(s)


def median_sensitivity(model, b):
    z, raw, base = model.z(b), model.raw(b), model.base
    rank = torch.argsort(base.unsorted(z, raw), dim=1)[:, MED]
    t = (z[:, None] - base.c[None, :]) / base.h
    slope = F.softplus(raw[..., 1]) + (F.softplus(raw[..., 2:]) * torch.cat([torch.sigmoid(t), torch.sigmoid(-t)], dim=1)[:, None, :]).sum(-1)
    return slope.gather(1, rank[:, None])[:, 0] * model.s.y_sd / model.s.lbar_sd


def measured_heat_flux(path, t):
    table = read_table(path)
    if table.shape[1] > 2:
        fail("the measured heat flux needs two columns: time_s and heat_flux_w_cm2")
    times, values = (np.arange(len(table)) / LABEL_HZ, table[:, 0]) if table.shape[1] == 1 else (table[:, 0], table[:, 1])
    if len(times) > 1 and not (np.diff(times) > 0).all():
        fail("the time column of the measured heat flux must be increasing")
    keep = np.isfinite(values)
    times, values = times[keep], values[keep]
    total = np.concatenate([[0.0], np.cumsum(values)])
    hi = np.searchsorted(times, t + 1e-9, side="right")
    lo = np.searchsorted(times, t - 1.0 + 1e-9, side="right")
    return np.where(hi > lo, (total[hi] - total[lo]) / np.maximum(hi - lo, 1), np.nan)


def risk_level(median, upper, chf_ref):
    return np.where(upper >= chf_ref, "alert", np.where(median / chf_ref >= NEAR_CHF, "near_chf", "normal"))


def csv_cell(name, v):
    if isinstance(v, str):
        return v
    return "" if not math.isfinite(v) else f"{v:.{CSV_DECIMALS.get(name, 5)}f}"


def check_predict_arguments(args):
    for name, value in (("--rate", args.rate), ("--scale", args.scale), ("--chf-ref", args.chf_ref),
                        ("--calibration-heat-flux", args.calibration_heat_flux)):
        if value is not None and not (math.isfinite(value) and value > 0.0):
            fail(f"{name} must be a positive finite number")
    if not math.isfinite(args.gain_db):
        fail("--gain-db must be a finite number")
    multiple = args.step / GRID_S
    if not (math.isfinite(multiple) and multiple >= 1.0 and abs(multiple - round(multiple)) < 1e-9):
        fail("--step must be a positive multiple of 0.5 s")
    if args.column is not None and args.column < 0:
        fail("--column must be 0 or larger")
    if args.calibration and args.calibration_heat_flux is None:
        fail("--calibration needs --calibration-heat-flux")
    if args.calibration_heat_flux is not None and not args.calibration:
        fail("--calibration-heat-flux needs --calibration")
    for name, path in (("--input", args.input), ("--calibration", args.calibration), ("--truth", args.truth)):
        if path and not os.path.isfile(path):
            fail(f"{name} file not found: {path}")
    if not os.path.isdir(args.models):
        fail(f"models directory not found: {args.models}")
    folder = os.path.dirname(os.path.abspath(args.output))
    if not os.path.isdir(folder) or os.path.isdir(args.output):
        fail(f"cannot write {args.output}: the output directory does not exist or the path is a directory")


def run_predict(args):
    check_predict_arguments(args)
    device = pick_device(args.device)
    config = read_config(args.models)
    protocol = int(args.protocol)
    folds = range(len(FOLDS)) if args.fold == "all" else [int(args.fold)]
    models = [m for k in folds for m in load_ensemble(args.models, config, protocol, k, device)]
    stored = protocol == 1 and "c_hat" in config.get("protocol1", {})
    if protocol == 1 and not stored:
        note(NO_MARGINS)
    margin = float(np.mean([config["protocol1"]["c_hat"][k] for k in folds])) if stored else 0.0
    bank, all_idx = make_bank(args.input, "recording", args, device)
    delta = 0.0
    if args.calibration:
        cal_bank, cal_idx = make_bank(args.calibration, "recalibration record", args, device)
        cal_idx = cal_idx[::max(1, math.ceil(len(cal_idx) / CALIBRATION_MAX_WINDOWS))]
        delta = solve_offset(models, cal_bank, cal_idx, args.gain_db, np.full(len(cal_idx), args.calibration_heat_flux), strict=True)
        if delta is None:
            fail(RECALIBRATION_FAILED)
    gain = args.gain_db + delta
    step = max(1, int(round(args.step / GRID_S)))
    idx = all_idx[::step]
    level, z, q, sensitivity = window_details(models, bank, idx, gain)
    t = bank.t[idx]
    lower, upper, median = q[:, 0] - margin, q[:, -1] + margin, q[:, MED]
    status = np.where(level < LEVEL_RANGE[0], "level_below_range", np.where(level > LEVEL_RANGE[1], "level_above_range", "ok"))
    columns = [t, level, z] + [q[:, j] for j in range(N_Q)] + [median, lower, upper, upper - lower, sensitivity, np.full(len(t), gain)]
    names = list(OUTPUT_COLUMNS)
    if args.chf_ref is not None:
        columns += [median / args.chf_ref, risk_level(median, upper, args.chf_ref)]
        names += RISK_COLUMNS
    columns.append(status)
    names.append("status")
    measured = None
    if args.truth:
        measured = measured_heat_flux(args.truth, t)
        columns += [measured, median - measured, np.abs(median - measured),
                    np.where(np.isfinite(measured), np.where((measured >= lower) & (measured <= upper), "True", "False"), "")]
        names += TRUTH_COLUMNS
    with open(args.output, "w", newline="") as fh:
        fh.write(",".join(names) + "\n")
        for row in zip(*columns):
            fh.write(",".join(csv_cell(n, v) for n, v in zip(names, row)) + "\n")
    outside = int((status != "ok").sum())
    if outside:
        note(f"{outside} of {len(idx)} windows have a window level outside the level range of the public dataset "
             f"({LEVEL_RANGE[0]:g} to {LEVEL_RANGE[1]:g} dB)")
    interval = "90% interval" if stored else "Q05 to Q95 band"
    spread = f"interval margin {margin:.3f} W/cm²" if stored else "quantile band Q05 to Q95"
    print(f"windows {len(idx)} ({t[0]:g} s to {t[-1]:g} s), ensemble of {len(models)} members, {spread}")
    print(f"mean estimate {median.mean():.2f} W/cm², mean {interval} [{lower.mean():.2f}, {upper.mean():.2f}] W/cm², gain offset {gain:+.3f} dB")
    if measured is not None and np.isfinite(measured).any():
        ok = np.isfinite(measured)
        print(f"against the measured heat flux: RMSE {np.sqrt(np.mean((median - measured)[ok] ** 2)):.2f} W/cm², "
              f"MAE {np.mean(np.abs(median - measured)[ok]):.2f} W/cm², coverage {np.mean(((measured >= lower) & (measured <= upper))[ok]):.3f}")
    print("written", args.output)


def build_parser(root):
    parser = argparse.ArgumentParser(prog="test.py", description="MonoSpec: evaluation on a feature file and inference on new recordings.")
    sub = parser.add_subparsers(dest="command")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--models", default=os.path.join(root, "models"), help="directory with config.json and the checkpoints (default: models/)")
    common.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N (default: auto)")
    ev = sub.add_parser("evaluate", parents=[common], help="reproduce the MonoSpec results of the paper on a feature file (default command)")
    ev.add_argument("--data", default=os.path.join(root, "data", "features.npz"), help="feature file (default: data/features.npz)")
    ev.add_argument("--protocol", nargs="+", choices=["1", "2", "gain"], default=["1", "gain", "2"],
                    help="parts to run: 1 (unseen experiments), gain (gain errors with recalibration), 2 (held-out surfaces); default: all three")
    ev.add_argument("--folds", nargs="+", type=int, choices=range(len(FOLDS)), default=list(range(len(FOLDS))), help="folds to run (default: all five)")
    ev.add_argument("--gains", nargs="+", type=float, default=[-3.0, -1.0, 1.0, 3.0], help="gain errors in dB for the gain test (default: -3 -1 1 3)")
    ev.add_argument("--out", default=None, help="directory for the CSV tables (default: none written)")
    pr = sub.add_parser("predict", parents=[common], help="estimate the heat flux of a new recording")
    pr.add_argument("--input", required=True, help="recording: WAV, waveform CSV or log-mel CSV")
    pr.add_argument("--format", choices=["wav", "csv", "logmel"], default=None, help="default: wav for .wav files, otherwise csv")
    pr.add_argument("--rate", type=float, default=None, help="sampling rate in Hz, at least 1024 (waveform CSV; overrides the WAV header)")
    pr.add_argument("--column", type=int, default=None, help="value column of a CSV or channel of a WAV, counted from 0 (default: last column, first channel)")
    pr.add_argument("--scale", type=float, default=1.0, help="factor applied to the waveform samples (default: 1)")
    pr.add_argument("--calibration", default=None,
                    help="recalibration record: recording of known mean heat flux (--format, --rate, --column and --scale apply to it too)")
    pr.add_argument("--calibration-heat-flux", type=float, default=None, help="mean heat flux of the recalibration record in W/cm²")
    pr.add_argument("--gain-db", type=float, default=0.0, help="gain added to the input level in dB (default: 0)")
    pr.add_argument("--protocol", choices=["1", "2"], default="1",
                    help="1: Protocol 1 fold ensembles with conformal interval, 2: Protocol 2 held-out-surface ensembles with the quantile band (default: 1)")
    pr.add_argument("--fold", default="all", choices=["all"] + [str(k) for k in range(len(FOLDS))],
                    help="ensemble of one fold, or all five folds (default), which averages their quantiles and uses the mean of their conformal margins")
    pr.add_argument("--chf-ref", type=float, default=None, help="critical heat flux of the surface in W/cm², enables the risk columns")
    pr.add_argument("--step", type=float, default=GRID_S, help="spacing of the output windows in s, a positive multiple of 0.5 (default: 0.5)")
    pr.add_argument("--truth", default=None, help="measured heat flux: CSV with the columns time_s, heat_flux_w_cm2")
    pr.add_argument("--output", default="prediction_results.csv", help="output CSV (default: prediction_results.csv)")
    return parser


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] not in ("evaluate", "predict", "-h", "--help"):
        argv = ["evaluate"] + argv
    args = build_parser(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).parse_args(argv)
    try:
        {"evaluate": run_evaluate, "predict": run_predict}[args.command](args)
    except (OSError, ValueError, EOFError) as error:
        fail(str(error))


if __name__ == "__main__":
    main()

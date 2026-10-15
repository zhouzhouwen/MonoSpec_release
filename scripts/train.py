import argparse
import json
import math
import os
import sys
import time
import warnings
from fractions import Fraction
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from scipy.signal import lfilter, resample_poly


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
BATCH = 512
HOP_TRAIN_STEADY_S = 5.0
HOP_VAL_STEADY_S = 2.5
STAT_BATCH = 2048
BASE_BATCH = 4096
HI_DROP = 0.3
WARMUP = 3
LR_PERIOD = 40
LR_MIN = 1e-6
PATIENCE = 10
BASE_STEPS = 1500
BASE_LR = 0.05
N_MEMBERS = 5
TPE_SEED = 1000
TPE_STARTUP = 10
PRUNER_STARTUP, PRUNER_WARMUP = 5, 8
SEARCH_SPACE = [
    ("lr", "log", 5e-5, 5e-3),
    ("weight_decay", "log", 1e-5, 1e-1),
    ("dropout", "float", 0.0, 0.4),
    ("batch_size", "cat", [32, 64, 128]),
    ("time_mask", "cat", [0, 8, 16, 32]),
    ("freq_mask", "cat", [0, 2, 4, 8]),
    ("use_hi", "cat", [False, True]),
    ("use_ctx", "cat", [False, True]),
    ("eq_aug", "cat", [0.0, 1.0, 2.0]),
    ("encoder", "cat", ["rff+stat", "rff", "stat"]),
    ("rff_dim", "cat", [1024, 2048, 4096]),
    ("rff_gamma", "cat", [0.001, 0.003, 0.01, 0.03]),
    ("stat_width", "cat", [64, 128, 256]),
    ("stat_layers", "cat", [1, 2, 3]),
    ("knots", "cat", [8, 16, 32]),
]
SIGMA_S, K_CU = 0.12, 392.0
TC_POS = np.array([0.0, 2.54, 5.08, 7.62]) * 1e-3
TC_POS_C = TC_POS - TC_POS.mean()
FIELDS = ("key", "surface", "role", "nominal_power", "duration_s", "t_exc", "q_mean", "q_event_max", "hydro_native_hz", "date")
RAW_EXPERIMENTS = (
    ("SS_PH0_105", "PH0", "steady", 105.0, 441.992829, None, 91.31982264699467, None, 11377.778, "2022/09/22"),
    ("SS_PH0_120", "PH0", "steady", 120.0, 443.992797, None, 104.68006503201218, None, 11377.778, "2022/09/22"),
    ("SS_PH0_135", "PH0", "steady", 135.0, 447.992733, None, 117.66719614486462, None, 11377.778, "2022/09/22"),
    ("SS_PH0_15", "PH0", "steady", 15.0, 446.992749, None, 11.093658342761362, None, 11377.778, "2022/09/22"),
    ("SS_PH0_150", "PH0", "steady", 150.0, 442.529912, None, 131.03183234130506, None, 11377.778, "2022/09/22"),
    ("SS_PH0_165", "PH0", "steady", 165.0, 438.569912, None, 144.52729464854144, None, 11377.778, "2022/09/22"),
    ("SS_PH0_180", "PH0", "steady", 180.0, 441.539912, None, 158.2391119384058, None, 11377.778, "2022/09/22"),
    ("SS_PH0_195", "PH0", "top_steady", 195.0, 441.539912, None, 171.1630529116112, None, 11377.778, "2022/09/22"),
    ("SS_PH0_30", "PH0", "steady", 30.0, 446.489912, None, 24.542991641841457, None, 11377.778, "2022/09/22"),
    ("SS_PH0_45", "PH0", "steady", 45.0, 443.992797, None, 38.158260857843, None, 11377.778, "2022/09/22"),
    ("SS_PH0_60", "PH0", "steady", 60.0, 442.529912, None, 51.14721946365261, None, 11377.778, "2022/09/22"),
    ("SS_PH0_75", "PH0", "steady", 75.0, 451.439912, None, 64.31669054385857, None, 11377.778, "2022/09/22"),
    ("SS_PH0_90", "PH0", "steady", 90.0, 442.992813, None, 77.70326153096009, None, 11377.778, "2022/09/22"),
    ("SS_PH0_CHF", "PH0", "step_chf", None, 1032.983358, 165.0, 45.568665723434684, 183.86872352755913, 11377.778, "2022/09/22"),
    ("SS_PH10_110", "PH10", "steady", 110.0, 474.9923, None, 105.60762500658115, None, 11377.778, "2022/11/23"),
    ("SS_PH10_130", "PH10", "steady", 130.0, 471.992348, None, 125.8620385709197, None, 11377.778, "2022/11/23"),
    ("SS_PH10_15", "PH10", "steady", 15.0, 474.9923, None, 10.182205206814437, None, 11377.778, "2022/11/23"),
    ("SS_PH10_150", "PH10", "steady", 150.0, 452.992653, None, 145.74888408425738, None, 11377.778, "2022/11/23"),
    ("SS_PH10_170", "PH10", "steady", 170.0, 451.439912, None, 166.2742322593574, None, 11377.778, "2022/11/23"),
    ("SS_PH10_190", "PH10", "steady", 190.0, 467.992412, None, 189.51507169298685, None, 11377.778, "2022/11/23"),
    ("SS_PH10_210", "PH10", "top_steady", 210.0, 455.399912, None, 206.57854107279252, None, 11377.778, "2022/11/23"),
    ("SS_PH10_30", "PH10", "steady", 30.0, 446.992749, None, 25.46676211279774, None, 11377.778, "2022/11/23"),
    ("SS_PH10_50", "PH10", "steady", 50.0, 429.659912, None, 45.359792593446315, None, 11377.778, "2022/11/23"),
    ("SS_PH10_70", "PH10", "steady", 70.0, 470.992364, None, 65.00979027739118, None, 11377.778, "2022/11/23"),
    ("SS_PH10_90", "PH10", "steady", 90.0, 454.992621, None, 85.40247418732937, None, 11377.778, "2022/11/23"),
    ("SS_PH10_CHF", "PH10", "step_chf", None, 566.990826, 65.69999694824219, 45.193938819056335, 218.5633795590551, 11377.778, "2022/11/23"),
    ("SS_PH12_110", "PH12", "steady", 110.0, 321.994752, None, 102.97415261549274, None, 11377.778, "2022/10/18"),
    ("SS_PH12_130", "PH12", "steady", 130.0, 328.99464, None, 122.5277724615325, None, 11377.778, "2022/10/18"),
    ("SS_PH12_15", "PH12", "steady", 15.0, 342.539912, None, 10.2674508836842, None, 11377.778, "2022/10/18"),
    ("SS_PH12_150", "PH12", "steady", 150.0, 325.709912, None, 142.37721452153303, None, 11377.778, "2022/10/18"),
    ("SS_PH12_170", "PH12", "steady", 170.0, 327.689912, None, 163.08494026341276, None, 11377.778, "2022/10/18"),
    ("SS_PH12_190", "PH12", "top_steady", 190.0, 328.99464, None, 186.2122152399993, None, 11377.778, "2022/10/18"),
    ("SS_PH12_30", "PH12", "steady", 30.0, 335.994528, None, 24.68284202378422, None, 11377.778, "2022/10/18"),
    ("SS_PH12_50", "PH12", "steady", 50.0, 334.619912, None, 43.94488256088517, None, 11377.778, "2022/10/18"),
    ("SS_PH12_70", "PH12", "steady", 70.0, 334.619912, None, 63.81640887556426, None, 11377.778, "2022/10/18"),
    ("SS_PH12_90", "PH12", "steady", 90.0, 328.679912, None, 83.253737281401, None, 11377.778, "2022/10/18"),
    ("SS_PH12_CHF", "PH12", "step_chf", None, 693.988791, 170.5, 62.2123740240082, 202.0695489763779, 11377.778, "2022/10/18"),
    ("SS_Polished MC_105", "Polished MC", "steady", 105.0, 382.999023, None, 91.68757254134098, None, 1024.0, "2023/12/01"),
    ("SS_Polished MC_120", "Polished MC", "steady", 120.0, 386.999023, None, 105.25961425029843, None, 1024.0, "2023/12/01"),
    ("SS_Polished MC_135", "Polished MC", "steady", 135.0, 386.999023, None, 118.85138537324393, None, 1024.0, "2023/12/01"),
    ("SS_Polished MC_15", "Polished MC", "steady", 15.0, 380.999023, None, 9.670031091581025, None, 1024.0, "2023/12/01"),
    ("SS_Polished MC_150", "Polished MC", "steady", 150.0, 544.999023, None, 133.0208595505411, None, 1024.0, "2023/12/01"),
    ("SS_Polished MC_165", "Polished MC", "steady", 165.0, 632.999023, None, 146.6068345939191, None, 1024.0, "2023/12/01"),
    ("SS_Polished MC_180", "Polished MC", "steady", 180.0, 383.999023, None, 159.8731818531598, None, 1024.0, "2023/12/01"),
    ("SS_Polished MC_195", "Polished MC", "steady", 195.0, 386.999023, None, 174.53792005430267, None, 1024.0, "2023/12/01"),
    ("SS_Polished MC_214", "Polished MC", "top_steady", 214.0, 398.999023, None, 191.25120175814774, None, 1024.0, "2023/12/01"),
    ("SS_Polished MC_30", "Polished MC", "steady", 30.0, 408.999023, None, 23.741014478165862, None, 1024.0, "2023/12/01"),
    ("SS_Polished MC_45", "Polished MC", "steady", 45.0, 384.999023, None, 37.24774366312197, None, 1024.0, "2023/12/01"),
    ("SS_Polished MC_60", "Polished MC", "steady", 60.0, 388.999023, None, 50.82116543820369, None, 1024.0, "2023/12/01"),
    ("SS_Polished MC_75", "Polished MC", "steady", 75.0, 419.999023, None, 64.71627076908148, None, 1024.0, "2023/12/01"),
    ("SS_Polished MC_90", "Polished MC", "steady", 90.0, 357.999023, None, 78.0542759183397, None, 1024.0, "2023/12/01"),
    ("SS_Polished MC_CHF", "Polished MC", "step_chf", None, 548.999023, 80.5999984741211, 52.10089898887086, 203.4799094173228, 1024.0, "2023/12/01"),
    ("SS_Polished_Copper_105", "Polished_Copper", "top_steady", 105.0, 595.990361, None, 93.67492384336968, None, 11377.778, "2022/10/06"),
    ("SS_Polished_Copper_15", "Polished_Copper", "steady", 15.0, 576.179912, None, 13.49895165986893, None, 11377.778, "2022/10/06"),
    ("SS_Polished_Copper_30", "Polished_Copper", "steady", 30.0, 525.689912, None, 27.27080226773488, None, 11377.778, "2022/10/06"),
    ("SS_Polished_Copper_45", "Polished_Copper", "steady", 45.0, 603.899912, None, 40.86177897610299, None, 11377.778, "2022/10/06"),
    ("SS_Polished_Copper_60", "Polished_Copper", "steady", 60.0, 704.988614, None, 54.05178705397071, None, 11377.778, "2022/10/06"),
    ("SS_Polished_Copper_75", "Polished_Copper", "steady", 75.0, 709.988534, None, 67.41390203609126, None, 11377.778, "2022/10/06"),
    ("SS_Polished_Copper_90", "Polished_Copper", "steady", 90.0, 599.990297, None, 80.45303744435583, None, 11377.778, "2022/10/06"),
    ("SS_Polished_Copper_CHF", "Polished_Copper", "step_chf", None, 727.988246, 70.30000305175781, 20.653701763725344, 102.62659851968509,
     11377.778, "2022/10/06"),
    ("TR_PH0_CHF", "PH0", "ramp_chf", None, 551.429912, 96.0999984741211, 31.72335486648177, 159.70156085039366, 11377.778, "2022/09/26"),
    ("TR_PH10_CHF", "PH10", "ramp_chf", None, 555.81998, 97.4000015258789, 31.37688985199633, 197.17437952755907, 3011.765, "2022/11/29"),
    ("TR_PH12_CHF", "PH12", "ramp_chf", None, 551.429912, 117.80000305175781, 31.5342021590998, 163.40147908661416, 11377.778, "2022/10/18"),
    ("TR_Polished MC_CHF", "Polished MC", "ramp_chf", None, 554.999023, 145.0, 59.61420717421401, 238.8848154330709, 1024.0, "2023/12/20"),
    ("TR_Polished_Copper_CHF", "Polished_Copper", "ramp_chf", None, 552.419912, 246.5, 44.781617030692985, 101.84641048818905,
     11377.778, "2022/10/06"),
)
COLUMN_ORDER = {
    "PH0|2022/09/22": [4, 1, 3, 2],
    "PH0|2022/09/26": [4, 1, 3, 2],
    "PH10|2022/11/23": [4, 2, 3, 1],
    "PH10|2022/11/29": [4, 2, 3, 1],
    "PH12|2022/10/18": [4, 2, 3, 1],
    "Polished MC|2023/12/01": [4, 1, 3, 2],
    "Polished MC|2023/12/20": [4, 1, 3, 2],
    "Polished_Copper|2022/10/06": [4, 1, 3, 2],
}


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


def fail(message):
    raise SystemExit(f"error: {message}")


def load_pandas():
    try:
        import pandas
    except ImportError:
        fail("pandas is required for 'prepare'; install it with 'pip install pandas'")
    return pandas


def read_lvm(fh, **kwargs):
    return load_pandas().read_csv(fh, sep="\t", header=None, engine="c", **kwargs)


def lvm_header(path):
    with open(path, "rb") as fh:
        for _ in range(60):
            line = fh.readline()
            if line.startswith(b"X_Value"):
                cols = line.decode(errors="replace").rstrip("\r\n").split("\t")[1:]
                return fh.tell(), [c for c in cols if c and c != "Comment"]
    raise ValueError(f"no X_Value line in the first 60 lines of {path}")


def select_channel(names, surface):
    if "Sound Pressure_0" in names:
        return "Sound Pressure_0"
    if surface == "Polished MC" and "Sound Pressure_1" in names:
        return "Sound Pressure_1"
    return names[0]


def read_hydrophone(path, surface):
    offset, names = lvm_header(path)
    with open(path, "rb") as fh:
        fh.seek(offset)
        table = read_lvm(fh, usecols=list(range(1, len(names) + 1)), dtype=np.float32, na_filter=False)
    return np.ascontiguousarray(table.to_numpy()[:, names.index(select_channel(names, surface))])


def change_points(path, n_channels=4):
    offset, _ = lvm_header(path)
    idx = [[] for _ in range(n_channels)]
    val = [[] for _ in range(n_channels)]
    prev, n_rows, t_last = None, 0, None
    with open(path, "rb") as fh:
        fh.seek(offset)
        for chunk in read_lvm(fh, usecols=list(range(7)), chunksize=1_500_000, dtype=np.float64):
            a = chunk.to_numpy()
            a = a[~np.isnan(a).any(axis=1)]
            if len(a) == 0:
                continue
            v = a[:, 1:7]
            if prev is None:
                vv, base = v, 0
                for c in range(n_channels):
                    idx[c].append(np.array([0], dtype=np.int64))
                    val[c].append(v[:1, c].copy())
            else:
                vv, base = np.vstack([prev[None, :], v]), n_rows - 1
            for c in range(n_channels):
                k = np.flatnonzero(vv[1:, c] != vv[:-1, c]) + 1
                if len(k):
                    idx[c].append(k.astype(np.int64) + base)
                    val[c].append(vv[k, c].copy())
            n_rows += len(a)
            prev = v[-1].copy()
            t_last = float(a[-1, 0])
    return [np.concatenate(i) for i in idx], [np.concatenate(v) for v in val], n_rows, t_last


def hold(idx, val, rows):
    return val[np.searchsorted(idx, rows, side="right") - 1]


def fit_profile(prof):
    m = prof.mean(1)
    pp = prof - m[:, None]
    sl = (pp * TC_POS_C).sum(1) / (TC_POS_C ** 2).sum()
    resid = pp - sl[:, None] * TC_POS_C
    ss = (pp ** 2).sum(1)
    r2 = 1.0 - (resid ** 2).sum(1) / np.maximum(ss, 1e-12)
    return -K_CU * sl / 1e4, r2


def heat_flux(path, order, duration_s):
    idx, val, n_rows, t_last = change_points(path)
    grid = np.unique(np.concatenate(idx))
    temps = np.stack([hold(idx[c], val[c], grid) for c in range(4)], axis=1)
    prof = temps[:, [c - 1 for c in order]]
    t_native = grid * (t_last / (n_rows - 1))
    n_out = int(np.floor(t_last * LABEL_HZ + 1e-9)) + 1
    t10 = np.arange(n_out) / LABEL_HZ
    lin = np.stack([np.interp(t10, t_native, prof[:, j]) for j in range(4)], axis=1)
    q, r2 = fit_profile(gaussian_filter1d(lin, SIGMA_S * LABEL_HZ, axis=0, mode="nearest"))
    keep = t10.astype(np.float32).astype(np.float64) <= duration_s + 1e-9
    return q.astype(np.float32)[keep], r2.astype(np.float32)[keep]


def raw_path(raw_dir, kind, key):
    load, rest = key.split("_", 1)
    return os.path.join(raw_dir, f"{'SS' if load == 'SS' else 'Tr'}_{kind}_{rest}.lvm")


def build_one(job):
    rec, raw_dir = job
    t0 = time.time()
    key = rec["key"]
    x = read_hydrophone(raw_path(raw_dir, "Hydrophone", key), rec["surface"])
    out = record_features(x, rec["hydro_native_hz"], rec["duration_s"])
    order = COLUMN_ORDER[f"{rec['surface']}|{rec['date']}"]
    out["q"], out["r2"] = heat_flux(raw_path(raw_dir, "Temperature", key), order, rec["duration_s"])
    return key, out, time.time() - t0


def index_entry(rec):
    category = "steady" if rec["role"] in ("steady", "top_steady") else rec["role"]
    return dict(key=rec["key"], surface=rec["surface"], category=category, role=rec["role"],
                nominal_power=rec["nominal_power"], duration_s=rec["duration_s"], t_exc=rec["t_exc"],
                q_mean=rec["q_mean"], q_event_max=rec["q_event_max"], hydro_native_hz=rec["hydro_native_hz"])


def run_prepare(args):
    table = [dict(zip(FIELDS, row)) for row in RAW_EXPERIMENTS]
    if args.only:
        unknown = sorted(set(args.only) - {r["key"] for r in table})
        if unknown:
            fail(f"unknown experiment keys: {', '.join(unknown)}")
        table = [r for r in table if r["key"] in args.only]
    if not os.path.isdir(args.raw):
        fail(f"raw data directory not found: {args.raw}")
    missing = [os.path.basename(p) for r in table for p in (raw_path(args.raw, k, r["key"]) for k in ("Hydrophone", "Temperature"))
               if not os.path.isfile(p)]
    if missing:
        fail(f"{len(missing)} raw files missing in {args.raw}, first: {', '.join(missing[:3])}")
    load_pandas()
    jobs = sorted(((r, args.raw) for r in table), key=lambda j: -j[0]["duration_s"])
    results, t0 = {}, time.time()

    def collect(item):
        key, out, seconds = item
        results[key] = out
        print(f"[{len(results):2d}/{len(jobs)}] {key:26s} frames {out['logmel'].shape[1]:6d} labels {len(out['q']):6d} "
              f"{seconds:6.1f}s  elapsed {time.time() - t0:7.1f}s", flush=True)

    if args.workers > 1:
        with Pool(args.workers) as pool:
            for item in pool.imap_unordered(build_one, jobs):
                collect(item)
    else:
        for job in jobs:
            collect(build_one(job))
    arrays = {f"{r['key']}__{name}": arr for r in table for name, arr in results[r["key"]].items()}
    arrays["index"] = np.array(json.dumps([index_entry(r) for r in table]))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print(f"wrote {len(table)} experiments to {args.out}")


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
        tab = {n: [] for n in ("e", "t", "k_end", "y", "r", "valid", "train", "val")}
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
            t_last = min((K - 1) / FPS + T0, np.floor(r["duration_s"] * DURATION_HZ) / DURATION_HZ, (len(r["q"]) - 1) / LABEL_HZ)
            t = np.arange(t_first, t_last + 1e-9, GRID_S)
            q, r2 = r["q"].astype(np.float64), r["r2"].astype(np.float64)
            j_end = np.rint(t * LABEL_HZ).astype(np.int64)
            jj = j_end[:, None] - np.arange(N_AVG)[None, :]
            y, r2min = q[jj].mean(axis=1), r2[jj].min(axis=1)
            transient = r["category"] in CHF_CATEGORIES
            valid = (t <= r["t_exc"] - GUARD_S + 1e-9) & (r2min >= R2_MIN) if transient else np.ones(len(t), bool)
            steps = np.rint((t - t_first) / GRID_S).astype(np.int64)
            train = steps % (1 if transient else int(round(HOP_TRAIN_STEADY_S / GRID_S))) == 0
            val = steps % (1 if transient else int(round(HOP_VAL_STEADY_S / GRID_S))) == 0
            chf = CHF_REF.get(r["surface"], np.nan)
            for name, v in (("e", np.full(len(t), e)), ("t", t), ("k_end", np.rint((t - T0) * FPS).astype(np.int64) + offset),
                            ("y", y), ("r", y / chf), ("valid", valid), ("train", train), ("val", val)):
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
        self.y_t = torch.from_numpy(self.y.astype(np.float32)).to(dev)
        self.k_range = torch.arange(-(N_T - 1), 1, device=dev)

    def windows(self, keys, grid=None):
        e = [self.index[k] for k in keys if k in self.index]
        mask = np.isin(self.e, e) & self.valid
        return np.flatnonzero(mask & getattr(self, grid)) if grid else np.flatnonzero(mask)

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
        return {"logmel": logmel, "level": level, "lbar": lbar, "hi": hi, "hi_mask": ok, "ctx": ctx, "y": self.y_t[idx]}

    @torch.no_grad()
    def norm_stats(self, idx):
        idx = torch.as_tensor(idx, device=self.device)
        acc = {n: [0.0, 0.0, 0.0] for n in ("shape", "dlev", "lbar", "y", "hishape", "ctx")}

        def add(name, v, dims, count):
            v, a = v.double(), acc[name]
            a[0] = a[0] + v.sum(dim=dims)
            a[1] = a[1] + (v ** 2).sum(dim=dims)
            a[2] = a[2] + count

        for s in range(0, len(idx), STAT_BATCH):
            b = self.batch(idx[s:s + STAT_BATCH])
            shape = b["logmel"] - b["level"][:, None, :]
            dlev = b["level"] - b["lbar"][:, None]
            m = b["hi_mask"].double()
            hishape = (b["hi"] - b["level"][:, None, :]).double() * m[:, :, None]
            add("shape", shape, (0, 2), shape.shape[0] * shape.shape[2])
            add("dlev", dlev, (0, 1), dlev.numel())
            add("lbar", b["lbar"], (0,), b["lbar"].shape[0])
            add("y", b["y"], (0,), b["y"].shape[0])
            add("hishape", hishape, (0, 2), m.sum(0) * hishape.shape[2])
            add("ctx", b["ctx"], (0,), b["ctx"].shape[0])
        stats = {}
        for name, (total, square, count) in acc.items():
            count = count.clamp(min=1.0) if name == "hishape" else max(count, 1.0) if name == "ctx" else count
            mu = total / count
            stats[f"{name}_mu"] = mu.float()
            stats[f"{name}_sd"] = torch.sqrt(torch.clamp(square / count - mu ** 2, min=1e-8)).float()
        return stats


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


def spec_augment(x, t_max, f_max):
    if t_max == 0 and f_max == 0:
        return x
    B, _, rows, T = x.shape
    x = x.clone()
    if t_max > 0:
        w = torch.randint(0, t_max + 1, (B,), device=x.device)
        s = (torch.rand(B, device=x.device) * (T - w).clamp(min=1)).long()
        ar = torch.arange(T, device=x.device)[None, :]
        x = x.masked_fill(((ar >= s[:, None]) & (ar < (s + w)[:, None]))[:, None, None, :], 0.0)
    if f_max > 0:
        w = torch.randint(0, f_max + 1, (B,), device=x.device)
        s = (torch.rand(B, device=x.device) * (rows - w).clamp(min=1)).long()
        ar = torch.arange(rows, device=x.device)[None, :]
        x = x.masked_fill(((ar >= s[:, None]) & (ar < (s + w)[:, None]))[:, None, :, None], 0.0)
    return x


class MonoSpec(nn.Module):
    def __init__(self, stats, p):
        super().__init__()
        self.s = Stats(stats)
        self.use_hi, self.use_ctx = bool(p["use_hi"]), bool(p["use_ctx"])
        self.time_mask, self.freq_mask, self.eq_aug = int(p["time_mask"]), int(p["freq_mask"]), float(p["eq_aug"])
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
            ok = b["hi_mask"]
            if self.training:
                ok = ok & (torch.rand(B, 1, device=ok.device) >= HI_DROP)
            raw = torch.cat([raw, b["hi"] - ref], dim=1)
            mu, sd = torch.cat([mu, s.hishape_mu]), torch.cat([sd, s.hishape_sd])
            full = torch.cat([torch.ones(B, N_LO, dtype=torch.bool, device=raw.device), ok], dim=1)
        if self.training and self.eq_aug > 0:
            u = torch.linspace(-1.0, 1.0, raw.shape[1], device=raw.device)[None, :]
            a = (torch.rand(B, 2, device=raw.device) * 2.0 - 1.0) * self.eq_aug
            raw = raw + (a[:, :1] * u + a[:, 1:] * (1.5 * u ** 2 - 0.5))[:, :, None]
        x = (raw - mu[None, :, None]) / sd[None, :, None]
        if full is not None:
            x = x * full[:, :, None]
        dlev = ((b["level"] - b["lbar"][:, None]) - s.dlev_mu) / s.dlev_sd
        x = torch.stack([x, dlev[:, None, :].expand_as(x)], dim=1)
        if self.training:
            x = spec_augment(x, self.time_mask, self.freq_mask)
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

    def forward_std(self, b):
        return self.base.quantiles(self.z(b), self.raw(b))

    def forward(self, b):
        return self.forward_std(b) * self.s.y_sd + self.s.y_mu


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


def validation_score(q, y, e):
    c = crps_windows(q, y)
    return float(np.mean([c[e == k].mean() for k in np.unique(e)]))


def mean_rmse(q, y, e):
    return float(np.mean([np.sqrt(np.mean((q[e == k, MED] - y[e == k]) ** 2)) for k in np.unique(e)]))


def experiment_scores(q, y, e):
    s = np.maximum(q[:, 0] - y, y - q[:, -1])
    return {int(k): float(np.quantile(s[e == k], 1.0 - BETA)) for k in np.unique(e)}


def conformal_margin(scores):
    v = np.sort(np.asarray(list(scores), dtype=np.float64))
    k = int(np.ceil((len(v) + 1) * (1.0 - ALPHA)))
    return float("inf") if len(v) == 0 or k > len(v) else float(v[k - 1])


def conformal_margins(bank, load):
    q, y, e, fold = [], [], [], []
    for k in range(len(FOLDS)):
        idx = bank.windows(FOLDS[k])
        q.append(predict_quantiles(load(k), bank, idx).mean(0, dtype=np.float64))
        y.append(bank.y[idx])
        e.append(bank.e[idx])
        fold.append(np.full(len(idx), k))
    q, y, e, fold = (np.concatenate(v) for v in (q, y, e, fold))
    scores = experiment_scores(q, y, e)
    fold_of = {k: int(fold[e == k][0]) for k in scores}
    return [conformal_margin([s for k, s in scores.items() if fold_of[k] != f]) for f in range(len(FOLDS))]


def pick_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(name)
        torch.zeros(1, device=device)
    except (RuntimeError, AssertionError) as error:
        fail(f"device {name} is not available: {str(error).splitlines()[0]}")
    return device


def load_optuna():
    try:
        import optuna
    except ImportError:
        fail("optuna is required for the hyperparameter search; install it with 'pip install optuna' or pass --no-search")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    return optuna


def lr_at(step, steps_per_epoch, lr):
    epoch = step / steps_per_epoch
    if epoch < WARMUP:
        return lr * (0.1 + 0.9 * epoch / WARMUP)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, (epoch - WARMUP) / (LR_PERIOD - WARMUP))))
    return LR_MIN + (lr - LR_MIN) * cosine


def pinball_loss(q, y, taus):
    d = y[:, None] - q
    return torch.maximum(taus[None, :] * d, (taus[None, :] - 1.0) * d).mean()


def standardize_y(model, y):
    return (y - model.s.y_mu) / model.s.y_sd


def fit_base_law(model, bank, idx):
    taus = torch.tensor(TAUS, device=bank.device, dtype=torch.float32)
    idx = torch.as_tensor(idx, device=bank.device)
    zs, ys = [], []
    with torch.no_grad():
        for s in range(0, len(idx), BASE_BATCH):
            b = bank.batch(idx[s:s + BASE_BATCH])
            zs.append(model.z(b))
            ys.append(standardize_y(model, b["y"]))
    z, y = torch.cat(zs), torch.cat(ys)
    opt = torch.optim.Adam(model.base.parameters(), lr=BASE_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, BASE_STEPS, eta_min=BASE_LR * 0.01)
    for _ in range(BASE_STEPS):
        opt.zero_grad(set_to_none=True)
        pinball_loss(model.base.quantiles(z), y, taus).backward()
        opt.step()
        sched.step()
    for p in model.base.parameters():
        p.requires_grad_(False)


def train_model(bank, params, idx, stats, seed, epochs, val_idx=None, report=None):
    dev = bank.device
    torch.manual_seed(seed)
    model = MonoSpec(stats, params).to(dev)
    fit_base_law(model, bank, idx)
    with torch.no_grad():
        model.hyper.bias.copy_(model.base.raw().reshape(-1))
    torch.manual_seed(seed)
    decay = [p for n, p in model.named_parameters() if not n.startswith("base.") and n != "hyper.bias"]
    opt = torch.optim.AdamW([dict(params=decay), dict(params=[model.hyper.bias], weight_decay=0.0)], lr=params["lr"],
                            weight_decay=params["weight_decay"])
    taus = torch.tensor(TAUS, device=dev, dtype=torch.float32)
    bs, n = int(params["batch_size"]), len(idx)
    spe = max(1, n // bs)
    idx_t = torch.as_tensor(idx, device=dev)
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    best, bad, step = dict(score=float("inf"), epoch=0, state=None), 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = idx_t[torch.randperm(n, device=dev, generator=gen)]
        total = torch.zeros((), device=dev)
        for i in range(spe):
            b = bank.batch(perm[i * bs:(i + 1) * bs])
            for g in opt.param_groups:
                g["lr"] = lr_at(step, spe, params["lr"])
            opt.zero_grad(set_to_none=True)
            loss = pinball_loss(model.forward_std(b), standardize_y(model, b["y"]), taus)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.detach()
            step += 1
        if not math.isfinite(float(total)):
            raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
        if val_idx is None:
            continue
        model.eval()
        score = validation_score(predict_quantiles([model], bank, val_idx)[0], bank.y[val_idx], bank.e[val_idx])
        if score < best["score"] - 1e-6:
            best.update(score=score, epoch=epoch, state={k: v.clone() for k, v in model.state_dict().items()})
            bad = 0
        else:
            bad += 1
        if report is not None:
            report(epoch, score)
        if bad >= PATIENCE:
            break
    if val_idx is None:
        return model.eval(), epochs, None
    model.load_state_dict(best["state"])
    return model.eval(), best["epoch"], best["score"]


def suggest(trial):
    params = {}
    for name, kind, *args in SEARCH_SPACE:
        if kind == "cat":
            params[name] = trial.suggest_categorical(name, args[0])
        else:
            params[name] = trial.suggest_float(name, args[0], args[1], log=kind == "log")
    return params


def run_search(bank, train_idx, val_idx, fold, args):
    optuna = load_optuna()
    stats = bank.norm_stats(train_idx)
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(PRUNER_STARTUP, PRUNER_WARMUP),
                                sampler=optuna.samplers.TPESampler(seed=TPE_SEED + args.seed + fold, multivariate=True,
                                                                   n_startup_trials=TPE_STARTUP))

    def objective(trial):
        params = suggest(trial)

        def report(epoch, score):
            trial.report(score, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        _, epoch, score = train_model(bank, params, train_idx, stats, args.seed, args.max_epochs, val_idx, report)
        trial.set_user_attr("epochs", epoch)
        trial.set_user_attr("params", params)
        return score

    complete = optuna.trial.TrialState.COMPLETE

    def progress(study, trial):
        done = [t.value for t in study.trials if t.state == complete]
        text = f"value {trial.value:.4f}  epochs {trial.user_attrs['epochs']:2d}" if trial.state == complete \
            else f"pruned at epoch {max(trial.intermediate_values)}"
        print(f"  trial {trial.number + 1:2d}/{args.trials}  {text}  best {min(done, default=float('nan')):.4f}", flush=True)

    start = time.time()
    study.optimize(objective, n_trials=args.trials, callbacks=[progress])
    print(f"  search: {time.time() - start:.0f} s, {(time.time() - start) / args.trials:.1f} s per trial")
    done = sorted((t for t in study.trials if t.state == complete), key=lambda t: t.value)
    if not done:
        fail("every trial of the search was pruned; use more trials")
    settings, seen = [], set()
    for t in done:
        key = json.dumps(t.user_attrs["params"], sort_keys=True)
        if key not in seen:
            seen.add(key)
            settings.append(dict(params=t.user_attrs["params"], epochs=t.user_attrs["epochs"], val_crps=t.value))
    if len(seen) < N_MEMBERS:
        print(f"  note: {len(seen)} distinct settings completed; the remaining members repeat them with their own seeds")
    while len(settings) < N_MEMBERS:
        settings.append(dict(settings[len(settings) % len(seen)]))
    return settings[:N_MEMBERS]


def fold_keys(bank, protocol, k):
    def surface(j):
        return [m["key"] for m in bank.meta if m["surface"] == SURFACES[j]]

    groups = (lambda j: FOLDS[j]) if protocol == "1" else surface
    train = [x for j in range(len(FOLDS)) if j not in (k, (k + 1) % len(FOLDS)) for x in groups(j)]
    return train, groups((k + 1) % len(FOLDS)), groups(k)


def read_config(models):
    path = os.path.join(models, "config.json")
    if os.path.isfile(path):
        with open(path) as fh:
            return json.load(fh)
    return dict(name="MonoSpec", quantiles=list(QUANTILES), alpha=ALPHA, protocol1=dict(folds=[]), protocol2=dict(folds=[]))


def write_config(models, config):
    with open(os.path.join(models, "config.json"), "w") as fh:
        json.dump(config, fh, indent=1)


def stored_settings(config, protocol, k, source):
    for f in config.get(f"protocol{protocol}", {}).get("folds", []):
        if f["fold"] == k:
            return f["members"]
    fail(f"--no-search needs protocol {protocol} fold {k} in {source}")


def train_fold(bank, config, protocol, k, args, source):
    train_keys, val_keys, test_keys = fold_keys(bank, protocol, k)
    tr, va = bank.windows(train_keys, "train"), bank.windows(val_keys, "val")
    trv, te = bank.windows(train_keys + val_keys, "train"), bank.windows(test_keys)
    if not (len(tr) and len(va) and len(te)):
        print(f"protocol {protocol}, fold {k}: training, validation or test windows missing in the data file, skipped")
        return False
    print(f"protocol {protocol}, fold {k}: {len(tr)} training, {len(va)} validation, {len(trv)} final training, {len(te)} test windows", flush=True)
    settings = stored_settings(source[0], protocol, k, source[1]) if args.no_search else run_search(bank, tr, va, k, args)
    stats = bank.norm_stats(trv)
    models, members = [], []
    for m, setting in enumerate(settings):
        start = time.time()
        model, _, _ = train_model(bank, setting["params"], trv, stats, args.seed + m, setting["epochs"])
        print(f"  member {m}: seed {args.seed + m}, E* {setting['epochs']}, {time.time() - start:.1f} s", flush=True)
        models.append(model)
        members.append(dict(seed=args.seed + m, epochs=int(setting["epochs"]), val_crps=float(setting["val_crps"]), params=setting["params"]))
    q = predict_quantiles(models, bank, te).mean(0, dtype=np.float64)
    if protocol == 1:
        print(f"  fold {k}: ensemble test RMSE {mean_rmse(q, bank.y[te], bank.e[te]):.3f} W/cm² "
              f"(mean over {len(np.unique(bank.e[te]))} experiments)", flush=True)
    os.makedirs(args.models, exist_ok=True)
    torch.save({"members": [{n: v.detach().cpu().contiguous() for n, v in m.state_dict().items()} for m in models]},
               os.path.join(args.models, f"protocol{protocol}_fold{k}.pt"))
    section = config.setdefault(f"protocol{protocol}", dict(folds=[]))
    section["folds"] = sorted([f for f in section["folds"] if f["fold"] != k] + [dict(fold=k, members=members)], key=lambda f: f["fold"])
    write_config(args.models, config)
    return True


def update_margins(bank, config, args, device):
    section = config["protocol1"]
    section.pop("c_hat", None)
    have = {f["fold"] for f in section["folds"]}
    if not all(k in have and os.path.isfile(os.path.join(args.models, f"protocol1_fold{k}.pt")) for k in range(len(FOLDS))):
        print("conformal margins not computed: the five Protocol 1 checkpoints are not all in", args.models)
    elif not all(k in bank.index for f in FOLDS for k in f):
        print("conformal margins not computed: the data file lacks some of the 65 experiments")
    else:
        cache = {}
        margins = conformal_margins(bank, lambda k: cache.setdefault(k, load_ensemble(args.models, config, 1, k, device)))
        section["c_hat"] = margins
        print("conformal margins c_hat (W/cm²):", ", ".join(f"{c:.3f}" for c in margins))
    write_config(args.models, config)


def run_train(args):
    start = time.time()
    device = pick_device(args.device)
    if not os.path.isfile(args.data):
        fail(f"data file not found: {args.data} (build it with 'train.py prepare --raw RAW_DIR')")
    if not args.no_search:
        load_optuna()
    config = read_config(args.models)
    own = os.path.join(args.models, "config.json")
    source = (config, own)
    if args.no_search and not os.path.isfile(own):
        released = os.path.join(ROOT, "models", "config.json")
        if not os.path.isfile(released):
            fail(f"--no-search needs the settings in {own} or {released}")
        with open(released) as fh:
            source = (json.load(fh), released)
        print(f"note: {own} does not exist; --no-search uses the settings in {released}")
    bank = WindowBank(read_features(args.data), device)
    print(f"MonoSpec training: {len(bank.keys)} experiments in {args.data}, device {device}, seed {args.seed}", flush=True)
    for protocol in args.protocol:
        trained = [train_fold(bank, config, protocol, k, args, source) for k in args.folds]
        if protocol == "1" and any(trained):
            update_margins(bank, config, args, device)
    print(f"finished in {time.time() - start:.0f} s")


def positive_int(text):
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value < 1:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer of at least 1")
    return value


def build_parser(root):
    parser = argparse.ArgumentParser(prog="train.py", description="MonoSpec: feature-file preparation and training.")
    sub = parser.add_subparsers(dest="command")
    pr = sub.add_parser("prepare", help="build the feature file from the raw Dataverse folder")
    pr.add_argument("--raw", required=True, help="folder with the raw .lvm files of the public dataset")
    pr.add_argument("--out", default=os.path.join(root, "data", "features.npz"), help="feature file to write (default: data/features.npz)")
    pr.add_argument("--workers", type=int, default=4, help="parallel processes (default: 4)")
    pr.add_argument("--only", nargs="+", default=None, help="experiment keys to build (default: all 65)")
    tr = sub.add_parser("train", help="search, train, and conformally calibrate the models (default command)")
    tr.add_argument("--data", default=os.path.join(root, "data", "features.npz"), help="feature file (default: data/features.npz)")
    tr.add_argument("--protocol", nargs="+", choices=["1", "2"], default=["1", "2"], help="protocols to train (default: 1 2)")
    tr.add_argument("--folds", nargs="+", type=int, choices=range(len(FOLDS)), default=list(range(len(FOLDS))), help="folds to train (default: all five)")
    tr.add_argument("--trials", type=positive_int, default=50, help="TPE trials per fold, at least 1 (default: 50)")
    tr.add_argument("--no-search", action="store_true",
                    help="skip the search and retrain with the settings and epochs E* in config.json of --models "
                         "(the released models/config.json if --models has none)")
    tr.add_argument("--max-epochs", type=positive_int, default=40, help="epoch cap of a search trial, at least 1 (default: 40)")
    tr.add_argument("--models", default=os.path.join(root, "models"),
                    help="directory for the checkpoints and config.json (default: models/, which overwrites the released models)")
    tr.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N (default: auto)")
    tr.add_argument("--seed", type=int, default=0, help="base seed of the search and of the members (default: 0)")
    return parser


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] not in ("prepare", "train", "-h", "--help"):
        argv = ["train"] + argv
    args = build_parser(ROOT).parse_args(argv)
    try:
        {"prepare": run_prepare, "train": run_train}[args.command](args)
    except (OSError, ValueError, EOFError) as error:
        fail(str(error))


if __name__ == "__main__":
    main()

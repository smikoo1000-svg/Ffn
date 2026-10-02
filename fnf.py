#!/usr/bin/env python3
"""Song2Piano ULTRA v2 — 완전 단일 파일 (uv 부트스트랩).

실행:  python song2piano_ultra_v2.py
접속:  http://localhost:8000
"""
from __future__ import annotations

# ══════════════════════════════════════════════════════════════════
#  0.  자동 설치 (uv 기반 bootstrap) — 무거운 import 전에 실행
# ══════════════════════════════════════════════════════════════════
import os, sys, subprocess, shutil, venv as _venv
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_VENV_DIR = _HERE / ".venv311"
_MARKER = _VENV_DIR / ".s2p_ready"

PKGS = [
    "numpy>=1.26,<3", "scipy", "soundfile", "librosa",
    "pretty_midi", "mido", "music21",
    "demucs", "basic-pitch", "torchcrepe", "beat_this",
    "flask", "verovio", "cairosvg", "pypdf",
]


def _venv_py():
    for p in (_VENV_DIR / "bin" / "python", _VENV_DIR / "Scripts" / "python.exe"):
        if p.exists():
            return p
    return None


def _in_venv():
    vpy = _venv_py()
    if vpy is None:
        return False
    try:
        return Path(sys.executable).resolve() == vpy.resolve()
    except Exception:
        return False


def _venv_py_version():
    vpy = _venv_py()
    if not vpy:
        return None
    try:
        r = subprocess.run(
            [str(vpy), "-c", "import sys;print(f'{sys.version_info[0]}.{sys.version_info[1]}')"],
            capture_output=True, text=True, timeout=15)
        return tuple(int(x) for x in r.stdout.strip().split("."))
    except Exception:
        return None


def _have_uv():
    if shutil.which("uv"):
        return shutil.which("uv")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=False)
    except Exception:
        pass
    return shutil.which("uv")


def _bootstrap():
    print("=" * 60, flush=True)
    print("  🎹 Song2Piano ULTRA v2 — 초기 설치 (uv)", flush=True)
    print("=" * 60, flush=True)

    uv = _have_uv()
    if not uv:
        print("  ✗ uv 설치 실패. 수동으로 'pip install uv' 실행 후 재시도.", flush=True)
        sys.exit(1)
    print(f"  · uv: {uv}", flush=True)

    # 기존 venv가 부적합(3.13+, <3.10)이면 재생성
    if _VENV_DIR.exists():
        v = _venv_py_version()
        if v is None or not ((3, 10) <= v <= (3, 12)):
            print(f"  ! 기존 venv {v} 부적합 → 삭제 후 재생성", flush=True)
            shutil.rmtree(_VENV_DIR, ignore_errors=True)

    if not _venv_py():
        print("  · Python 3.11 설치 시도 (uv)", flush=True)
        try:
            subprocess.run([uv, "python", "install", "3.11"], check=True, timeout=300)
        except Exception as e:
            print(f"  ! uv python install 실패: {e}", flush=True)
        try:
            r = subprocess.run([uv, "python", "find", "3.11"],
                               capture_output=True, text=True, check=True, timeout=30)
            py311 = r.stdout.strip().splitlines()[0]
            print(f"  · Python 3.11: {py311}", flush=True)
            subprocess.run([uv, "venv", "--python", py311, str(_VENV_DIR)],
                           check=True, timeout=180)
        except Exception as e:
            print(f"  ! uv venv 실패: {e}", flush=True)
            print("  → 현재 파이썬으로 venv 생성", flush=True)
            _venv.create(_VENV_DIR, with_pip=True)

    vpy = _venv_py()
    if not vpy:
        print("  ✗ 가상환경 생성 실패", flush=True)
        sys.exit(1)

    v = _venv_py_version()
    print(f"  · 가상환경 Python {v[0]}.{v[1]} 준비됨", flush=True)

    # uv pip으로 설치 (훨씬 빠름)
    print("  · numpy/scipy 설치", flush=True)
    subprocess.run([uv, "pip", "install", "--python", str(vpy),
                    "numpy>=1.26,<3", "scipy"], check=False, timeout=600)

    print("  · PyTorch CPU 설치 (수백 MB)", flush=True)
    subprocess.run([uv, "pip", "install", "--python", str(vpy),
                    "torch", "torchaudio",
                    "--index-url", "https://download.pytorch.org/whl/cpu"],
                   check=False, timeout=3600)

    for p in PKGS:
        print(f"  · {p}", flush=True)
        subprocess.run([uv, "pip", "install", "--python", str(vpy), p],
                       check=False, timeout=1800)

    try:
        _MARKER.write_text("ok")
    except Exception:
        pass

    print("\n  ✅ 설치 완료 — 앱 시작\n", flush=True)
    os.execv(str(vpy), [str(vpy), str(Path(__file__).resolve())] + sys.argv[1:])


if not _in_venv():
    _bootstrap()

# ══════════════════════════════════════════════════════════════════
#  1.  앱 본체
# ══════════════════════════════════════════════════════════════════
import argparse, hashlib, io, json, re, tempfile, threading, time, uuid, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Optional
import numpy as np

warnings.filterwarnings("ignore")
SR, HOP = 22050, 512

# ──────────────────────────────────────────────────────────────────
#  1-0.  지연 로드
# ──────────────────────────────────────────────────────────────────
_LIBS = {}


def _lib(name):
    if name in _LIBS:
        return _LIBS[name]
    try:
        if name == "librosa":
            import librosa; _LIBS[name] = librosa
        elif name == "pretty_midi":
            import pretty_midi; _LIBS[name] = pretty_midi
        elif name == "sf":
            import soundfile as sf; _LIBS[name] = sf
        elif name == "mido":
            import mido; _LIBS[name] = mido
        elif name == "torch":
            import torch; _LIBS[name] = torch
        elif name == "torchcrepe":
            import torchcrepe; _LIBS[name] = torchcrepe
        elif name == "bp":
            from basic_pitch.inference import Model, predict
            from basic_pitch import ICASSP_2022_MODEL_PATH
            _LIBS[name] = (Model, predict, ICASSP_2022_MODEL_PATH)
        else:
            _LIBS[name] = __import__(name)
    except Exception as e:
        _LIBS[name] = None
        print(f"[경고] {name} 없음: {e}", file=sys.stderr)
    return _LIBS[name]


# ──────────────────────────────────────────────────────────────────
#  1-1.  음악 이론
# ──────────────────────────────────────────────────────────────────
음이름 = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_장조프로파일 = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_단조프로파일 = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

코드종류 = {
    "": (0, 4, 7), "m": (0, 3, 7), "dim": (0, 3, 6), "aug": (0, 4, 8),
    "sus2": (0, 2, 7), "sus4": (0, 5, 7), "5": (0, 7),
    "6": (0, 4, 7, 9), "m6": (0, 3, 7, 9), "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11),
    "m7": (0, 3, 7, 10), "m7b5": (0, 3, 6, 10), "dim7": (0, 3, 6, 9),
    "add9": (0, 4, 7, 14), "madd9": (0, 3, 7, 14), "9": (0, 4, 7, 10, 14),
    "maj9": (0, 4, 7, 11, 14), "m9": (0, 3, 7, 10, 14), "7sus4": (0, 5, 7, 10),
    "13": (0, 4, 7, 10, 14, 21), "m11": (0, 3, 7, 10, 14, 17),
}
_장조가중 = {0: 1.0, 2: 0.7, 4: 0.55, 5: 0.9, 7: 1.0, 9: 0.8, 11: 0.35,
             3: 0.25, 8: 0.25, 10: 0.2, 1: 0.15, 6: 0.1}
_단조가중 = {9: 1.0, 0: 0.95, 3: 0.8, 5: 0.8, 7: 0.65, 8: 0.6, 10: 0.6,
             2: 0.4, 4: 0.3, 6: 0.2, 1: 0.15, 11: 0.1}
_장조샵 = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 1: -5, 8: -4, 3: -3, 10: -2, 5: -1}
_단조샵 = {9: 0, 4: 1, 11: 2, 6: 3, 1: 4, 8: 5, 3: 6, 10: -5, 5: -4, 0: -3, 7: -2, 2: -1}


def 조성추정(크로마평균):
    cm = np.asarray(크로마평균, float)
    if cm.sum() < 1e-9:
        return 0, "maj", 0.0
    cm = cm / cm.sum()
    점수 = []
    for r in range(12):
        for 모드, prof in (("maj", _장조프로파일), ("min", _단조프로파일)):
            p = np.roll(prof, r); p = p / p.sum()
            점수.append((float(np.dot(cm, p)), r, 모드))
    점수.sort(reverse=True)
    s1, r1, m1 = 점수[0]
    conf = float((s1 - 점수[1][0]) / (s1 + 1e-9))
    return r1, m1, min(1.0, conf * 3)


_템플릿 = None


def 템플릿():
    global _템플릿
    if _템플릿 is None:
        out = []
        for r in range(12):
            for kind, ivs in 코드종류.items():
                pcs = set((r + i) % 12 for i in ivs)
                v = np.zeros(12)
                for i in ivs:
                    w = (1.0 if i == 0 else 0.9 if i in (3, 4) else
                         0.85 if i in (6, 7, 8) else 0.75 if i in (10, 11) else 0.5)
                    v[(r + i) % 12] = w
                out.append((r, kind, pcs, v / (np.linalg.norm(v) + 1e-9)))
        _템플릿 = out
    return _템플릿


def 코드prior(조근음, 조모드, 코드근음, 코드종):
    deg = (코드근음 - 조근음) % 12
    w = _장조가중 if 조모드 == "maj" else _단조가중
    base = w.get(deg, 0.05)
    if 조모드 == "maj":
        kb = {"": 0.3, "m": -0.15, "7": 0.15, "maj7": 0.2, "m7": 0.0,
              "sus4": 0.0, "sus2": -0.05, "add9": 0.1}
    else:
        kb = {"m": 0.3, "": -0.25, "m7": 0.15, "7": 0.1, "dim": 0.1, "m7b5": 0.1}
    return base * (1.0 + kb.get(코드종, 0.0))


def 코드이름(root, kind):
    return f"{음이름[root % 12]}{kind}"


def 박자표추정(beats, downs):
    if len(downs) < 3 or len(beats) < 4:
        return 4
    idx = np.searchsorted(beats, downs)
    gaps = np.diff(idx)
    gaps = gaps[gaps > 0]
    if len(gaps) == 0:
        return 4
    m = int(round(np.median(gaps)))
    return min((2, 3, 4, 6), key=lambda x: abs(x - m))


# ──────────────────────────────────────────────────────────────────
#  1-2.  오디오 I/O + 캐시 (LRU)
# ──────────────────────────────────────────────────────────────────
_캐시 = {}
_캐시최대 = 60


def _캐시저장(k, v):
    if len(_캐시) > _캐시최대:
        for kk in list(_캐시.keys())[:len(_캐시) // 2]:
            del _캐시[kk]
    _캐시[k] = v


def 오디오로드(wav, sr=SR, mono=True):
    librosa = _lib("librosa")
    k = ("audio", str(wav), sr, mono)
    if k not in _캐시:
        _캐시저장(k, librosa.load(str(wav), sr=sr, mono=mono)[0])
    return _캐시[k]


def cqt(wav):
    librosa = _lib("librosa")
    k = ("cqt", str(wav))
    if k not in _캐시:
        C = np.abs(librosa.cqt(오디오로드(wav), sr=SR, hop_length=HOP,
                                fmin=librosa.midi_to_hz(24), n_bins=84, bins_per_octave=12))
        _캐시저장(k, C / (np.percentile(C, 95) + 1e-9))
    return _캐시[k]


def 프레임(t):
    return int(t * SR / HOP)


def 온셋환경(wav):
    librosa = _lib("librosa")
    k = ("env", str(wav))
    if k not in _캐시:
        y = 오디오로드(wav)
        env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
        rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
        db = librosa.amplitude_to_db(rms + 1e-6, ref=1.0)
        ons = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=HOP,
                                          units="time", delta=0.08)
        _캐시저장(k, (env, db, ons))
    return _캐시[k]


# ──────────────────────────────────────────────────────────────────
#  1-3.  음원 분리 (Demucs 앙상블)
# ──────────────────────────────────────────────────────────────────
STEM6 = ("vocals", "bass", "other", "drums", "guitar", "piano")
STEM4 = ("vocals", "bass", "other", "drums")


def wav변환(audio: Path, work: Path, max_sec=None) -> Path:
    sf = _lib("sf")
    out = work / f"{audio.stem}.wav"
    try:
        data, sr = sf.read(str(audio), always_2d=True)
    except Exception:
        librosa = _lib("librosa")
        y, sr = librosa.load(str(audio), sr=None, mono=False)
        data = (y if y.ndim > 1 else y[None]).T
    if max_sec:
        data = data[:int(max_sec * sr)]
    sf.write(str(out), data, sr, subtype="PCM_16")
    return out


def _demucs_한모델(mix: Path, model: str, tmp: Path):
    subprocess.run([sys.executable, "-m", "demucs", "-n", model,
                    "--overlap", "0.25", "-o", tmp, str(mix)],
                   check=True, capture_output=True, timeout=3600)
    out = Path(tmp) / model / "mix"
    names = STEM6 if "6s" in model else STEM4
    if not all((out / f"{n}.wav").exists() for n in names):
        return None
    return {n: out / f"{n}.wav" for n in names}


def _enstemble_평균(stem_dicts: list, out_dir: Path, names: tuple):
    sf = _lib("sf")
    result = {}
    for n in names:
        arrays, srs = [], []
        for d in stem_dicts:
            if d is None or n not in d:
                continue
            data, sr = sf.read(str(d[n]), always_2d=True)
            arrays.append(data)
            srs.append(sr)
        if not arrays:
            continue
        min_len = min(a.shape[0] for a in arrays)
        stacked = np.stack([a[:min_len] for a in arrays])
        avg = stacked.mean(axis=0)
        out = out_dir / f"{n}.wav"
        sf.write(str(out), avg, srs[0], subtype="PCM_16")
        result[n] = out
    return result


def 음원분리(audio: Path, cache_dir: Path, ensemble=True, max_sec=None, force=False):
    cache_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1()
    with open(audio, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    h.update(f"ensemble={ensemble}|max={max_sec}".encode())
    d = cache_dir / h.hexdigest()[:16]
    d.mkdir(exist_ok=True)

    mix = d / "mix.wav"
    if not mix.exists() or force:
        wav변환(audio, d, max_sec).replace(mix)

    expect6 = {n: d / f"{n}.wav" for n in STEM6}
    if all(p.exists() for p in expect6.values()):
        print("[분리] 캐시 재사용")
        return {**expect6, "mix": mix}

    models = ["htdemucs_ft", "htdemucs_6s", "hdemucs_mmi"] if ensemble else ["htdemucs_6s"]
    stem_dicts = []

    if ensemble:
        print(f"[분리] 앙상블 시작: {models}")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            with ThreadPoolExecutor(3) as ex:
                futs = {ex.submit(_demucs_한모델, mix, m, tmp_p): m for m in models}
                for f in as_completed(futs):
                    m = futs[f]
                    try:
                        r = f.result()
                        stem_dicts.append(r)
                        print(f"[분리] {m} 완료")
                    except Exception as e:
                        print(f"[분리] {m} 실패: {e}", file=sys.stderr)
            if stem_dicts:
                print("[분리] 앙상블 평균 계산 중")
                common = ("vocals", "bass", "other", "drums")
                result = _enstemble_평균(stem_dicts, d, common)
                for n in ("guitar", "piano"):
                    for sd in stem_dicts:
                        if sd and n in sd:
                            shutil.copy(str(sd[n]), str(d / f"{n}.wav"))
                            result[n] = d / f"{n}.wav"
                            break
                if len(result) >= 4:
                    print(f"[분리] 앙상블 완료: {sorted(result.keys())}")
                    dirs = sorted((p for p in cache_dir.iterdir() if p.is_dir()),
                                  key=lambda p: p.stat().st_mtime)
                    for p in dirs[:-2]:
                        shutil.rmtree(p, ignore_errors=True)
                    return {**result, "mix": mix}

    for m in ([models[-1]] if ensemble else models) + ["htdemucs"]:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                r = _demucs_한모델(mix, m, Path(tmp))
                if r is None:
                    continue
                for n, p in r.items():
                    shutil.move(str(p), str(d / f"{n}.wav"))
                names = STEM6 if "6s" in m else STEM4
                print(f"[분리] {m} 폴백 완료")
                dirs = sorted((p for p in cache_dir.iterdir() if p.is_dir()),
                              key=lambda p: p.stat().st_mtime)
                for p in dirs[:-2]:
                    shutil.rmtree(p, ignore_errors=True)
                return {**{n: d / f"{n}.wav" for n in names}, "mix": mix}
            except Exception as e:
                print(f"[분리] {m} 실패: {e}", file=sys.stderr)

    raise RuntimeError("음원 분리 실패")


# ──────────────────────────────────────────────────────────────────
#  1-4.  박자 추적 (Beat This! + DBN 후처리)
# ──────────────────────────────────────────────────────────────────
def _dbn_후처리(beats, downs, beats_per_bar=4):
    if len(beats) < 8:
        return beats, downs
    beats = np.asarray(beats, float)
    ivs = np.diff(beats)
    med = float(np.median(ivs))
    if med <= 0:
        return beats, downs
    keep = [beats[0]]
    for b in beats[1:]:
        gap = b - keep[-1]
        if gap < 0.4 * med:
            continue
        elif gap > 2.5 * med:
            n = int(round(gap / med))
            for k in range(1, n):
                keep.append(keep[-1] + (b - keep[-1]) / (n - k))
        keep.append(b)
    beats = np.asarray(sorted(keep), float)
    if len(downs) >= 3 and beats_per_bar >= 2:
        new_downs = []
        for d in downs:
            i = int(np.argmin(np.abs(beats - d)))
            new_downs.append(beats[i])
        downs = np.asarray(sorted(set(new_downs)), float)
    return beats, downs


class beats:
    """Convenient wrapper around the project's beat-tracking pipeline."""

    def __init__(self, mix=None, bpm_override=None, use_dbn=True):
        self.mix = Path(mix) if mix is not None else None
        self.bpm_override = bpm_override
        self.use_dbn = use_dbn
        self.bpm = 120.0
        self.beats = np.asarray([], float)
        self.downs = np.asarray([], float)
        self.meter = 4
        if self.mix is not None:
            self.track(self.mix)

    def track(self, mix=None):
        if mix is not None:
            self.mix = Path(mix)
        if self.mix is None:
            raise ValueError("mix path is required")
        self.bpm, self.beats, self.downs, self.meter = 박자추적(
            self.mix, bpm_override=self.bpm_override, use_dbn=self.use_dbn
        )
        return self.bpm, self.beats, self.downs, self.meter

    def load(self, path):
        p = Path(path)
        z = np.load(p, allow_pickle=True)
        self.bpm = float(z["bpm"])
        self.beats = np.asarray(z["beats"], float)
        self.downs = np.asarray(z["downs"], float)
        self.meter = int(z["meter"])
        self.mix = p.with_name("mix.wav") if p.name == "beats.npz" else p
        return self.bpm, self.beats, self.downs, self.meter

    def save(self, path=None):
        if self.mix is not None and path is None:
            target = self.mix.with_name("beats.npz")
        else:
            target = Path(path) if path is not None else Path("beats.npz")
        np.savez(target, bpm=float(self.bpm), beats=np.asarray(self.beats, float),
                 downs=np.asarray(self.downs, float), meter=int(self.meter))
        return target

    def estimate(self, mix=None):
        if mix is not None:
            return self.track(mix)
        if self.mix is None:
            raise ValueError("mix path is required")
        return self.track(self.mix)

    def __len__(self):
        return len(self.beats)

    def __iter__(self):
        return iter((self.bpm, self.beats, self.downs, self.meter))

    def __repr__(self):
        return f"beats(mix={self.mix}, bpm={self.bpm}, meter={self.meter}, beats={len(self.beats)})"


def 박자추적(mix: Path, bpm_override=None, use_dbn=True):
    cache = mix.with_name("beats.npz")
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return float(z["bpm"]), z["beats"], z["downs"], int(z["meter"])

    beats = downs = None
    try:
        from beat_this.inference import File2Beats
        beats, downs = File2Beats(device="cpu", dbn=False)(str(mix))
        beats = np.asarray(beats, float); downs = np.asarray(downs, float)
        print("[박자] Beat This! OK")
    except Exception as e:
        print(f"[박자] Beat This! 불가 ({e}), librosa 사용", file=sys.stderr)

    if beats is None or len(beats) < 8:
        librosa = _lib("librosa")
        _, beats = librosa.beat.beat_track(y=오디오로드(mix), sr=SR,
                                            units="time", tightness=100)
        beats = np.asarray(beats, float); downs = np.array([], float)

    if len(beats) >= 8:
        iv = np.diff(beats); ref = float(np.median(iv))
        filled = [beats[0]]
        for i, v in enumerate(iv):
            n = int(round(v / ref))
            if v > 1.35 * ref and n >= 2:
                filled += [beats[i] + v * k / n for k in range(1, n)]
            filled.append(beats[i + 1])
        beats = np.asarray(filled, float)

    if len(beats) > 1:
        bpm = 60 * (len(beats) - 1) / float(beats[-1] - beats[0])
        while bpm < 60 and len(beats) > 1:
            beats = np.sort(np.concatenate([beats, (beats[:-1] + beats[1:]) / 2])); bpm *= 2
        while bpm > 220 and len(beats) > 1:
            beats = beats[::2]; bpm /= 2
    else:
        bpm = 120.0

    if bpm_override:
        bpm = float(bpm_override)
        beats = beats[0] + np.arange(0, beats[-1] - beats[0] + 60 / bpm, 60 / bpm)

    if use_dbn:
        meter_guess = 박자표추정(beats, downs) if len(downs) >= 3 else 4
        beats, downs = _dbn_후처리(beats, downs, beats_per_bar=meter_guess)

    meter = 박자표추정(beats, downs) if len(downs) >= 3 else 4
    bpm = round(float(bpm), 2)
    try:
        np.savez(cache, bpm=bpm, beats=beats, downs=downs, meter=meter)
    except Exception:
        pass
    return bpm, beats, downs, meter


# ──────────────────────────────────────────────────────────────────
#  1-5.  음 전사 (BP + CREPE + pyin)
# ──
_BP모델 = None


def basic_pitch전사(wav: Path, **kw):
    bp = _lib("bp")
    if bp is None:
        return []
    global _BP모델
    Model, predict, ckpt = bp
    if _BP모델 is None:
        _BP모델 = Model(ckpt)
    d = dict(minimum_note_length=80, onset_threshold=0.35, frame_threshold=0.2,
             melodia_trick=True, minimum_frequency=60, maximum_frequency=1500)
    d.update(kw)
    try:
        _, midi, _ = predict(str(wav), _BP모델, **d)
        return [n for i in midi.instruments for n in i.notes]
    except Exception as e:
        print(f"[BP] 실패: {e}", file=sys.stderr)
        return []


def crepe전사(wav: Path, fmin=65, fmax=1600, conf=0.5):
    tc = _lib("torchcrepe"); pm = _lib("pretty_midi"); torch = _lib("torch")
    if tc is None or torch is None:
        return []
    try:
        y = 오디오로드(wav)
        t = torch.from_numpy(y).float().unsqueeze(0)
        with torch.no_grad():
            pitch, period = tc.predict(t, sr=SR, hop_length=HOP, fmin=fmin, fmax=fmax,
                                        model="full", return_periodicity=True)
        pitch = pitch.squeeze(0).cpu().numpy()
        period = period.squeeze(0).cpu().numpy()
    except Exception as e:
        print(f"[CREPE] 실패: {e}", file=sys.stderr)
        return []
    notes, s, p = [], None, None
    for i, (pv, cv) in enumerate(zip(pitch, period)):
        t = i * HOP / SR
        if cv >= conf and not np.isnan(pv) and pv > 0:
            mp = int(round(69 + 12 * np.log2(pv / 440.0)))
            if p is None or abs(mp - p) > 1:
                if s is not None and p is not None and t - s >= 0.06:
                    notes.append(pm.Note(velocity=80, pitch=p, start=s, end=t))
                s, p = t, mp
            else:
                p = mp
        else:
            if s is not None and p is not None and t - s >= 0.06:
                notes.append(pm.Note(velocity=80, pitch=p, start=s, end=t))
            s = p = None
    return notes


def pyin전사(wav: Path, fmin=60, fmax=1500):
    librosa = _lib("librosa"); pm = _lib("pretty_midi")
    try:
        y = 오디오로드(wav)
        f0, voiced, vprob = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=SR, hop_length=HOP)
    except Exception as e:
        print(f"[pyin] 실패: {e}", file=sys.stderr)
        return []
    notes, s, p = [], None, None
    for i, (f, v, prob) in enumerate(zip(f0, voiced, vprob)):
        t = i * HOP / SR
        if v and f and f > 0 and prob > 0.3:
            mp = int(round(librosa.hz_to_midi(f)))
            if p is None or abs(mp - p) > 1:
                if s is not None and p is not None and t - s >= 0.06:
                    notes.append(pm.Note(velocity=80, pitch=p, start=s, end=t))
                s, p = t, mp
            else:
                p = mp
        else:
            if s is not None and p is not None and t - s >= 0.06:
                notes.append(pm.Note(velocity=80, pitch=p, start=s, end=t))
            s = p = None
    return notes


def 전사앙상블(notes_a, notes_b, notes_c, tol=0.06):
    if not notes_a:
        notes_a = []
    if not notes_b:
        notes_b = []
    if not notes_c:
        notes_c = []
    base = list(notes_a)
    if not base:
        base = list(notes_b or notes_c)
    starts = np.array([n.start for n in base], dtype=float)
    for source in (notes_b, notes_c):
        for note in source:
            if len(starts) == 0 or np.min(np.abs(starts - note.start)) > tol:
                base.append(note)
                starts = np.append(starts, note.start)
    return sorted(base, key=lambda n: n.start)


def 음역추정(wav: Path, lo_pct=2, hi_pct=98):
    librosa = _lib("librosa")
    try:
        y = 오디오로드(wav)
        f0, v, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C1"),
                                 fmax=librosa.note_to_hz("C8"), sr=SR, hop_length=HOP)
        vv = f0[v & ~np.isnan(f0)]
        if len(vv) > 20:
            p = librosa.hz_to_midi(vv)
            return int(np.percentile(p, lo_pct)) - 2, int(np.percentile(p, hi_pct)) + 2
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────────────────────────
#  1-6.  후처리
# ──────────────────────────────────────────────────────────────────
_배음간격 = (12, 19, 24, 28, 31, 34)


def 배음제거(notes, ratio=0.55, min_ov=0.6, spans=None, beats=None):
    if not notes:
        return notes
    spans_pcs = []
    if spans and beats is not None:
        for s, e, r, k in spans:
            ivs = 코드종류.get(k, (0, 4, 7))
            spans_pcs.append((s, e, {(r + i) % 12 for i in ivs}))

    def in_chord(t, pitch):
        if not spans_pcs or beats is None:
            return False
        bt = np.asarray(beats, float)
        if not (bt[0] <= t <= bt[-1]):
            return False
        idx = float(np.interp(t, bt, np.arange(len(bt))))
        pc = pitch % 12
        for s, e, pcs in spans_pcs:
            if s <= idx < e:
                return pc in pcs
        return False

    ns = sorted(notes, key=lambda n: n.start)
    keep = []
    for i, n in enumerate(ns):
        if in_chord(n.start, n.pitch):
            keep.append(n); continue
        dur = max(n.end - n.start, 1e-3)
        exp = False
        for m in ns[max(0, i - 60):i + 60]:
            if m is n or (n.pitch - m.pitch) not in _배음간격:
                continue
            ov = min(n.end, m.end) - max(n.start, m.start)
            if (ov >= min_ov * dur and m.velocity >= n.velocity / ratio
                    and abs(m.start - n.start) <= 0.1):
                exp = True; break
        if not exp:
            keep.append(n)
    return keep


def refine_offsets(notes, wav, decay=0.22, max_ext=0.3, min_len=0.05):
    if not notes:
        return notes
    pm = _lib("pretty_midi")
    C = cqt(wav)
    nfr = C.shape[1]
    ext = int(max_ext * SR / HOP)
    nxt = {}; last = {}
    for i in sorted(range(len(notes)), key=lambda i: notes[i].start, reverse=True):
        p = notes[i].pitch
        nxt[i] = last.get(p); last[p] = notes[i].start
    out = []
    for i, n in enumerate(notes):
        idx = n.pitch - 24
        if idx < 1 or idx >= 83:
            out.append(n); continue
        e = C[idx - 1:idx + 2].max(axis=0)
        a = min(프레임(n.start), nfr - 2); b = min(프레임(n.end), nfr - 1)
        peak = float(e[a:a + max(3, int(0.12 * SR / HOP))].max())
        if peak < 1e-3:
            out.append(n); continue
        hi = min(nfr - 2, b + ext)
        t_end = None
        for t in range(a + 3, hi):
            if e[t] < decay * peak and e[t + 1] < decay * peak:
                t_end = t; break
        new_end = (t_end * HOP / SR) if t_end else ((hi * HOP / SR)
                    if e[min(b, nfr - 1)] > 0.6 * peak else n.end)
        new_end = max(n.start + min_len, new_end)
        if nxt.get(i) is not None:
            new_end = min(new_end, max(n.start + min_len, nxt[i] - 0.01))
        out.append(pm.Note(velocity=n.velocity, pitch=n.pitch,
                            start=n.start, end=float(new_end)))
    return out


def 벨로시티(notes, wav, base=80, spread=19, lo=28, hi=127, phrase_win=2.0):
    if len(notes) < 8:
        return notes
    pm = _lib("pretty_midi")
    env, db, _ = 온셋환경(wav)
    w = max(2, int(0.12 * SR / HOP))
    pw = max(w, int(phrase_win * SR / HOP))
    att, loud, phr = [], [], []
    for n in notes:
        a = min(프레임(n.start), len(env) - 1)
        att.append(float(env[max(0, a - 1):a + 4].max()))
        loud.append(float(db[a:a + w].mean()))
        phr.append(float(db[max(0, a - pw // 2):a + pw // 2].mean()))
    att = np.log1p(np.array(att)); loud = np.array(loud); phr = np.array(phr)

    def z(x):
        m = np.median(x); mad = np.median(np.abs(x - m)) + 1e-6
        return (x - m) / (1.4826 * mad)

    zz = np.clip(0.45 * z(att) + 0.40 * z(loud) + 0.15 * z(phr), -3.0, 3.0)
    return [pm.Note(velocity=int(np.clip(round(base + spread * v), lo, hi)),
                    pitch=n.pitch, start=n.start, end=n.end)
            for n, v in zip(notes, zz)]


# ──────────────────────────────────────────────────────────────────
#  1-7.  화음 인식
# ──────────────────────────────────────────────────────────────────
def 화음인식(stems, beats, key=None):
    librosa = _lib("librosa")
    size = int(beats[-1] * SR) + SR
    y = None
    weights = {"vocals": 0.4, "other": 1.0, "bass": 0.7, "guitar": 0.6, "piano": 1.0}
    for k, w in weights.items():
        if k not in stems or not Path(stems[k]).exists():
            continue
        yi = librosa.util.fix_length(
            librosa.load(str(stems[k]), sr=SR, mono=True)[0], size=size)
        y = (yi * w) if y is None else (y + yi * w)
    if y is None:
        return [], (0, "maj", 0.0)
    chroma = np.log1p(librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP) * 10)
    fr = librosa.time_to_frames(beats, sr=SR, hop_length=HOP)
    n = len(beats) - 1
    if n < 1:
        return [], (0, "maj", 0.0)
    C = np.zeros((n, 12))
    for i in range(n):
        seg = chroma[:, fr[i]:max(fr[i] + 1, fr[i + 1])]
        C[i] = np.median(seg, axis=1) if seg.size else 0
    norms = np.linalg.norm(C, axis=1, keepdims=True)
    C = C / np.maximum(norms, 1e-9)
    if key is None:
        root, mode, conf = 조성추정(C.mean(axis=0))
    else:
        root, mode = key; conf = 1.0
    T = 템플릿()
    V = np.stack([t[3] for t in T])
    prob = np.exp((C @ V.T) * 15)
    prob /= prob.sum(axis=1, keepdims=True)
    prior = np.array([코드prior(root, mode, t[0], t[1]) for t in T])
    prior /= prior.sum()
    prob = prob * prior[None, :]
    prob /= prob.sum(axis=1, keepdims=True)
    stay = 0.85; Tm = len(T)
    trans = np.full((Tm, Tm), (1 - stay) / (Tm - 1))
    np.fill_diagonal(trans, stay)
    for i, (r1, k1, _, _) in enumerate(T):
        for j, (r2, k2, _, _) in enumerate(T):
            if r1 == r2 and k1 != k2:
                trans[i, j] *= 3.0
    trans /= trans.sum(axis=1, keepdims=True)
    path = librosa.sequence.viterbi(prob.T, trans, p_init=np.full(Tm, 1 / Tm))
    spans, st = [], 0
    for i in range(1, n + 1):
        if i == n or path[i] != path[st]:
            r, k, pcs, _ = T[path[st]]
            if norms[st:i].mean() > 1e-3:
                spans.append((st, i, r, k))
            st = i
    return spans, (root, mode, conf)


# ──────────────────────────────────────────────────────────────────
#  1-8.  편곡 유틸
# ──────────────────────────────────────────────────────────────────
def 단선율(notes, min_len, step=0.01):
    pm = _lib("pretty_midi")
    notes = [n for n in notes if n.end > n.start]
    if not notes:
        return []
    T = int(max(n.end for n in notes) / step) + 2
    owner = np.full(T, -1, int)
    for i in sorted(range(len(notes)),
                    key=lambda i: (notes[i].velocity, notes[i].end - notes[i].start)):
        owner[int(notes[i].start / step):int(notes[i].end / step) + 1] = i
    out, t = [], 0
    while t < T:
        if owner[t] < 0:
            t += 1; continue
        i, u = owner[t], t
        while u < T and owner[u] == i:
            u += 1
        src = notes[i]
        out.append(pm.Note(velocity=src.velocity, pitch=src.pitch,
                            start=t * step, end=u * step))
        t = u
    merged = []
    for n in out:
        if merged and merged[-1].pitch == n.pitch and n.start - merged[-1].end <= 2 * step:
            merged[-1].end = n.end
        else:
            merged.append(n)
    return [n for n in merged if n.end - n.start >= min_len]


def 동음병합(notes, gap, onsets=None):
    on = np.asarray(onsets if onsets is not None else [], float)
    out = []
    for n in sorted(notes, key=lambda n: (n.pitch, n.start)):
        re_at = (bool(len(on)) and out and out[-1].pitch == n.pitch
                 and n.start - out[-1].start > 0.08
                 and np.min(np.abs(on - n.start)) <= 0.05)
        if out and out[-1].pitch == n.pitch and n.start - out[-1].end <= gap and not re_at:
            out[-1].end = max(out[-1].end, n.end)
            out[-1].velocity = max(out[-1].velocity, n.velocity)
        else:
            out.append(n)
    return sorted(out, key=lambda n: n.start)


def 폴리제한(notes, max_poly, min_len, min_vel):
    notes = sorted((n for n in notes
                    if n.end - n.start >= min_len and n.velocity >= min_vel),
                   key=lambda n: n.start)
    keep = []
    for n in notes:
        live = [k for k in keep if k.end > n.start]
        if len(live) >= max_poly:
            w = min(live + [n], key=lambda k: k.velocity)
            if w is n:
                continue
            keep.remove(w)
        keep.append(n)
    return keep


def 옥타브보정(notes):
    ns = sorted(notes, key=lambda n: n.start)
    for i in range(1, len(ns) - 1):
        p, c, n = ns[i - 1].pitch, ns[i].pitch, ns[i + 1].pitch
        if abs(c - p) >= 8 and abs(c - n) >= 8 and abs(p - n) <= 5:
            target = (p + n) / 2
            cand = min((c - 12, c + 12), key=lambda x: abs(x - target))
            if abs(cand - target) <= 6:
                ns[i].pitch = cand
    return ns


def 온셋정밀화(notes, wav, win=0.07):
    librosa = _lib("librosa"); pm = _lib("pretty_midi")
    ref = librosa.onset.onset_detect(y=오디오로드(wav), sr=SR, units="time")
    if len(ref) == 0:
        return notes
    out = []
    for n in notes:
        t = float(ref[np.argmin(np.abs(ref - n.start))])
        s0 = t if abs(t - n.start) <= win and t < n.end - 0.02 else n.start
        out.append(pm.Note(velocity=n.velocity, pitch=n.pitch, start=s0, end=n.end))
    return sorted(out, key=lambda n: n.start)


def 격자정렬(notes, bpm, beats, k0, divs=(4,), shift=None, phase=0.0):
    pm = _lib("pretty_midi")
    beats = np.asarray(beats, float)
    if len(beats) < 4:
        beats = np.arange(0, 600, 60 / bpm)
    idx = np.arange(len(beats)) - k0
    spb = 60 / bpm

    def pos(t):
        if t <= beats[0]:
            return idx[0] + (t - beats[0]) / spb
        if t >= beats[-1]:
            return idx[-1] + (t - beats[-1]) / spb
        return float(np.interp(t, beats, idx))

    raw = [(pos(n.start) - phase, pos(n.end) - phase, n) for n in notes]
    if shift is None:
        shift = 4 * int(np.ceil(max(0, -min((r[0] for r in raw), default=0)) / 4))
    dt = tuple(divs) if isinstance(divs, (tuple, list)) else (divs,)
    near = lambda p: min((round(p * d) / d for d in dt), key=lambda g: abs(g - p))
    return [pm.Note(velocity=n.velocity, pitch=n.pitch,
                    start=near(s + shift) * spb,
                    end=max(near(e + shift), near(s + shift) + 1 / max(dt)) * spb)
            for s, e, n in raw]


# ──────────────────────────────────────────────────────────────────
#  1-9.  페달 생성
# ──────────────────────────────────────────────────────────────────
def 페달생성(stems, spans, beats, total, min_len=0.2, lift=0.03):
    wav = stems.get("other") or stems.get("piano") or stems.get("mix")
    if wav is None or not Path(wav).exists() or not spans:
        return []
    C = cqt(wav)
    T = C.shape[1]
    energy = C.sum(axis=0)
    if energy.max() > 0:
        energy = energy / energy.max()
    bt = np.asarray(beats, float)
    if len(bt) < 4:
        return []
    T_beat = lambda b: float(bt[min(int(b), len(bt) - 1)])
    segs = []
    merge_b = 1.0
    for s, e, r, k in spans:
        if segs and (e - s) < merge_b:
            segs[-1][1] = e
        else:
            segs.append([s, e])
    lo_thr = 0.15
    out = []
    for s, e in segs:
        t0 = T_beat(s) + 0.015
        t1 = min(T_beat(e), total) - lift
        if t1 - t0 < min_len:
            continue
        i0 = min(int(t0 * SR / HOP), T - 1)
        i1 = min(int(t1 * SR / HOP), T - 1)
        if i1 <= i0:
            continue
        seg_e = energy[i0:i1]
        if seg_e.size == 0 or seg_e.max() < lo_thr:
            continue
        out.append((t0, t1))
    return out


# ──────────────────────────────────────────────────────────────────
#  1-10.  MIDI 저장
# ──────────────────────────────────────────────────────────────────
def timed_midi저장(notes, beats, path, tpb=480, pedal=None):
    mido = _lib("mido")
    beats = np.asarray(beats, float)
    ivs = np.diff(beats)
    idx = np.arange(len(beats), dtype=float)

    def pos(t):
        if t < beats[0]:
            return (t - beats[0]) / ivs[0]
        if t >= beats[-1]:
            return (len(beats) - 1) + (t - beats[-1]) / ivs[-1]
        return float(np.interp(t, beats, idx))

    p0 = pos(0.0)
    tick = lambda t: max(0, int(round((pos(t) - p0) * tpb)))
    mid = mido.MidiFile(type=1, ticks_per_beat=tpb)
    tt = mido.MidiTrack(); mid.tracks.append(tt)
    ev = [(0, mido.MetaMessage("set_tempo", tempo=int(round(ivs[0] * 1e6))))]
    for j, iv in enumerate(ivs):
        ev.append((max(0, int(round((j - p0) * tpb))),
                   mido.MetaMessage("set_tempo", tempo=int(round(iv * 1e6)))))
    ev.append((0, mido.MetaMessage("time_signature", numerator=4, denominator=4)))
    ev.sort(key=lambda e: e[0])
    last = 0
    for t_, m in ev:
        tt.append(m.copy(time=t_ - last)); last = t_
    tr = mido.MidiTrack(); mid.tracks.append(tr)
    tr.append(mido.MetaMessage("track_name", name="Piano", time=0))
    tr.append(mido.Message("program_change", program=0, channel=0, time=0))
    evs = []
    for n in notes:
        on_, off_ = tick(n.start), tick(max(n.end, n.start + 0.03))
        off_ = max(off_, on_ + 1)
        v = int(min(127, max(1, n.velocity)))
        evs.append((on_, 1, mido.Message("note_on", note=int(n.pitch),
                                          velocity=v, channel=0)))
        evs.append((off_, 0, mido.Message("note_off", note=int(n.pitch),
                                           velocity=0, channel=0)))
    for d_, u_ in (pedal or []):
        evs.append((tick(d_), 2, mido.Message("control_change", control=64,
                                               value=127, channel=0)))
        evs.append((max(tick(u_), tick(d_) + 1), -1,
                    mido.Message("control_change", control=64, value=0, channel=0)))
    evs.sort(key=lambda e: (e[0], e[1]))
    last = 0
    for t_, _, m in evs:
        tr.append(m.copy(time=t_ - last)); last = t_
    mid.save(str(path))


# ──────────────────────────────────────────────────────────────────
#  1-11.  악보 생성
# ──────────────────────────────────────────────────────────────────
_QL = [Fraction(1, 6), Fraction(1, 3), Fraction(1, 4), Fraction(1, 2),
       Fraction(2, 3), Fraction(3, 4), Fraction(1), Fraction(4, 3),
       Fraction(3, 2), Fraction(2), Fraction(3), Fraction(4), Fraction(6), Fraction(8)]


def _표현가능(ql):
    return min(_QL, key=lambda q: abs(float(q) - ql))


def 손축소(pitches, hand, max_notes=4, max_span=12):
    ps = sorted(set(pitches))
    if len(ps) <= 1:
        return ps
    anchor = ps[-1] if hand == "r" else ps[0]
    keep = [anchor]
    for q in sorted((x for x in ps if x != anchor), key=lambda x: abs(x - anchor)):
        if len(keep) >= max_notes:
            break
        if max(keep + [q]) - min(keep + [q]) <= max_span:
            keep.append(q)
    return sorted(keep)


def 악보생성(midi_path, xml_path, bpm=100, playable=True, title=None, meter=4):
    from music21 import (converter, stream, clef, meter as m21m,
                          note, chord, key as m21k, pitch as m21p)
    s = converter.parse(str(midi_path), quantizePost=True, quarterLengthDivisors=(4, 3))
    flat = s.flatten()
    right, left = stream.Part(), stream.Part()
    right.insert(0, clef.TrebleClef()); left.insert(0, clef.BassClef())
    items = []
    for el in flat.notes:
        for pp in (list(el.pitches) if el.isChord else [el.pitch]):
            items.append((Fraction(int(round(float(el.offset) * 12)), 12), pp.midi,
                          _표현가능(max(0.25, float(el.quarterLength)))))
    split, prev = {}, 60.0
    for m in range(int(max((o for o, _, _ in items), default=0) // 4) + 1):
        ps = [pm_ for o, pm_, _ in items if int(o // 4) == m]
        if len(ps) >= 2:
            c = [min(ps), max(ps)]
            for _ in range(8):
                lo_ = [x for x in ps if abs(x - c[0]) <= abs(x - c[1])] or [c[0]]
                hi_ = [x for x in ps if abs(x - c[0]) > abs(x - c[1])] or [c[1]]
                c = [float(np.mean(lo_)), float(np.mean(hi_))]
            prev = 0.5 * prev + 0.5 * float(np.clip((c[0] + c[1]) / 2, 54, 66))
        split[m] = prev
    try:
        sharps = int(flat.analyze("key").sharps)
    except Exception:
        sharps = 0

    def mk(ps, ql):
        pitches = []
        for x in ps:
            pp = m21p.Pitch(x)
            if sharps < 0 and pp.accidental is not None and pp.accidental.alter == 1:
                pp = pp.getEnharmonic()
            pitches.append(pp)
        n = chord.Chord(pitches) if len(pitches) > 1 else note.Note(pitches[0])
        n.quarterLength = ql
        return n

    groups = {}
    for o, pm_, ql in items:
        side = "r" if pm_ >= split[int(o // 4)] else "l"
        groups.setdefault((side, o, ql), []).append(pm_)
    if playable:
        byhand = {}
        for (side, o, ql), ps in groups.items():
            for x in ps:
                byhand.setdefault((side, o), []).append((x, ql))
        for side, part in (("r", right), ("l", left)):
            onsets = sorted(o for (sd, o) in byhand if sd == side)
            for i, o in enumerate(onsets):
                lst = byhand[(side, o)]
                keep = 손축소([x for x, _ in lst], side)
                ql = max(q for _, q in lst)
                if i + 1 < len(onsets):
                    ql = min(ql, onsets[i + 1] - o)
                ql = max((q for q in _QL if q <= ql), default=_QL[0])
                part.insert(o, mk(keep, ql))
    else:
        for (side, o, ql), ps in groups.items():
            (right if side == "r" else left).insert(o, mk(ps, ql))

    from music21 import expressions, metadata
    right.insert(0, expressions.TextExpression(f"BPM {round(bpm)}"))
    score = stream.Score([right, left])
    if title:
        score.insert(0, metadata.Metadata(title=title, composer="Song2Piano ULTRA v2"))
    for p in score.parts:
        p.insert(0, m21k.KeySignature(sharps))
        p.insert(0, m21m.TimeSignature(f"{meter}/4"))
        p.makeNotation(inPlace=True)
    score.write("musicxml", fp=str(xml_path))


def pdf생성(xml_path, pdf_path):
    try:
        import verovio, cairosvg
        from pypdf import PdfReader, PdfWriter
    except ImportError as e:
        print(f"[PDF] 스킵: {e}", file=sys.stderr)
        return 0
    tk = verovio.toolkit()
    tk.setOptions({"pageWidth": 2100, "pageHeight": 2970, "scale": 45,
                   "pageMarginLeft": 80, "pageMarginRight": 80,
                   "pageMarginTop": 100, "pageMarginBottom": 100,
                   "adjustPageHeight": False, "breaks": "auto",
                   "svgViewBox": True, "footer": "none"})
    tk.loadData(Path(xml_path).read_text(encoding="utf-8"))
    w = PdfWriter()
    for pg in range(1, tk.getPageCount() + 1):
        svg = tk.renderToSVG(pg)
        buf = io.BytesIO()
        cairosvg.svg2pdf(bytestring=svg.encode("utf-8"), write_to=buf,
                         output_width=793.7, output_height=1122.5)
        for page in PdfReader(io.BytesIO(buf.getvalue())).pages:
            w.add_page(page)
    with open(pdf_path, "wb") as f:
        w.write(f)
    return tk.getPageCount()


# ──────────────────────────────────────────────────────────────────
#  1-12.  파이프라인 본체
# ──────────────────────────────────────────────────────────────────
@dataclass
class 설정:
    audio: Path
    out: Path = Path("out")
    mode: str = "auto"
    ensemble: bool = True
    bpm: Optional[float] = None
    max_sec: Optional[float] = None
    poly: int = 3
    offsets: bool = True
    dynamics: bool = True
    pedal: bool = True
    harmony: bool = True
    legato: bool = False
    grid: str = "auto"
    playable: bool = True
    no_score: bool = False
    cache: Path = field(default_factory=lambda: Path(__file__).parent / "cache")


class Song2Piano:
    def __init__(self, cfg: 설정):
        self.cfg = cfg

    def stage(self, m):
        print(f"STAGE:{m}", flush=True)

    def run(self):
        cfg = self.cfg
        cfg.out.mkdir(parents=True, exist_ok=True)
        cfg.cache.mkdir(exist_ok=True)

        self.stage("음원 분리 + 박자 추적")
        with ThreadPoolExecutor(2) as ex:
            f1 = ex.submit(음원분리, cfg.audio, cfg.cache, cfg.ensemble, cfg.max_sec)
            self.stems = f1.result()
            mix = self.stems.get("mix", list(self.stems.values())[0])
            self.bpm, self.beats, self.downs, self.meter = 박자추적(mix, cfg.bpm)

        self.stage("화음 인식 + 음역 추정")
        with ThreadPoolExecutor(2) as ex:
            f1 = ex.submit(화음인식, self.stems, self.beats)
            f2 = ex.submit(self._음역)
            self.spans, self.key = f1.result()
            self.ranges = f2.result()
        print(f"[조성] {음이름[self.key[0]]} {self.key[1]} (신뢰 {self.key[2]:.2f})")
        print(f"[박자] {self.meter}/4, BPM {self.bpm}")
        if self.spans:
            print("[화음]", " ".join(코드이름(r, k) for _, _, r, k in self.spans[:16]), "...")

        self.stage("음 전사 (BP+CREPE+pyin)")
        parts = self._전사()

        self.stage("편곡")
        멜로디, 베이스, 화음 = self._편곡(parts)

        self.stage("후처리")
        wav_other = (self.stems.get("other") or self.stems.get("vocals")
                     or list(self.stems.values())[0])
        멜로디 = refine_offsets(멜로디, self.stems.get("vocals", wav_other)) if 멜로디 else []
        베이스 = refine_offsets(베이스, self.stems.get("bass", wav_other)) if 베이스 else []
        전체 = list(멜로디) + list(베이스) + list(화음 or [])
        전체 = 배음제거(전체, spans=self.spans, beats=self.beats)
        전체 = 동음병합(전체, 0.05, onsets=온셋환경(wav_other)[2])
        if cfg.dynamics:
            전체 = 벨로시티(전체, wav_other, base=78)
        if cfg.offsets:
            전체 = refine_offsets(전체, wav_other)

        self.stage("격자 정렬")
        divs = self._격자선택(전체)
        quant = 격자정렬(전체, self.bpm, self.beats, 0, divs=divs) \
            if cfg.grid != "off" else list(전체)

        pedal = []
        if cfg.pedal and self.spans:
            pedal = 페달생성(self.stems, self.spans, self.beats,
                             float(self.beats[-1]))

        self.stage("MIDI 출력")
        stem = cfg.audio.stem
        raw_mid = cfg.out / f"{stem}_piano.mid"
        quant_mid = cfg.out / f"{stem}_piano_quantized.mid"
        timed_midi저장(전체, self.beats, raw_mid, pedal=pedal)
        timed_midi저장(quant, self.beats, quant_mid, pedal=pedal)
        if pedal:
            timed_midi저장([], self.beats, cfg.out / f"{stem}_piano_pedal.mid", pedal=pedal)
        print(f"[MIDI] {raw_mid}")

        info = dict(bpm=self.bpm, meter=self.meter,
                    key=[음이름[self.key[0]], self.key[1], round(self.key[2], 2)],
                    grid=list(divs),
                    chords=[(s, e, 코드이름(r, k)) for s, e, r, k in self.spans],
                    layers=dict(melody=len(멜로디), bass=len(베이스),
                                 harmony=len(화음 or [])),
                    ranges={k: list(v) for k, v in self.ranges.items()},
                    ensemble=cfg.ensemble)
        (cfg.out / f"{stem}_analysis.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")

        if not cfg.no_score:
            self.stage("악보 생성")
            try:
                xml = cfg.out / f"{stem}_score.musicxml"
                악보생성(raw_mid, xml, bpm=self.bpm, playable=cfg.playable,
                          title=stem, meter=self.meter)
                pdf = cfg.out / f"{stem}_score.pdf"
                pages = pdf생성(xml, pdf)
                print(f"[악보] {pdf} ({pages}p)")
            except Exception as e:
                print(f"[악보] 실패: {e}", file=sys.stderr)

        self.stage("완료")
        return raw_mid, quant_mid

    def _음역(self):
        r = {}
        for k in ("vocals", "bass", "other", "guitar", "piano"):
            if k in self.stems and Path(self.stems[k]).exists():
                v = 음역추정(self.stems[k])
                if v:
                    r[k] = v
        return r

    def _전사(self):
        cfg, S = self.cfg, self.stems
        P = {}
        vl, vh = self.ranges.get("vocals", (48, 84))
        bl, bh = self.ranges.get("bass", (28, 60))
        ol, oh = self.ranges.get("other", (48, 88))

        if "vocals" in S:
            v_bp = [n for n in basic_pitch전사(S["vocals"], minimum_note_length=80,
                       onset_threshold=0.35, frame_threshold=0.2,
                       minimum_frequency=80, maximum_frequency=1200)
                    if vl <= n.pitch <= vh]
            v_cr = [n for n in crepe전사(S["vocals"], fmin=80, fmax=1200)
                    if vl <= n.pitch <= vh]
            v_py = [n for n in pyin전사(S["vocals"], fmin=80, fmax=1200)
                    if vl <= n.pitch <= vh]
            v = 전사앙상블(v_bp, v_cr, v_py)
            v = 단선율(동음병합(v, 0.12, 온셋환경(S["vocals"])[2]), 0.06)
            v = 옥타브보정(v)
            v = 온셋정밀화(v, S["vocals"])
            P["melody"] = v

        if "bass" in S:
            b = [n for n in basic_pitch전사(S["bass"], minimum_note_length=90,
                     onset_threshold=0.4, frame_threshold=0.25,
                     minimum_frequency=35, maximum_frequency=350)
                 if bl <= n.pitch <= bh]
            P["bass"] = 온셋정밀화(단선율(동음병합(b, 0.15), 0.1), S["bass"])

        if cfg.mode in ("auto", "cover") and "other" in S:
            o = [n for n in basic_pitch전사(S["other"], minimum_note_length=100,
                     onset_threshold=0.45, frame_threshold=0.3,
                     minimum_frequency=ol, maximum_frequency=oh)
                 if ol <= n.pitch <= oh]
            P["other"] = 폴리제한(동음병합(o, 0.2), cfg.poly, 0.2, 30)

        if cfg.mode == "piano":
            src = S.get("piano") or S.get("other")
            if src:
                p = basic_pitch전사(src, minimum_note_length=80,
                                     onset_threshold=0.35, frame_threshold=0.2)
                P["piano"] = 폴리제한(동음병합(p, 0.15), 6, 0.08, 20)
        return P

    def _편곡(self, P):
        cfg = self.cfg
        mel = P.get("melody", [])
        bass = P.get("bass", [])
        if cfg.mode == "piano" and "piano" in P:
            return P["piano"], [], []
        화음 = []
        if cfg.mode in ("auto", "cover") and cfg.harmony and self.spans:
            화음 = self._코드보이싱(self.spans)
        if mel:
            pm = _lib("pretty_midi")
            mel = [n if n.pitch <= 96 else pm.Note(velocity=n.velocity,
                    pitch=n.pitch - 12, start=n.start, end=n.end) for n in mel]
        if 화음 and mel:
            hi = int(np.clip(np.percentile([n.pitch for n in mel], 25) - 2, 55, 72))
            pm = _lib("pretty_midi")
            화음 = [n if n.pitch <= hi else pm.Note(velocity=n.velocity,
                     pitch=max(48, n.pitch - 12), start=n.start, end=n.end)
                    for n in 화음]
        return mel, bass, 화음

    def _코드보이싱(self, spans):
        pm = _lib("pretty_midi")
        bt = self.beats
        out, prev = [], None
        for s, e, root, kind in spans:
            pcs = [(root + i) % 12 for i in 코드종류.get(kind, (0, 4, 7))]
            cands = [[p for p in range(48, 85) if p % 12 == pc] for pc in pcs]
            best, bd = None, 1e9
            for a in cands[0]:
                for b in cands[1]:
                    for c in cands[2]:
                        v = sorted([a, b, c])
                        if v[0] < 48 or v[-1] > 84 or v[-1] - v[0] > 16:
                            continue
                        d = 0 if prev is None else sum(abs(x - y) for x, y in zip(v, prev))
                        if d < bd:
                            best, bd = v, d
            if best is None:
                continue
            prev = best
            t0 = float(bt[min(s, len(bt) - 1)])
            t1 = float(bt[min(e, len(bt) - 1)]) if e < len(bt) \
                else float(bt[-1] + (bt[-1] - bt[-2]))
            out += [pm.Note(velocity=58, pitch=p, start=t0, end=t1) for p in best]
        return out

    def _격자선택(self, notes):
        cfg = self.cfg
        if cfg.grid == "off":
            return (4,)
        n = {"8": (2,), "16": (4,), "3": (3,), "mixed": (4, 3), "32": (8,)}
        if cfg.grid != "auto":
            return n.get(cfg.grid, (4,))
        ons = np.array([x.start for x in notes])
        if len(ons) < 30:
            return (4,)
        bt = np.asarray(self.beats, float)
        pos = np.interp(ons, bt, np.arange(len(bt)))
        frac = pos - np.floor(pos)
        tol = 0.028 * len(bt) / max(bt[-1] - bt[0], 1e-6)
        n16 = int(np.sum((np.abs(frac - 0.25) <= tol) | (np.abs(frac - 0.75) <= tol)))
        n8 = int(np.sum(np.abs(frac - 0.5) <= tol))
        n3 = int(np.sum((np.abs(frac - 1 / 3) <= tol) | (np.abs(frac - 2 / 3) <= tol)))
        if n3 >= 25 and n3 > 1.5 * n16:
            return (4, 3) if n16 > 0.4 * n3 else (3,)
        if n16 >= 0.15 * max(n16 + n8, 1) and n16 >= 20:
            return (4,)
        return (2,) if n8 >= 20 else (4,)


# ──────────────────────────────────────────────────────────────────
#  1-13.  웹 서버
# ──────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Song2Piano ULTRA v2</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans KR",sans-serif;
background:linear-gradient(135deg,#0f172a,#1e293b);color:#e2e8f0;min-height:100vh;
padding:20px;display:flex;justify-content:center;align-items:flex-start}
.wrap{max-width:800px;width:100%}
h1{font-size:30px;font-weight:800;margin-bottom:6px;
background:linear-gradient(90deg,#60a5fa,#a78bfa,#f472b6);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.sub{color:#94a3b8;font-size:14px;margin-bottom:20px;line-height:1.6}
.card{background:rgba(30,41,59,.7);border:1px solid rgba(148,163,184,.2);
border-radius:16px;padding:22px;margin-bottom:14px;backdrop-filter:blur(10px)}
.drop{position:relative;border:2px dashed rgba(96,165,250,.4);border-radius:12px;
padding:36px 20px;text-align:center;cursor:pointer;transition:all .2s;
background:rgba(15,23,42,.4);user-select:none}
.drop:hover,.drop.over{border-color:#60a5fa;background:rgba(96,165,250,.1)}
.drop:focus{outline:none;border-color:#60a5fa;box-shadow:0 0 0 3px rgba(96,165,250,.2)}
.drop .icon{font-size:44px;margin-bottom:10px;pointer-events:none}
.drop .label{font-size:15px;font-weight:600;margin-bottom:4px;pointer-events:none}
.drop .hint{font-size:12px;color:#94a3b8;pointer-events:none}
.opts{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:14px}
.opt label{display:block;font-size:12px;color:#94a3b8;margin-bottom:5px}
select{width:100%;padding:9px 11px;border-radius:8px;background:rgba(15,23,42,.8);
color:#e2e8f0;border:1px solid rgba(148,163,184,.25);font-size:13px;font-family:inherit}
select:focus{outline:none;border-color:#60a5fa}
.checks{display:flex;flex-wrap:wrap;gap:12px;margin-top:14px}
.checks label{display:flex;align-items:center;gap:5px;font-size:12px;color:#cbd5e1;cursor:pointer}
.checks input{accent-color:#60a5fa}
button{width:100%;padding:13px;border:none;border-radius:10px;font-size:15px;
font-weight:700;cursor:pointer;margin-top:14px;
background:linear-gradient(90deg,#3b82f6,#8b5cf6);color:#fff;
transition:transform .1s,opacity .2s;font-family:inherit}
button:hover:not(:disabled){transform:translateY(-1px)}
button:disabled{opacity:.5;cursor:not-allowed}
.progress{height:8px;background:rgba(15,23,42,.8);border-radius:4px;
overflow:hidden;margin:16px 0 10px}
.progress-bar{height:100%;background:linear-gradient(90deg,#3b82f6,#8b5cf6,#ec4899);
width:0%;transition:width .4s;background-size:200% 100%;
animation:shift 2s linear infinite}
@keyframes shift{0%{background-position:0% 0}100%{background-position:200% 0}}
.stage{font-size:13px;color:#cbd5e1;text-align:center}
color.files{margin-top:16px}
.file-row{:#display:flex;align-items:center;justify-contentf:space-between;
padding:ca11px 14px;background:rgba(15,523,42,.6);border-radius:8px;
margina-bottom:7px;border:1px solid rgba(148,163,184,.15)}
.file-row .name{font-size:13px}
.file-row a{color:#60a5fa;text-decoration:none;font-weight:600;font-size:12px;
padding:5px 12px;border-radius:6px;background:rgba(96,165,250,.15);transition:background .2s}
.file-row a:hover{background:rgba(96,165,250,.3)}
.info{font-size:12px;color:#94a3b8;line-height:1.7;margin-top:6px}
.error{5;background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);
padding:10px;border-radius:8px;font-size:12px;margin-top:10px;
white-space:pre-wrap;word-break:break-all}
.hidden{display:none}
.tag{display:inline-block;padding:2px 9px;font-size:10px;
background:rgba(96,165,250,.2);color:#93c5fd;border-radius:10px;
margin-right:5px;margin-top:5px}
#file{position:absolute;left:-9999px;top:0;opacity:0;width:1px;height:1px}
</style></head>
<body><div class="wrap">
<h1>🎹 Song2Piano ULTRA v2</h1>
<p class="sub">Demucs 앙상블 + Beat This! + 3중 전사 앙상블.<br>
음원 분리 → 음 전사 → 화음 인식 → 편곡 → 페달 → 악보까지 자동.</p>
<div class="card">
<div class="drop" id="drop" tabindex="0" role="button">
<div class="icon">🎵</div>
<div class="label">음악 파일을 드래그하거나 클릭</div>
<div class="hint">MP3, WAV, M4A, FLAC, OGG (최대 100MB)</div></div>
<input type="file" id="file" accept="audio/*,.mp3,.wav,.m4a,.flac,.ogg">
<div class="sub" id="fname" style="margin-top:10px;color:#93c5fd"></div>
<div class="opts">
<div class="opt"><label>편곡 모드</label>
<select id="mode">
<option value="auto">자동</option>
<option value="cover">팝/록 커버</option>
<option value="piano">피아노곡 특화</option>
<option value="vocal">보컬 중심</option>
<option value="instrumental">연주곡</option></select></div>
<div class="opt"><label>격자</label>
<select id="grid">
<option value="auto">자동</option>
<option value="8">8분음표</option>
<option value="16">16분음표</option>
<option value="3">3연음</option>
<option value="mixed">16분+3연음</option>
<option value="off">정렬 안 함</option></select></div>
</div>
<div class="checks">
<label><input type="checkbox" id="opt-ens" checked> 앙상블 (고품질)</label>
<label><input type="checkbox" id="opt-pedal" checked> 페달</label>
<label><input type="checkbox" id="opt-dynamics" checked> 다이내믹</label>
<label><input type="checkbox" id="opt-harmony" checked> 화음</label>
<label><input type="checkbox" id="opt-offsets" checked> 길이 정밀</label>
<label><input type="checkbox" id="opt-score" checked> 악보(PDF)</label>
</div>
<button id="go" disabled>🎹 변환 시작</button>
</div>
<div class="card hidden" id="prog-card">
<div class="progress"><div class="progress-bar" id="bar"></div></div>
<div class="stage" id="stage">준비 중…</div>
<div class="error hidden" id="err"></div></div>
<div class="card hidden" id="res-card">
<div style="font-size:15px;font-weight:700;margin-bottom:10px">✅ 변환 완료</div>
<div id="meta" class="info"></div><div class="files" id="files"></div></div>
<div class="card" style="font-size:11px;color:#64748b;line-height:1.7">
<b>지원 장르</b>
<span class="tag">팝/록</span><span class="tag">발라드</span><span class="tag">재즈</span>
<span class="tag">K-POP</span><span class="tag">클래식</span><span class="tag">EDM</span><span class="tag">힙합</span><br>
· 앙상블 켜면 품질 ↑, 시간 ↑ (곡당 8~20분).<br>
· 앙상블 끄면 3~10분.</div>
</div>
<script>
(function(){
'use strict';
var $=function(id){return document.getElementById(id)};
var fileIn=$('file'),drop=$('drop'),fname=$('fname'),go=$('go'),
progCard=$('prog-card'),bar=$('bar'),stageEl=$('stage'),errEl=$('err'),
resCard=$('res-card'),metaEl=$('meta'),filesEl=$('files');
var chosen=null,polling=false;
drop.addEventListener('click',function(e){e.preventDefault();e.stopPropagation();fileIn.click();});
drop.addEventListener('keydown',function(e){if(e.key==='Enter'||e.key===' '){e.preventDefault();fileIn.click();}});
fileIn.addEventListener('change',function(){if(fileIn.files&&fileIn.files.length)pick(fileIn.files[0]);});
['dragenter','dragover'].forEach(function(ev){drop.addEventListener(ev,function(e){e.preventDefault();e.stopPropagation();drop.classList.add('over');});});
['dragleave','dragend'].forEach(function(ev){drop.addEventListener(ev,function(e){e.preventDefault();e.stopPropagation();drop.classList.remove('over');});});
drop.addEventListener('drop',function(e){e.preventDefault();e.stopPropagation();drop.classList.remove('over');
var f=e.dataTransfer&&e.dataTransfer.files&&e.dataTransfer.files[0];if(f)pick(f);});
['dragover','drop'].forEach(function(ev){window.addEventListener(ev,function(e){e.preventDefault();});});
document.addEventListener('paste',function(e){var items=e.clipboardData&&e.clipboardData.items;if(!items)return;
for(var i=0;i<items.length;i++){if(items[i].kind==='file'){var f=items[i].getAsFile();if(f){pick(f);break;}}}});
function pick(f){
if(f.size>100*1024*1024){alert('파일이 너무 큽니다 (최대 100MB)');return;}
var okExt=/\.(mp3|wav|m4a|flac|ogg|aac|opus|wma|webm)$/i;
var okType=f.type&&f.type.indexOf('audio')===0;
if(!okType&&!okExt.test(f.name)){alert('오디오 파일만 업로드할 수 있습니다');return;}
chosen=f;fname.textContent='📎 '+f.name+' ('+(f.size/1048576).toFixed(1)+'MB)';
go.disabled=false;go.textContent='🎹 변환 시작';}
go.addEventListener('click',async function(){
if(!chosen){alert('먼저 파일을 선택하세요');return;}
go.disabled=true;progCard.classList.remove('hidden');resCard.classList.add('hidden');
errEl.classList.add('hidden');bar.style.width='5%';stageEl.textContent='업로드 중…';
var fd=new FormData();
fd.append('file',chosen);fd.append('mode',$('mode').value);fd.append('grid',$('grid').value);
fd.append('ensemble',$('opt-ens').checked?'1':'0');
fd.append('pedal',$('opt-pedal').checked?'1':'0');
fd.append('dynamics',$('opt-dynamics').checked?'1':'0');
fd.append('harmony',$('opt-harmony').checked?'1':'0');
fd.append('offsets',$('opt-offsets').checked?'1':'0');
fd.append('score',$('opt-score').checked?'1':'0');
try{var r=await fetch('/convert',{method:'POST',body:fd});
var text=await r.text();var j;
try{j=JSON.parse(text);}catch(e){throw new Error('서버 응답 오류: '+text.slice(0,200));}
if(!r.ok||j.error)throw new Error(j.error||('업로드 실패 HTTP '+r.status));
polling=true;poll(j.id);
}catch(e){fail(e.message||String(e));}});
var STAGES=[['시작',5],['분리',12],['박자',30],['화음',42],['음역',48],
['전사',62],['편곡',74],['후처리',82],['격자',86],['MIDI',90],['악보',95],['완료',100]];
var missCount=0;
function bump(s){for(var i=0;i<STAGES.length;i++){if(s.indexOf(STAGES[i][0])!==-1){bar.style.width=STAGES[i][1]+'%';return;}}
var cur=parseFloat(bar.style.width)||5;bar.style.width=Math.min(90,cur+2)+'%';}
async function poll(id){
if(!polling)return;
try{var r=await fetch('/status/'+id,{cache:'no-store'});var j=await r.json();
missCount=0;
if(j.status==='running'){var s=j.stage||'처리 중…';stageEl.textContent=s;bump(s);
setTimeout(function(){poll(id);},1500);}
else if(j.status==='done'){bar.style.width='100%';stageEl.textContent='완료!';showResult(j);}
else{fail(j.error||'알 수 없는 오류');}
}catch(e){missCount++;if(missCount>5){fail('서버 응답 없음: '+e.message);return;}
setTimeout(function(){poll(id);},2500);}}
function showResult(j){
polling=false;
setTimeout(function(){
progCard.classList.add('hidden');resCard.classList.remove('hidden');
var m=j.meta||{},info=[];
if(m.bpm)info.push('BPM '+m.bpm);
if(m.meter)info.push('박자 '+m.meter+'/4');
if(m.key)info.push('조성 '+m.key[0]+' '+m.key[1]);
if(m.ensemble)info.push('앙상블 ✓');
if(m.layers)info.push('멜로디 '+(m.layers.melody||0)+' · 베이스 '+(m.layers.bass||0)+' · 화음 '+(m.layers.harmony||0));
metaEl.innerHTML=info.join(' · ');
filesEl.innerHTML='';
var names={'_piano.mid':'🎼 원본 리듬 MIDI','_piano_quantized.mid':'📐 격자 MIDI',
'_piano_pedal.mid':'🎹 페달만','_score.musicxml':'📄 MusicXML',
'_score.pdf':'📕 악보 PDF','_analysis.json':'📊 분석 JSON'};
(j.files||[]).forEach(function(f){
var lbl=f;Object.keys(names).forEach(function(k){if(f.indexOf(k)!==-1)lbl=names[k];});
var row=document.createElement('div');row.className='file-row';
row.innerHTML='<span class="name">'+lbl+'<br><span style="font-size:10px;color:#64748b">'+f+'</span></span>'+
'<a href="/jobs/'+j.id+'/'+encodeURIComponent(f)+'" download>다운로드</a>';
filesEl.appendChild(row);});
go.disabled=false;go.textContent='🔄 다시 변환';},400);}
function fail(msg){polling=false;bar.style.width='0%';stageEl.textContent='오류 발생';
errEl.textContent=msg;errEl.classList.remove('hidden');go.disabled=false;go.textContent='🔁 다시 시도';}
})();
</script></body></html>
"""


def 웹서버실행(host="0.0.0.0", port=8000):
    from flask import Flask, request, jsonify, send_from_directory, Response
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
    ROOT = Path(__file__).parent
    JOBS = ROOT / "jobs"; JOBS.mkdir(exist_ok=True)
    STATE = {}
    JID = re.compile(r"^[0-9a-f]{10}$")

    @app.get("/")
    def index():
        return Response(HTML, mimetype="text/html")

    @app.get("/health")
    def health():
        return jsonify(ok=True)

    def 작업(jid, cfg, d):
        tail = []
        try:
            sp = Song2Piano(cfg)
            orig = sp.stage
            def patched(m):
                STATE[jid] = dict(status="running", stage=m)
                orig(m)
            sp.stage = patched
            sp.run()
            files = sorted({x.name for pat in ("*_piano*.*", "*_score.*", "*_analysis.*")
                            for x in d.glob(pat)})
            meta = {}
            mp = d / "meta.json"
            if mp.exists():
                try: meta = json.loads(mp.read_text(encoding="utf-8"))
                except Exception: pass
            for ap in d.glob("*_analysis.json"):
                try:
                    a = json.loads(ap.read_text(encoding="utf-8"))
                    for k in ("bpm", "meter", "key", "grid", "layers", "ensemble"):
                        if k in a: meta[k] = a[k]
                except Exception: pass
            STATE[jid] = dict(status="done", files=files, meta=meta)
        except Exception:
            import traceback
            tail = traceback.format_exc().splitlines()[-25:]
            STATE[jid] = dict(status="error", error="\n".join(tail)[-1500:])

    @app.post("/convert")
    def convert():
        up = request.files.get("file")
        if not up:
            return jsonify(error="파일이 없습니다"), 400
        jid = uuid.uuid4().hex[:10]
        d = JOBS / jid; d.mkdir()
        src = d / ("input" + Path(up.filename or "a.wav").suffix.lower()[:8])
        up.save(src)
        meta = dict(name=up.filename or "audio", ts=int(time.time()),
                    mode=request.form.get("mode", "auto"))
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        cfg = 설정(
            audio=src, out=d,
            mode=request.form.get("mode", "auto") if request.form.get("mode") in
                 ("auto", "piano", "cover", "vocal", "instrumental") else "auto",
            grid=request.form.get("grid", "auto") if request.form.get("grid") in
                 ("auto", "8", "16", "3", "mixed", "32", "off") else "auto",
            ensemble=request.form.get("ensemble", "1") == "1",
            pedal=request.form.get("pedal", "1") == "1",
            dynamics=request.form.get("dynamics", "1") == "1",
            harmony=request.form.get("harmony", "1") == "1",
            offsets=request.form.get("offsets", "1") == "1",
            no_score=request.form.get("score", "1") != "1",
            cache=ROOT / "cache",
        )
        STATE[jid] = dict(status="running", stage="시작")
        threading.Thread(target=작업, args=(jid, cfg, d), daemon=True).start()
        return jsonify(id=jid)

    def job_info(jid):
        d = JOBS / jid
        if not JID.match(jid) or not d.is_dir():
            return None
        meta = {}
        try: meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        except Exception: pass
        for ap in d.glob("*_analysis.json"):
            try:
                a = json.loads(ap.read_text(encoding="utf-8"))
                for k in ("bpm", "meter", "key", "grid", "layers", "ensemble"):
                    if k in a: meta[k] = a[k]
            except Exception: pass
        st = STATE.get(jid)
        files = sorted({x.name for pat in ("*_piano*.*", "*_score.*", "*_analysis.*")
                        for x in d.glob(pat)})
        if st is None:
            st = dict(status="done", files=files, meta=meta) if files else \
                 dict(status="error", error="작업을 찾을 수 없어요")
        info = dict(st)
        info["id"] = jid
        info.setdefault("meta", meta)
        return info

    @app.get("/status/<jid>")
    def status(jid):
        info = job_info(jid)
        return jsonify(info or dict(status="error", error="없는 작업"))

    @app.get("/jobs/<jid>/<path:name>")
    def getfile(jid, name):
        if not JID.match(jid):
            return jsonify(error="잘못된 요청"), 400
        return send_from_directory(JOBS / jid, name, as_attachment=True)

    print("=" * 60)
    print(f"🎹 Song2Piano ULTRA v2 웹 서버")
    print(f"   접속: http://localhost:{port}")
    print("=" * 60, flush=True)
    app.run(host=host, port=port, threaded=True, debug=False)


# ──────────────────────────────────────────────────────────────────
#  1-14.  진입점
# ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Song2Piano ULTRA v2")
    ap.add_argument("audio", nargs="?", type=Path, help="(선택) CLI 변환용 파일")
    ap.add_argument("-o", "--out", type=Path, default=Path("out"))
    ap.add_argument("--mode", choices=["auto", "piano", "cover", "vocal", "instrumental"],
                    default="auto")
    ap.add_argument("--no-ensemble", action="store_true")
    ap.add_argument("--bpm", type=float, default=None)
    ap.add_argument("--max-sec", type=float, default=None)
    ap.add_argument("--grid", choices=["auto", "8", "16", "3", "mixed", "32", "off"],
                    default="auto")
    ap.add_argument("--no-pedal", action="store_true")
    ap.add_argument("--no-dynamics", action="store_true")
    ap.add_argument("--no-harmony", action="store_true")
    ap.add_argument("--no-offsets", action="store_true")
    ap.add_argument("--no-score", action="store_true")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()

    if a.audio:
        cfg = 설정(audio=a.audio, out=a.out, mode=a.mode,
                   ensemble=not a.no_ensemble, bpm=a.bpm,
                   max_sec=a.max_sec, grid=a.grid,
                   pedal=not a.no_pedal, dynamics=not a.no_dynamics,
                   harmony=not a.no_harmony, offsets=not a.no_offsets,
                   no_score=a.no_score)
        Song2Piano(cfg).run()
    else:
        웹서버실행(a.host, a.port)


if __name__ == "__main__":
    main()
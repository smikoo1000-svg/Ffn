#!/usr/bin/env python3
"""
PianoForge Studio v15 — Universal Transcription + Score Quality Engine
[신규] 온라인 공유 시스템 + YouTube 링크 변환 (쿠키 인증 지원)
"""
# ═════════════════════════════════════════════════════════════════════════════
# 0. Bootstrap
# ═════════════════════════════════════════════════════════════════════════════
import os, sys, shutil, subprocess
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_VENV = _HERE / ".venv"
_VENV_PY = _VENV / "bin" / "python"
_MARK = _VENV / ".ready"

_PKGS = [
    "flask", "flask-cors", "waitress",
    "numpy<2.0", "scipy", "librosa>=0.10.2", "soundfile",
    "pretty_midi", "mido", "music21>=9.1",
    "onnxruntime", "setuptools<81",
    "basic-pitch[onnx]", "piano-transcription-inference", "demucs",
    "pyfluidsynth", "verovio", "cairosvg", "pypdf", "tqdm",
    "huggingface_hub", "transformers", "safetensors",
    "yt-dlp", "bgutil-ytdlp-pot-provider",
]


def _in_correct_venv():
    try:
        return _VENV_PY.exists() and Path(sys.executable).resolve() == _VENV_PY.resolve()
    except Exception:
        return False


def _bootstrap():
    in_venv = _in_correct_venv()
    installed = _MARK.exists() and _VENV_PY.exists()
    if in_venv and installed: return
    if (not in_venv) and installed:
        os.execv(str(_VENV_PY), [str(_VENV_PY), __file__] + sys.argv[1:]); return
    if os.environ.get("_PF_BOOTSTRAP_ATTEMPTED") == "1":
        print("[warn] bootstrap 재시도"); return
    os.environ["_PF_BOOTSTRAP_ATTEMPTED"] = "1"

    print("=" * 72); print("  PianoForge Studio v15 — 최초 환경 구성 (6~10분)"); print("=" * 72)

    if not _VENV_PY.exists():
        uv = shutil.which("uv")
        if not uv:
            try: subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--user", "uv"], check=True)
            except Exception: pass
            for c in [Path.home() / ".local/bin/uv", Path.home() / ".cargo/bin/uv"]:
                if c.exists():
                    os.environ["PATH"] = f"{c.parent}:{os.environ['PATH']}"; break
            uv = shutil.which("uv")
        if uv:
            if _VENV.exists(): shutil.rmtree(_VENV, ignore_errors=True)
            try: subprocess.run([uv, "venv", "--seed", "--python", "3.11", str(_VENV)], check=True)
            except Exception:
                subprocess.run([uv, "python", "install", "3.11"], check=False)
                subprocess.run([uv, "venv", "--seed", "--python", "3.11", str(_VENV)], check=True)
        else:
            subprocess.run([sys.executable, "-m", "venv", str(_VENV)], check=True)

    print("[info] 시스템 라이브러리 설치 중...", flush=True)
    try:
        subprocess.run(["sudo", "apt-get", "update", "-qq"], check=True, capture_output=True)
        subprocess.run(["sudo", "apt-get", "install", "-y", "-qq", "fluidsynth", "libfluidsynth-dev", "ffmpeg", "libsndfile1"], check=True, capture_output=True)
    except Exception as e:
        print(f"[warn] 시스템 라이브러리 설치 실패: {e}", flush=True)

    uv = shutil.which("uv")
    print("[info] PyTorch 설치 중...", flush=True)
    if uv:
        subprocess.run([uv, "pip", "install", "--python", str(_VENV_PY), "--upgrade", "pip", "setuptools", "wheel"], capture_output=True)
    else:
        try:
            subprocess.run([str(_VENV_PY), "-m", "pip", "install", "-q", "--upgrade", "pip", "setuptools", "wheel"], check=True)
        except Exception:
            import urllib.request
            gp = _HERE / "get-pip.py"
            urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", gp)
            subprocess.run([str(_VENV_PY), str(gp), "--upgrade"], check=True)
            gp.unlink(missing_ok=True)

    torch_pkgs = ["torch", "torchaudio"]
    torch_index = "https://download.pytorch.org/whl/cpu"
    if uv:
        r = subprocess.run([uv, "pip", "install", "--python", str(_VENV_PY), "--index-url", torch_index] + torch_pkgs, capture_output=True, text=True)
    else:
        r = subprocess.run([str(_VENV_PY), "-m", "pip", "install", "-q", "--index-url", torch_index] + torch_pkgs, capture_output=True, text=True)
    if r.returncode != 0: raise RuntimeError("PyTorch 설치 실패")

    print("[info] 악보 필수 패키지 설치 중...", flush=True)
    for pkgs in (["music21>=9.1", "verovio", "cairosvg", "pyfluidsynth"], _PKGS):
        if uv:
            res = subprocess.run([uv, "pip", "install", "--python", str(_VENV_PY)] + pkgs, capture_output=True, text=True)
        else:
            res = subprocess.run([str(_VENV_PY), "-m", "pip", "install", "-q", "--prefer-binary"] + pkgs, capture_output=True, text=True)
        if res.returncode != 0:
            if uv:
                res2 = subprocess.run([str(_VENV_PY), "-m", "pip", "install", "-q", "--prefer-binary"] + pkgs, capture_output=True, text=True)
                if res2.returncode != 0: raise RuntimeError(f"패키지 설치 실패: {pkgs}")
            else:
                raise RuntimeError(f"패키지 설치 실패: {pkgs}")

    _MARK.write_text("ok")
    os.execv(str(_VENV_PY), [str(_VENV_PY), __file__] + sys.argv[1:])


_bootstrap()

# ═════════════════════════════════════════════════════════════════════════════
# 1. Imports
# ═════════════════════════════════════════════════════════════════════════════
import hashlib, json, logging, threading, time, uuid, warnings, re
from typing import Dict, List, Optional, Tuple
warnings.filterwarnings("ignore")
import numpy as np
import pretty_midi
import librosa
import soundfile as sf
from flask import Flask, render_template_string, request, jsonify, send_file, abort
from flask_cors import CORS

try:
    from music21 import converter, meter, clef, instrument, dynamics
    HAS_M21 = True
except ImportError: HAS_M21 = False
try:
    import verovio
    HAS_VEROVIO = True
except ImportError: HAS_VEROVIO = False
try:
    from piano_transcription_inference import PianoTranscription, sample_rate as BD_SR
    HAS_BD = True
except ImportError: HAS_BD = False
try:
    from basic_pitch.inference import predict as bp_predict, Model as BPModel
    from basic_pitch import ICASSP_2022_MODEL_PATH
    HAS_BP = True
except ImportError: HAS_BP = False

HAS_MUSCRIPTOR = False
try:
    import muscriptor
    HAS_MUSCRIPTOR = True
except ImportError: pass

HAS_DEMUCS = False
try:
    import torch
    from demucs.pretrained import get_model as demucs_get_model
    from demucs.apply import apply_model as demucs_apply
    HAS_DEMUCS = True
except ImportError: pass

HAS_FLUID = False
try:
    import fluidsynth
    HAS_FLUID = True
except ImportError: pass

HAS_YTDLP = False
try:
    import yt_dlp
    HAS_YTDLP = True
except ImportError: pass

# ═════════════════════════════════════════════════════════════════════════════
# 2. Logging
# ═════════════════════════════════════════════════════════════════════════════
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pf")


class Log:
    _t0 = time.time()
    @classmethod
    def info(cls, m): log.info(f"[{time.time() - cls._t0:6.1f}s] {m}")
    @classmethod
    def stage(cls, m): log.info(f"\n{'━' * 4} {m} {'━' * 4}")
    @classmethod
    def warn(cls, m): log.warning(m)
    @classmethod
    def err(cls, m): log.error(m)


_MODEL_LOCK = threading.Lock()

# ═════════════════════════════════════════════════════════════════════════════
# 3. Audio Utilities
# ═════════════════════════════════════════════════════════════════════════════
def load_audio(path, sr=22050, mono=True, max_sec=None):
    try:
        data, orig = sf.read(str(path), always_2d=True)
        data = data.mean(axis=1) if mono else data.T
        if orig != sr: data = librosa.resample(data, orig_sr=orig, target_sr=sr)
    except Exception:
        data, _ = librosa.load(str(path), sr=sr, mono=mono)
    if max_sec: data = data[..., :int(max_sec * sr)]
    return np.ascontiguousarray(data, dtype=np.float32)


def rms_norm(y, db=-20.0):
    r = np.sqrt(np.mean(y ** 2)) + 1e-9
    return (y * (10 ** (db / 20) / r)).astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# 4. Genre Analyzer
# ═════════════════════════════════════════════════════════════════════════════
class GenreAnalyzer:
    def __init__(self, y, sr=22050):
        self.y = rms_norm(y); self.sr = sr
        self.tempo = 0.0; self.harmonic_ratio = 0.5; self.piano_band_ratio = 0.0
        self.zcr = 0.0; self.spectral_flatness = 0.0; self.onset_density = 0.0
        self.spectral_centroid = 0.0; self.sub_bass_ratio = 0.0

    def extract(self):
        y, sr = self.y, self.sr
        env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
        tempo, beats = librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=512,
            units="time", start_bpm=150, tightness=100, trim=False)
        self.tempo = float(tempo) if np.isscalar(tempo) else float(tempo[0])
        S = np.abs(librosa.stft(y))
        self.spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)))
        self.spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(S=S)))
        self.zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        yh, yp = librosa.effects.hpss(y)
        eh = float(np.mean(yh ** 2)); ep = float(np.mean(yp ** 2))
        self.harmonic_ratio = eh / (eh + ep + 1e-9)
        ons = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=512, units="time")
        dur = len(y) / sr
        self.onset_density = len(ons) / max(dur, 1.0)
        freqs = librosa.fft_frequencies(sr=sr)
        tot = float(np.mean(S)) + 1e-9
        sb = freqs < 100
        self.sub_bass_ratio = float(np.mean(S[sb, :])) / tot if sb.any() else 0.0
        pb = (freqs >= 200) & (freqs <= 4000)
        self.piano_band_ratio = float(np.mean(S[pb, :])) / tot if pb.any() else 0.0
        return self

    def has_piano(self):
        s = 0
        if self.harmonic_ratio > 0.55: s += 2
        if self.piano_band_ratio > 0.3: s += 2
        if self.zcr < 0.15: s += 1
        if self.spectral_flatness < 0.02: s += 1
        return s >= 4

    def classify_genre(self):
        bpm = self.tempo
        if 118 <= bpm <= 138 and self.harmonic_ratio < 0.55: return "edm"
        if 140 <= bpm <= 200 and self.onset_density > 4.0: return "fnf"
        if 150 <= bpm <= 210 and self.spectral_centroid > 2200: return "vocaloid"
        if 125 <= bpm <= 175 and self.sub_bass_ratio > 0.15: return "phonk"
        if self.harmonic_ratio > 0.65 and self.onset_density < 3.0: return "classical_jazz"
        return "pop_rock"


# ═════════════════════════════════════════════════════════════════════════════
# 5. 악보 품질 개선
# ═════════════════════════════════════════════════════════════════════════════
def separate_voices(notes, max_voices=3):
    if not notes: return []
    notes_sorted = sorted(notes, key=lambda n: n.start)
    chords, cur, cs = [], [], None
    for n in notes_sorted:
        if cs is None: cs, cur = n.start, [n]
        elif n.start - cs < 0.08: cur.append(n)
        else:
            if cur: chords.append(cur)
            cs, cur = n.start, [n]
    if cur: chords.append(cur)
    up, low = [], []
    for ch in chords:
        chs = sorted(ch, key=lambda n: n.pitch)
        if len(chs) == 1: up.append(chs[0])
        else:
            up.append(chs[-1])
            for n in chs[:-1]: low.append(n)
    v = []
    if up: v.append(up)
    if low: v.append(low)
    return v[:max_voices]


def assign_to_grand_staff(voices, split_pitch=60):
    treble, bass = [], []
    for voice in voices:
        for n in voice:
            (treble if n.pitch >= split_pitch else bass).append(n)
    return treble, bass


def quantize_beat_aware(notes, beats, bpm, grid_div=16):
    if len(beats) < 4 or not notes: return notes
    beats = np.asarray(beats, float)
    spb = 60.0 / max(bpm, 1.0)
    gu = spb / 4.0
    out = []
    for n in notes:
        idx = np.searchsorted(beats, n.start)
        idx = max(1, min(len(beats) - 1, idx))
        pb, nb = beats[idx - 1], beats[idx]
        bp = (n.start - pb) / max(nb - pb, 1e-6)
        ss = round(bp * 4) / 4
        ss_start = pb + ss * (nb - pb)
        dur = max(gu, n.end - n.start)
        sd = max(gu, round(dur / gu) * gu)
        out.append(pretty_midi.Note(velocity=n.velocity, pitch=n.pitch,
                                     start=max(0.0, ss_start), end=ss_start + sd))
    return sorted(out, key=lambda n: n.start)


def add_articulations(notes, bpm):
    if not notes: return notes
    spb = 60.0 / max(bpm, 1.0)
    out = []
    for n in notes:
        dur = n.end - n.start
        ne = n.start + dur * 0.6 if dur < spb * 0.4 else n.end
        out.append(pretty_midi.Note(velocity=n.velocity, pitch=n.pitch,
                                     start=n.start, end=max(n.start + 0.05, ne)))
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 6. Stem Separation
# ═════════════════════════════════════════════════════════════════════════════
_STEMS = ("vocals", "drums", "bass", "other", "mix")
_demucs_model = None


def _get_demucs():
    global _demucs_model
    if not HAS_DEMUCS: return None
    with _MODEL_LOCK:
        if _demucs_model is None:
            try:
                _demucs_model = demucs_get_model("htdemucs")
                _demucs_model.eval()
            except Exception as e:
                Log.warn(f"HTDemucs 로드 실패: {e}"); return None
        return _demucs_model


def separate_stems(audio, work, no_separate=False):
    stems = {n: work / f"{n}.wav" for n in _STEMS}
    if all(p.exists() for p in stems.values()): return stems
    mix = work / f"{audio.stem}.wav"
    if not mix.exists():
        data = load_audio(audio, sr=44100, mono=False)
        if data.ndim == 1: data = data[:, None]
        sf.write(str(mix), data.T, 44100, subtype="PCM_16")
    shutil.copy(mix, stems["mix"])
    if no_separate:
        for n in ("vocals", "drums", "bass", "other"): shutil.copy(mix, stems[n])
        return stems
    model = _get_demucs()
    if model is None:
        for n in ("vocals", "drums", "bass", "other"): shutil.copy(mix, stems[n])
        return stems
    try:
        Log.stage("HTDemucs 4-stem 분리")
        wav, sr = sf.read(str(mix), always_2d=True); wav = wav.T
        if sr != 44100:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=44100); sr = 44100
        if wav.shape[0] == 1: wav = np.vstack([wav, wav])
        wav = wav[:2, :]
        import torch
        tensor = torch.from_numpy(wav).float().unsqueeze(0)
        with torch.no_grad():
            sources = demucs_apply(model, tensor, device="cpu", shifts=0, split=True, overlap=0.25)
        for i, name in enumerate(model.sources):
            if name in stems:
                out = sources[0, i].cpu().numpy()
                out_mono = out.mean(axis=0) if out.ndim > 1 else out
                sf.write(str(stems[name]), out_mono, 44100, subtype="PCM_16")
    except Exception as e:
        Log.warn(f"HTDemucs 실패: {e}")
        for n in ("vocals", "drums", "bass", "other"): shutil.copy(mix, stems[n])
    return stems


# ═════════════════════════════════════════════════════════════════════════════
# 7. Transcription Models
# ═════════════════════════════════════════════════════════════════════════════
_bd = None
_bp = None


def _get_bd():
    global _bd
    if not HAS_BD: return None
    with _MODEL_LOCK:
        if _bd is None:
            try:
                import torch
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                _bd = PianoTranscription(device=dev)
            except Exception as e:
                Log.warn(f"ByteDance 초기화 실패: {e}"); return None
        return _bd


def _get_bp():
    global _bp
    if not HAS_BP: return None
    with _MODEL_LOCK:
        if _bp is None:
            try: _bp = BPModel(ICASSP_2022_MODEL_PATH)
            except Exception as e:
                Log.warn(f"BP 초기화 실패: {e}"); return None
        return _bp


def transcribe_bytedance(wav):
    m = _get_bd()
    if m is None: return []
    try:
        audio, _ = librosa.load(str(wav), sr=BD_SR, mono=True)
        tmp = wav.with_suffix(".bd.mid")
        m.transcribe(audio, str(tmp))
        pm = pretty_midi.PrettyMIDI(str(tmp))
        notes = [n for i in pm.instruments for n in i.notes]
        tmp.unlink(missing_ok=True)
        return [n for n in notes if (n.end - n.start) >= 0.045 and n.velocity >= 25]
    except Exception as e:
        Log.warn(f"ByteDance 실패: {e}"); return []


def transcribe_basic_pitch(wav, **kw):
    m = _get_bp()
    if m is None: return []
    try:
        _, midi, _ = bp_predict(str(wav), m, **kw)
        notes = [n for i in midi.instruments for n in i.notes]
        return [n for n in notes if (n.end - n.start) >= 0.055 and n.velocity >= 30]
    except Exception as e:
        Log.warn(f"BP 실패: {e}"); return []


def transcribe_muscriptor(wav):
    if not HAS_MUSCRIPTOR: return []
    try:
        from muscriptor import transcribe as ms_transcribe
        midi = ms_transcribe(str(wav))
        return [n for i in midi.instruments for n in i.notes]
    except Exception as e:
        Log.warn(f"MuScriptor 실패: {e}"); return []


# ═════════════════════════════════════════════════════════════════════════════
# 8. Piano Reduction & Merge
# ═════════════════════════════════════════════════════════════════════════════
def reduce_to_piano(notes, piano_range=(21, 108)):
    lo, hi = piano_range
    out = []
    for n in notes:
        p = n.pitch
        while p < lo: p += 12
        while p > hi: p -= 12
        out.append(pretty_midi.Note(velocity=min(127, max(30, n.velocity)),
                                     pitch=p, start=max(0.0, n.start),
                                     end=max(n.start + 0.03, n.end)))
    by_pitch = {}
    for n in out: by_pitch.setdefault(n.pitch, []).append(n)
    result = []
    for pitch, group in by_pitch.items():
        group.sort(key=lambda n: n.start); kept = []
        for n in group:
            merged = False
            for k in reversed(kept):
                if k.end + 0.03 < n.start: break
                if k.start < n.end - 0.03 and n.start < k.end - 0.03:
                    k.end = max(k.end, n.end); k.start = min(k.start, n.start)
                    k.velocity = max(k.velocity, n.velocity); merged = True; break
            if not merged: kept.append(n)
        result.extend(kept)
    return sorted(result, key=lambda n: n.start)


def merge_notes(bd_notes, bp_notes, tol=0.06):
    all_notes = [("bd", n) for n in bd_notes] + [("bp", n) for n in bp_notes]
    all_notes.sort(key=lambda x: x[1].start)
    merged, used = [], set()
    for i, (m1, n1) in enumerate(all_notes):
        if i in used: continue
        cluster = [(m1, n1)]; used.add(i)
        for j in range(i + 1, len(all_notes)):
            if j in used: continue
            m2, n2 = all_notes[j]
            if abs(n2.start - n1.start) <= tol and n2.pitch == n1.pitch:
                cluster.append((m2, n2)); used.add(j)
            elif n2.start - n1.start > tol + 0.1: break
        models = set(m for m, _ in cluster)
        if models == {"bp"}: continue
        starts = [n.start for _, n in cluster]; ends = [n.end for _, n in cluster]
        vels = [n.velocity for _, n in cluster]; vel = int(np.mean(vels))
        if len(models) >= 2: vel = min(127, int(vel * 1.15))
        s = float(np.median(starts)); e = float(np.percentile(ends, 75))
        if e <= s: e = s + 0.05
        merged.append(pretty_midi.Note(velocity=max(30, vel), pitch=n1.pitch, start=s, end=e))
    return sorted(merged, key=lambda n: n.start)


def cleanup_notes(notes, min_dur=0.06, min_vel=25, max_poly=10):
    notes = [n for n in notes if (n.end - n.start) >= min_dur]
    by_pitch = {}
    for n in notes: by_pitch.setdefault(n.pitch, []).append(n)
    result = []
    for pitch, group in by_pitch.items():
        group.sort(key=lambda n: n.start); kept = []
        for n in group:
            merged = False
            for k in reversed(kept):
                if k.end + 0.03 < n.start: break
                if k.start < n.end - 0.03 and n.start < k.end - 0.03:
                    k.end = max(k.end, n.end); k.start = min(k.start, n.start)
                    k.velocity = max(k.velocity, n.velocity); merged = True; break
            if not merged: kept.append(n)
        result.extend(kept)
    result = [n for n in result if n.velocity >= min_vel]
    out = []
    for n in result:
        s = max(0.0, float(n.start)); e = max(s + 0.03, float(n.end))
        out.append(pretty_midi.Note(velocity=max(1, min(127, int(n.velocity))),
                                     pitch=max(0, min(127, int(n.pitch))), start=s, end=e))
    return sorted(out, key=lambda n: n.start)


# ═════════════════════════════════════════════════════════════════════════════
# 9. MIDI + Score Generation
# ═════════════════════════════════════════════════════════════════════════════
def save_midi(notes, path, bpm=120.0):
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    inst = pretty_midi.Instrument(program=0, name="Piano")
    for n in notes:
        inst.notes.append(pretty_midi.Note(velocity=n.velocity, pitch=n.pitch,
                                            start=float(n.start), end=max(float(n.end), n.start + 0.03)))
    pm.instruments.append(inst); pm.write(str(path)); return path


def generate_score(midi_path, output_dir, title="score", bpm=120.0):
    results = {}
    if not HAS_M21: return results
    try:
        Log.stage("music21 악보 생성")
        score = converter.parse(str(midi_path), format="midi", quantizePost=False)
        try:
            k = score.analyze("key"); score.insert(0, k)
        except Exception: pass
        for p in score.parts:
            p.insert(0, meter.TimeSignature("4/4"))
            p.insert(0, clef.TrebleClef())
            p.insert(0, instrument.Piano())
        xml = output_dir / f"{title}.musicxml"
        score.write("musicxml", fp=str(xml))
        results["musicxml"] = xml
    except Exception as e:
        Log.warn(f"music21 실패: {e}"); return results
    if HAS_VEROVIO and "musicxml" in results:
        try:
            tk = verovio.toolkit()
            tk.setOptions({"pageWidth": 2100, "pageHeight": 2970,
                           "spacingStaff": 12, "spacingSystem": 16})
            tk.loadFile(str(results["musicxml"]))
            svg = output_dir / f"{title}.svg"
            Path(svg).write_text(tk.renderToSVG(1), encoding="utf-8")
            results["svg"] = svg
        except Exception as e:
            Log.warn(f"Verovio 실패: {e}")
    return results


def render_piano_mp3(midi_path, out_mp3, soundfont=None):
    if not HAS_FLUID: return None
    if soundfont is None or not soundfont.exists():
        for c in [Path("/usr/share/sounds/sf2/FluidR3_GM.sf2"),
                  Path("/usr/share/sounds/sf2/default-GM.sf2"),
                  _HERE / "soundfonts" / "FluidR3_GM.sf2"]:
            if c.exists(): soundfont = c; break
    if soundfont is None or not soundfont.exists(): return None
    try:
        from midi2audio import FluidSynth as M2AFS
        m2a = M2AFS(soundfont_path=str(soundfont))
        wav_path = out_mp3.with_suffix(".wav")
        m2a.midi_to_audio(str(midi_path), str(wav_path))
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            subprocess.run([ffmpeg, "-y", "-i", str(wav_path),
                            "-codec:a", "libmp3lame", "-b:a", "320k",
                            str(out_mp3)], capture_output=True, check=True)
            wav_path.unlink(missing_ok=True)
            return out_mp3
    except Exception as e:
        Log.warn(f"FluidSynth 실패: {e}")
    return None


# ═════════════════════════════════════════════════════════════════════════════
# 10. ★ YouTube Downloader (쿠키 + 봇 우회 지원)
# ═════════════════════════════════════════════════════════════════════════════
COOKIE_FILE = _HERE / "cookies.txt"


def _build_yt_opts(out_dir: Path, use_cookies: bool, player_clients: list) -> dict:
    tmpl = str(out_dir / "%(id)s.%(ext)s")
    opts = {
        "format": "bestaudio/best",
        "outtmpl": tmpl,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "retries": 3,
        "fragment_retries": 3,
        "socket_timeout": 30,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
    }
    if use_cookies and COOKIE_FILE.exists():
        opts["cookiefile"] = str(COOKIE_FILE)
    if player_clients:
        opts["extractor_args"] = {"youtube": {"player_client": player_clients}}
    return opts


def _try_download(url: str, out_dir: Path, opts: dict) -> Optional[dict]:
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return info
    except Exception as e:
        Log.warn(f"YouTube 시도 실패: {str(e)[:200]}")
        return None


def download_youtube(url, out_dir, max_sec=300):
    if not HAS_YTDLP:
        raise RuntimeError("yt-dlp가 설치되지 않았습니다")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 시도 순서: 쿠키+tv/mweb → 쿠키만 → tv/mweb만 → 기본
    attempts = []
    if COOKIE_FILE.exists():
        attempts.append(("쿠키 + tv/mweb", True, ["tv", "mweb"]))
        attempts.append(("쿠키만", True, None))
    attempts.append(("tv/mweb 클라이언트", False, ["tv", "mweb"]))
    attempts.append(("기본 설정", False, None))

    info = None
    used = None
    last_err = "알 수 없는 오류"
    for label, use_cookies, clients in attempts:
        Log.info(f"  YouTube 다운로드 시도: {label}")
        opts = _build_yt_opts(out_dir, use_cookies, clients)
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
            used = label
            break
        except Exception as e:
            last_err = str(e)
            Log.warn(f"  [{label}] 실패: {last_err[:150]}")

    if info is None:
        # 쿠키 파일이 없을 때 상세 안내
        hint = ""
        if not COOKIE_FILE.exists():
            hint = ("\n\n💡 해결 방법: 브라우저에서 YouTube 로그인 후 cookies.txt 파일을 "
                    f"내보내서 {COOKIE_FILE} 경로에 넣어주세요. "
                    "(Chrome 확장: 'Get cookies.txt LOCALLY')")
        raise RuntimeError(f"YouTube 봇 감지로 다운로드 실패.{hint}\n\n마지막 오류: {last_err[:300]}")

    Log.info(f"  YouTube 다운로드 성공 ({used})")

    video_id = info.get("id", "audio")
    title = info.get("title", video_id)
    uploader = info.get("uploader", "")
    duration = info.get("duration", 0)

    wav_path = out_dir / f"{video_id}.wav"
    if not wav_path.exists():
        for c in out_dir.glob(f"{video_id}.*"):
            if c.suffix in (".wav", ".m4a", ".webm", ".mp3", ".opus"):
                wav_path = c; break
    if not wav_path.exists():
        raise RuntimeError("다운로드된 오디오 파일을 찾을 수 없습니다")

    if max_sec and duration and duration > max_sec:
        y, sr = librosa.load(str(wav_path), sr=22050, mono=True)
        y = y[: max_sec * sr]
        sf.write(str(wav_path), y, sr)
        Log.info(f"  {max_sec}초로 잘라냄")

    return {"path": wav_path, "title": title, "uploader": uploader,
            "duration": duration, "video_id": video_id, "method": used}


# ═════════════════════════════════════════════════════════════════════════════
# 11. Full Pipeline
# ═════════════════════════════════════════════════════════════════════════════
def run_pipeline(audio_path, output_dir, mode="auto", progress_cb=None):
    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    work = output_dir / "_work"; work.mkdir(exist_ok=True)

    def stage(msg):
        if progress_cb: progress_cb(msg)
        Log.stage(msg)

    stage("곡 분석")
    y = load_audio(audio_path, sr=22050, max_sec=300)
    an = GenreAnalyzer(y).extract()
    genre = an.classify_genre(); has_piano = an.has_piano()

    stage("스템 분리")
    stems = separate_stems(audio_path, work)

    if mode == "auto":
        mode = "piano" if has_piano else "universal"

    all_notes = []
    if mode == "piano":
        stage("피아노 전사")
        bd = transcribe_bytedance(stems["other"])
        bp = transcribe_basic_pitch(stems["other"], minimum_note_length=70,
            onset_threshold=0.5, frame_threshold=0.35, minimum_frequency=60,
            maximum_frequency=4000, melodia_trick=True)
        all_notes = merge_notes(bd, bp)
    elif mode == "universal":
        stage("다악기 전사")
        multi = transcribe_muscriptor(stems["mix"])
        if not multi:
            for sn in ("other", "bass", "vocals"):
                if stems[sn].exists():
                    all_notes.extend(transcribe_basic_pitch(stems[sn],
                        minimum_note_length=70, onset_threshold=0.5,
                        frame_threshold=0.35, minimum_frequency=40,
                        maximum_frequency=4000, melodia_trick=True))
        else:
            all_notes = multi
    if not all_notes: raise RuntimeError("음이 검출되지 않았습니다")

    if mode in ("universal", "cover"):
        stage("피아노 Reduction")
        all_notes = reduce_to_piano(all_notes)

    stage("비트 추적")
    env = librosa.onset.onset_strength(y=y, sr=22050, hop_length=512)
    tempo, beats = librosa.beat.beat_track(onset_envelope=env, sr=22050, hop_length=512,
        units="time", start_bpm=150, tightness=100, trim=False)
    bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    if bpm < 60 or bpm > 240: bpm = 120.0

    stage("악보 품질 개선")
    all_notes = quantize_beat_aware(all_notes, beats, bpm, grid_div=16)
    all_notes = add_articulations(all_notes, bpm)

    stage("성부 분리")
    voices = separate_voices(all_notes, max_voices=3)
    treble, bass = assign_to_grand_staff(voices, split_pitch=60)

    stage("노트 정리")
    notes = cleanup_notes(all_notes, min_dur=0.06, min_vel=25, max_poly=10)

    stage("MIDI 저장")
    name = audio_path.stem
    mid = output_dir / f"{name}.mid"
    save_midi(notes, mid, bpm=bpm)

    stage("악보 생성")
    scores = generate_score(mid, output_dir, title=name, bpm=bpm)

    stage("피아노 MP3 렌더링")
    mp3 = render_piano_mp3(mid, output_dir / f"{name}_piano.mp3")

    meta = {
        "filename": name, "mode": mode, "genre": genre, "has_piano": has_piano,
        "bpm": round(bpm, 1), "duration_sec": round(len(y) / 22050, 1),
        "notes": len(notes), "voices": len(voices),
        "treble_notes": len(treble), "bass_notes": len(bass),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    meta_path = output_dir / f"{name}_info.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    out = {"midi": mid, "meta": meta_path}
    if mp3: out["piano_mp3"] = mp3
    out.update(scores)
    stage(f"완료 ({time.time() - t0:.1f}초)")
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 12. Flask App
# ═════════════════════════════════════════════════════════════════════════════
app = Flask(__name__)
CORS(app)
UPLOAD = Path("uploads"); UPLOAD.mkdir(exist_ok=True)
OUTPUT = Path("outputs"); OUTPUT.mkdir(exist_ok=True)
SHARED_DIR = Path("shared"); SHARED_DIR.mkdir(exist_ok=True)
SHARED_FILE = SHARED_DIR / "index.json"

JOBS: Dict[str, dict] = {}
LOCK = threading.Lock()
SHARED: Dict[str, dict] = {}


def _load_shared():
    global SHARED
    if SHARED_FILE.exists():
        try: SHARED = json.loads(SHARED_FILE.read_text(encoding="utf-8"))
        except Exception: SHARED = {}


def _save_shared():
    SHARED_FILE.write_text(json.dumps(SHARED, ensure_ascii=False, indent=2), encoding="utf-8")


_load_shared()


def _worker(job_id, audio_path, mode):
    try:
        def cb(msg):
            with LOCK:
                if job_id in JOBS: JOBS[job_id]["message"] = msg
        with LOCK:
            JOBS[job_id] = {"status": "running", "message": "시작...", "result": None}
        out = run_pipeline(audio_path, OUTPUT / job_id, mode=mode, progress_cb=cb)
        files = {k: f"/download/{job_id}/{p.name}" for k, p in out.items() if p.exists()}
        with LOCK:
            JOBS[job_id] = {"status": "done", "message": "완료", "result": files, "finished_at": time.time()}
    except Exception as e:
        import traceback; traceback.print_exc()
        with LOCK:
            JOBS[job_id] = {"status": "error", "message": str(e), "result": None, "finished_at": time.time()}


@app.route("/")
def index(): return render_template_string(HTML)


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files: return jsonify({"error": "파일 없음"}), 400
    f = request.files["file"]
    if not f.filename: return jsonify({"error": "파일명 없음"}), 400
    ext = Path(f.filename).suffix.lower()
    if ext not in (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"):
        return jsonify({"error": f"지원하지 않는 형식: {ext}"}), 400
    jid = uuid.uuid4().hex[:12]
    p = UPLOAD / f"{jid}{ext}"
    f.save(str(p))
    mode = request.form.get("mode", "auto")
    threading.Thread(target=_worker, args=(jid, p, mode), daemon=True).start()
    return jsonify({"job_id": jid})


@app.route("/youtube", methods=["POST"])
def youtube():
    data = request.get_json() or {}
    url = (data.get("url") or "").strip()
    mode = data.get("mode", "auto")
    if not url: return jsonify({"error": "URL이 비어있습니다"}), 400
    if not re.match(r"^https?://(www\.)?(youtube\.com|youtu\.be|m\.youtube\.com|music\.youtube\.com)/", url):
        return jsonify({"error": "유효한 YouTube URL이 아닙니다"}), 400
    if not HAS_YTDLP:
        return jsonify({"error": "yt-dlp가 설치되지 않았습니다"}), 500
    jid = uuid.uuid4().hex[:12]
    out_dir = UPLOAD / jid
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        info = download_youtube(url, out_dir, max_sec=300)
    except Exception as e:
        return jsonify({"error": f"다운로드 실패: {e}"}), 500

    def worker_with_meta():
        try:
            def cb(msg):
                with LOCK:
                    if jid in JOBS: JOBS[jid]["message"] = msg
            with LOCK:
                JOBS[jid] = {"status": "running", "message": "변환 준비 중...",
                             "result": None, "yt_title": info["title"],
                             "yt_uploader": info["uploader"]}
            out = run_pipeline(info["path"], OUTPUT / jid, mode=mode, progress_cb=cb)
            files = {k: f"/download/{jid}/{p.name}" for k, p in out.items() if p.exists()}
            with LOCK:
                JOBS[jid] = {"status": "done", "message": "완료", "result": files,
                             "finished_at": time.time(),
                             "yt_title": info["title"], "yt_uploader": info["uploader"]}
        except Exception as e:
            import traceback; traceback.print_exc()
            with LOCK:
                JOBS[jid] = {"status": "error", "message": str(e), "result": None, "finished_at": time.time()}

    threading.Thread(target=worker_with_meta, daemon=True).start()
    return jsonify({"job_id": jid, "title": info["title"], "uploader": info["uploader"], "method": info.get("method", "")})


@app.route("/status/<jid>")
def status(jid):
    with LOCK:
        j = JOBS.get(jid)
        if j is not None: j = dict(j)
    if not j: return jsonify({"error": "작업 없음"}), 404
    return jsonify(j)


@app.route("/download/<jid>/<fn>")
def download(jid, fn):
    safe = Path(fn).name
    p = OUTPUT / jid / safe
    if not p.exists(): return jsonify({"error": "파일 없음"}), 404
    return send_file(str(p), as_attachment=True)


# ═════════════════════════════════════════════════════════════════════════════
# 13. Share / Community Endpoints
# ═════════════════════════════════════════════════════════════════════════════
@app.route("/api/share", methods=["POST"])
def api_share():
    data = request.get_json() or {}
    job_id = data.get("job_id", "").strip()
    title = (data.get("title") or "Untitled").strip()[:100]
    author = (data.get("author") or "익명").strip()[:50]
    if not job_id: return jsonify({"error": "job_id 필요"}), 400
    job_dir = OUTPUT / job_id
    if not job_dir.exists(): return jsonify({"error": "작업 결과가 없습니다"}), 404

    sid = uuid.uuid4().hex[:10]
    dst = SHARED_DIR / sid
    dst.mkdir(parents=True, exist_ok=True)

    files = {}
    meta = None
    for p in job_dir.iterdir():
        if p.is_file() and p.suffix.lower() in (".mid", ".musicxml", ".svg", ".pdf", ".mp3", ".json"):
            shutil.copy(p, dst / p.name)
            key = p.suffix.lower().lstrip(".")
            if key == "json": meta = p.name
            else: files[key] = p.name

    meta_info = {}
    if meta:
        try: meta_info = json.loads((dst / meta).read_text(encoding="utf-8"))
        except Exception: pass

    SHARED[sid] = {
        "id": sid, "title": title, "author": author,
        "date": time.time(), "views": 0,
        "files": files, "meta": meta_info,
        "has_svg": "svg" in files, "has_mp3": "mp3" in files,
    }
    _save_shared()
    return jsonify({"ok": True, "share_id": sid, "url": f"/s/{sid}"})


@app.route("/api/community")
def api_community():
    items = sorted(SHARED.values(), key=lambda x: x.get("date", 0), reverse=True)
    return jsonify({"items": items, "total": len(items)})


@app.route("/api/unshare", methods=["POST"])
def api_unshare():
    data = request.get_json() or {}
    sid = (data.get("share_id") or "").strip()
    if sid in SHARED:
        shutil.rmtree(SHARED_DIR / sid, ignore_errors=True)
        del SHARED[sid]; _save_shared()
        return jsonify({"ok": True})
    return jsonify({"error": "없음"}), 404


@app.route("/shared/<sid>/<fn>")
def shared_file(sid, fn):
    safe = Path(fn).name
    p = SHARED_DIR / sid / safe
    if not p.exists(): return jsonify({"error": "파일 없음"}), 404
    return send_file(str(p), as_attachment=False)


@app.route("/shared_dl/<sid>/<fn>")
def shared_dl(sid, fn):
    safe = Path(fn).name
    p = SHARED_DIR / sid / safe
    if not p.exists(): return jsonify({"error": "파일 없음"}), 404
    return send_file(str(p), as_attachment=True)


@app.route("/s/<sid>")
def shared_view(sid):
    item = SHARED.get(sid)
    if not item: abort(404)
    item["views"] = item.get("views", 0) + 1
    _save_shared()
    return render_template_string(SHARED_TEMPLATE, item=item, sid=sid)


# ═════════════════════════════════════════════════════════════════════════════
# 14. Main HTML
# ═════════════════════════════════════════════════════════════════════════════
HTML = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PianoForge — AI Audio to Score Platform</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css">
<style>
:root{--bg:#08080a;--bg-elev:#0f0f12;--surface:rgba(23,23,28,0.65);--surface-solid:#17171c;
--border:rgba(255,255,255,0.07);--border-strong:rgba(255,255,255,0.12);--text:#fafafa;--text-dim:#a1a1aa;
--text-mute:#52525b;--brand:#6366f1;--brand-glow:rgba(99,102,241,0.35);
--grad:linear-gradient(135deg,#6366f1 0%,#a855f7 50%,#ec4899 100%);
--grad-subtle:linear-gradient(135deg,rgba(99,102,241,0.15),rgba(168,85,247,0.1));
--green:#10b981;--red:#ef4444;--font:"Pretendard Variable",-apple-system,BlinkMacSystemFont,system-ui,sans-serif;
--mono:ui-monospace,"SF Mono",Menlo,monospace}
*{box-sizing:border-box;margin:0;padding:0}html,body{height:100%}
body{font-family:var(--font);background:var(--bg);color:var(--text);font-size:14px;line-height:1.5;
-webkit-font-smoothing:antialiased;overflow-x:hidden}
body::before{content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
background:radial-gradient(1000px 600px at 10% -10%,rgba(99,102,241,0.10),transparent 60%),
radial-gradient(800px 500px at 90% 0%,rgba(168,85,247,0.08),transparent 60%)}
::-webkit-scrollbar{width:8px;height:8px}::-webkit-scrollbar-thumb{background:rgba(255,255,255,0.06);border-radius:4px}
button{font-family:inherit;cursor:pointer;border:none;background:none;color:inherit}
input,select,textarea{font-family:inherit}
.view{display:none;min-height:100vh;position:relative;z-index:1}.view.active{display:block}
.view-flex{display:none;min-height:100vh}.view-flex.active{display:flex}
.landing{display:flex;flex-direction:column;min-height:100vh}
.nav{display:flex;align-items:center;justify-content:space-between;padding:20px 40px;
position:sticky;top:0;z-index:50;backdrop-filter:blur(20px);background:rgba(8,8,10,0.7);border-bottom:1px solid var(--border)}
.nav-brand{display:flex;align-items:center;gap:10px;font-weight:700;font-size:15px;cursor:pointer}
.logo-mark{width:28px;height:28px;border-radius:8px;background:var(--grad);display:grid;place-items:center;
font-size:14px;box-shadow:0 0 20px var(--brand-glow);color:#fff}
.nav-links{display:flex;gap:28px}
.nav-links a{color:var(--text-dim);text-decoration:none;font-size:13px;font-weight:500;transition:color .2s;cursor:pointer}
.nav-links a:hover{color:var(--text)}
.nav-actions{display:flex;gap:10px;align-items:center}
.btn-ghost{padding:8px 16px;border-radius:8px;font-size:13px;font-weight:600;color:var(--text-dim);transition:all .2s}
.btn-ghost:hover{color:var(--text);background:rgba(255,255,255,0.04)}
.btn-solid{padding:8px 16px;border-radius:8px;font-size:13px;font-weight:600;background:var(--text);color:#000;transition:all .2s}
.btn-solid:hover{opacity:0.9;transform:translateY(-1px)}
.hero{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:80px 40px 100px;text-align:center}
.badge{display:inline-flex;align-items:center;gap:8px;padding:6px 12px;border-radius:100px;
background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);font-size:12px;color:#a5b4fc;
font-weight:600;margin-bottom:28px;animation:fadeUp 0.8s ease}
.badge .dot{width:6px;height:6px;border-radius:50%;background:#6366f1;box-shadow:0 0 8px #6366f1}
.hero h1{font-size:clamp(36px,6vw,64px);font-weight:800;letter-spacing:-0.04em;line-height:1.05;
max-width:900px;margin-bottom:20px;background:linear-gradient(180deg,#fff 30%,#a1a1aa 100%);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;animation:fadeUp 0.8s ease 0.1s both}
.hero p{font-size:17px;color:var(--text-dim);max-width:600px;margin-bottom:36px;line-height:1.6;animation:fadeUp 0.8s ease 0.2s both}
.hero-cta{display:flex;gap:12px;animation:fadeUp 0.8s ease 0.3s both;flex-wrap:wrap;justify-content:center}
.btn-hero{padding:14px 28px;border-radius:100px;font-size:14px;font-weight:700;
transition:all 0.3s cubic-bezier(0.16,1,0.3,1);display:inline-flex;align-items:center;gap:8px}
.btn-hero.primary{background:var(--grad);color:#fff;box-shadow:0 8px 32px -8px var(--brand-glow)}
.btn-hero.primary:hover{transform:translateY(-2px);box-shadow:0 16px 40px -8px var(--brand-glow)}
.btn-hero.secondary{background:rgba(255,255,255,0.04);border:1px solid var(--border-strong);color:var(--text)}
.features{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px;
max-width:1100px;width:100%;margin-top:80px;animation:fadeUp 0.8s ease 0.4s both}
.feat{padding:24px;border-radius:16px;background:var(--surface);border:1px solid var(--border);
backdrop-filter:blur(12px);text-align:left;transition:all 0.3s cubic-bezier(0.16,1,0.3,1)}
.feat:hover{border-color:var(--border-strong);transform:translateY(-4px)}
.feat-icon{width:40px;height:40px;border-radius:8px;margin-bottom:16px;background:var(--grad-subtle);
border:1px solid rgba(99,102,241,0.2);display:grid;place-items:center;color:#a5b4fc}
.feat h3{font-size:15px;font-weight:700;margin-bottom:6px}
.feat p{font-size:13px;color:var(--text-dim);line-height:1.6}
@keyframes fadeUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
.modal-backdrop{position:fixed;inset:0;z-index:1000;background:rgba(0,0,0,0.7);
backdrop-filter:blur(8px);display:none;align-items:center;justify-content:center;padding:24px}
.modal-backdrop.on{display:flex}
.modal{width:100%;max-width:420px;background:var(--surface-solid);border:1px solid var(--border-strong);
border-radius:24px;padding:32px;animation:modalIn 0.3s cubic-bezier(0.16,1,0.3,1);position:relative}
@keyframes modalIn{from{opacity:0;transform:scale(0.95) translateY(20px)}to{opacity:1;transform:scale(1) translateY(0)}}
.modal-close{position:absolute;top:16px;right:16px;width:32px;height:32px;border-radius:8px;
display:grid;place-items:center;color:var(--text-mute);transition:all .2s}
.modal-close:hover{background:rgba(255,255,255,0.06);color:var(--text)}
.modal h2{font-size:22px;font-weight:800;margin-bottom:6px;letter-spacing:-0.02em}
.modal .sub{font-size:13px;color:var(--text-dim);margin-bottom:24px}
.tabs{display:flex;gap:4px;margin-bottom:24px;padding:4px;background:rgba(0,0,0,0.3);border-radius:10px}
.tab{flex:1;padding:8px;text-align:center;font-size:13px;font-weight:600;color:var(--text-dim);
border-radius:8px;transition:all .2s;cursor:pointer}
.tab.active{background:rgba(255,255,255,0.06);color:var(--text)}
.field{margin-bottom:14px}
.field label{display:block;font-size:12px;font-weight:600;color:var(--text-dim);margin-bottom:6px}
.field input,.field textarea{width:100%;padding:12px 14px;border-radius:10px;font-size:14px;
background:rgba(0,0,0,0.3);border:1px solid var(--border);color:var(--text);transition:all .2s;outline:none}
.field input:focus,.field textarea:focus{border-color:var(--brand);box-shadow:0 0 0 3px var(--brand-glow)}
.field input::placeholder,.field textarea::placeholder{color:var(--text-mute)}
.field textarea{resize:vertical;min-height:80px}
.btn-submit{width:100%;padding:12px;border-radius:10px;font-size:14px;font-weight:700;
background:var(--grad);color:#fff;margin-top:8px;transition:all 0.3s;box-shadow:0 8px 24px -8px var(--brand-glow)}
.btn-submit:hover{transform:translateY(-1px)}.btn-submit:disabled{opacity:0.5;cursor:not-allowed;transform:none}
.auth-switch{text-align:center;margin-top:20px;font-size:13px;color:var(--text-dim)}
.auth-switch a{color:#a5b4fc;text-decoration:none;font-weight:600;cursor:pointer}
.sidebar{width:260px;flex-shrink:0;background:var(--bg-elev);border-right:1px solid var(--border);
display:flex;flex-direction:column;position:sticky;top:0;height:100vh}
.sidebar-head{padding:20px;display:flex;align-items:center;gap:10px;border-bottom:1px solid var(--border)}
.sidebar-nav{flex:1;padding:12px;overflow-y:auto}
.nav-section{margin-bottom:20px}
.nav-section-title{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;
color:var(--text-mute);padding:8px 12px 6px}
.nav-item{display:flex;align-items:center;gap:10px;padding:9px 12px;border-radius:8px;font-size:13px;
font-weight:500;color:var(--text-dim);cursor:pointer;transition:all .15s;margin-bottom:2px;user-select:none}
.nav-item:hover{background:rgba(255,255,255,0.04);color:var(--text)}
.nav-item.active{background:rgba(99,102,241,0.1);color:#a5b4fc}
.nav-item svg{width:16px;height:16px;flex-shrink:0}
.nav-item .badge-count{margin-left:auto;font-size:10px;padding:2px 6px;border-radius:100px;
background:rgba(99,102,241,0.15);color:#a5b4fc;font-weight:700}
.sidebar-foot{padding:12px;border-top:1px solid var(--border)}
.user-card{display:flex;align-items:center;gap:10px;padding:8px;border-radius:10px;cursor:pointer;transition:all .15s}
.user-card:hover{background:rgba(255,255,255,0.04)}
.avatar{width:32px;height:32px;border-radius:50%;background:var(--grad);display:grid;place-items:center;
font-size:12px;font-weight:700;color:#fff;flex-shrink:0}
.user-info{flex:1;min-width:0}
.user-name{font-size:13px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.user-plan{font-size:10px;color:var(--text-mute)}
.main{flex:1;display:flex;flex-direction:column;min-width:0}
.topbar{display:flex;align-items:center;justify-content:space-between;padding:16px 32px;
border-bottom:1px solid var(--border);position:sticky;top:0;z-index:40;
backdrop-filter:blur(20px);background:rgba(8,8,10,0.7)}
.topbar-actions{display:flex;gap:8px;align-items:center}
.icon-btn{width:36px;height:36px;border-radius:8px;display:grid;place-items:center;
color:var(--text-dim);transition:all .15s}
.icon-btn:hover{background:rgba(255,255,255,0.05);color:var(--text)}
.content{padding:32px;max-width:1200px;width:100%}
.page-title{font-size:26px;font-weight:800;letter-spacing:-0.02em;margin-bottom:6px}
.page-sub{font-size:14px;color:var(--text-dim);margin-bottom:28px}
.stats-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:32px}
.stat{padding:18px;border-radius:12px;background:var(--surface);border:1px solid var(--border);
backdrop-filter:blur(12px);transition:all 0.3s}
.stat:hover{border-color:var(--border-strong);transform:translateY(-2px)}
.stat-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px}
.stat-label{font-size:11px;font-weight:600;color:var(--text-mute);text-transform:uppercase;letter-spacing:0.06em}
.stat-value{font-size:28px;font-weight:800;letter-spacing:-0.02em}
.card{background:var(--surface);border:1px solid var(--border);border-radius:16px;
backdrop-filter:blur(12px);padding:24px;margin-bottom:20px}
.card-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:18px}
.card-title{font-size:15px;font-weight:700}
.card-desc{font-size:12px;color:var(--text-dim);margin-top:2px}
.source-tabs{display:flex;gap:4px;padding:4px;background:rgba(0,0,0,0.3);border-radius:10px;margin-bottom:16px}
.source-tab{flex:1;padding:10px;text-align:center;font-size:13px;font-weight:600;color:var(--text-dim);
border-radius:8px;cursor:pointer;transition:all .2s}
.source-tab.active{background:rgba(255,255,255,0.06);color:var(--text)}
.source-pane{display:none}.source-pane.active{display:block}
.dropzone{display:block;position:relative;border:1.5px dashed var(--border-strong);
border-radius:16px;padding:44px 24px;text-align:center;cursor:pointer;
transition:all 0.3s cubic-bezier(0.16,1,0.3,1);background:rgba(0,0,0,0.2);overflow:hidden}
.dropzone:hover{border-color:var(--brand);background:rgba(99,102,241,0.05)}
.dropzone.active{border-color:var(--brand);background:rgba(99,102,241,0.1);transform:scale(0.99)}
.dropzone.has-file{border-style:solid;border-color:var(--green);background:rgba(16,185,129,0.05)}
.dropzone input{position:absolute;inset:0;opacity:0;cursor:pointer}
.dz-icon{width:56px;height:56px;margin:0 auto 14px;border-radius:16px;background:var(--grad-subtle);
border:1px solid rgba(99,102,241,0.2);display:grid;place-items:center;color:#a5b4fc;transition:all .3s}
.dz-main{font-size:15px;font-weight:600;margin-bottom:4px}
.dz-hint{font-size:12px;color:var(--text-mute)}
.yt-row{display:flex;gap:10px}
.yt-row input{flex:1;padding:14px;border-radius:10px;font-size:14px;background:rgba(0,0,0,0.3);
border:1px solid var(--border);color:var(--text);outline:none;transition:all .2s}
.yt-row input:focus{border-color:var(--brand);box-shadow:0 0 0 3px var(--brand-glow)}
.yt-row button{padding:14px 22px;border-radius:10px;font-size:13px;font-weight:700;
background:var(--grad);color:#fff;box-shadow:0 8px 24px -8px var(--brand-glow);
transition:all .2s;white-space:nowrap}
.yt-row button:disabled{opacity:0.5;cursor:not-allowed}
.yt-hint{font-size:11px;color:var(--text-mute);margin-top:8px}
.mode-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:16px}
.mode-card{padding:14px 12px;border-radius:12px;cursor:pointer;border:1px solid var(--border);
background:rgba(0,0,0,0.2);transition:all 0.2s cubic-bezier(0.16,1,0.3,1);text-align:center}
.mode-card:hover{border-color:var(--border-strong);background:rgba(255,255,255,0.03)}
.mode-card.active{border-color:var(--brand);background:rgba(99,102,241,0.08);box-shadow:0 0 0 1px var(--brand)}
.mode-card .m-name{font-size:13px;font-weight:700;margin-bottom:2px}
.mode-card.active .m-name{color:#c7d2fe}
.mode-card .m-desc{font-size:10px;color:var(--text-mute)}
.btn-primary{width:100%;padding:14px;border-radius:12px;font-size:14px;font-weight:700;
background:var(--grad);color:#fff;margin-top:20px;transition:all 0.3s;
box-shadow:0 8px 24px -8px var(--brand-glow);display:flex;align-items:center;justify-content:center;gap:8px}
.btn-primary:hover:not(:disabled){transform:translateY(-2px)}
.btn-primary:disabled{background:rgba(255,255,255,0.04);color:var(--text-mute);box-shadow:none;cursor:not-allowed}
.progress-card{display:none;margin-top:20px;padding:20px;border-radius:12px;
background:rgba(99,102,241,0.05);border:1px solid rgba(99,102,241,0.15)}
.progress-card.on{display:block}
.prog-bar-bg{height:4px;background:rgba(0,0,0,0.4);border-radius:100px;overflow:hidden;margin-bottom:14px}
.prog-bar-fill{height:100%;background:var(--grad);border-radius:100px;
animation:indeterminate 1.5s ease-in-out infinite;width:40%}
@keyframes indeterminate{0%{transform:translateX(-100%)}100%{transform:translateX(400%)}}
.prog-status{font-size:13px;color:var(--text-dim);display:flex;align-items:center;gap:10px}
.spinner{width:14px;height:14px;border:2px solid rgba(255,255,255,0.1);border-top-color:var(--brand);
border-radius:50%;animation:spin 0.7s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.result-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px;margin-top:16px}
.result-item{display:flex;align-items:center;gap:12px;padding:14px;border-radius:12px;
text-decoration:none;color:inherit;background:rgba(0,0,0,0.2);border:1px solid var(--border);transition:all 0.2s}
.result-item:hover{border-color:var(--brand);background:rgba(99,102,241,0.06);transform:translateX(2px)}
.result-item .ri-icon{width:36px;height:36px;border-radius:10px;background:var(--grad-subtle);
display:grid;place-items:center;color:#a5b4fc;font-size:16px;flex-shrink:0}
.result-item .ri-name{font-size:13px;font-weight:600}
.result-item .ri-desc{font-size:11px;color:var(--text-mute)}
.result-item .ri-arrow{margin-left:auto;color:var(--text-mute);font-size:16px}
.share-bar{margin-top:16px;padding:16px;border-radius:12px;background:rgba(16,185,129,0.05);
border:1px solid rgba(16,185,129,0.2);display:flex;align-items:center;justify-content:space-between;gap:16px}
.share-bar .sb-title{font-size:13px;font-weight:700;color:#6ee7b7}
.share-bar .sb-desc{font-size:11px;color:var(--text-dim);margin-top:2px}
.share-bar button{padding:10px 20px;border-radius:8px;font-size:13px;font-weight:700;
background:var(--green);color:#fff;transition:all .2s}
.share-bar button:hover{transform:translateY(-1px)}
.share-bar button:disabled{opacity:0.5;cursor:not-allowed}
.community-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px}
.community-card{padding:18px;border-radius:14px;background:var(--surface);border:1px solid var(--border);
cursor:pointer;transition:all 0.3s;text-decoration:none;color:inherit;display:block}
.community-card:hover{border-color:var(--brand);background:var(--surface-hover);transform:translateY(-3px)}
.cc-top{display:flex;align-items:center;gap:10px;margin-bottom:12px}
.cc-avatar{width:36px;height:36px;border-radius:10px;background:var(--grad);display:grid;place-items:center;
font-size:14px;font-weight:700;color:#fff;flex-shrink:0}
.cc-author{font-size:12px;color:var(--text-dim);font-weight:600}
.cc-date{font-size:10px;color:var(--text-mute);margin-top:2px}
.cc-title{font-size:14px;font-weight:700;margin-bottom:10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.cc-tags{display:flex;gap:6px;flex-wrap:wrap}
.cc-tag{font-size:10px;padding:3px 8px;border-radius:100px;background:rgba(99,102,241,0.1);
color:#a5b4fc;font-weight:600}
.cc-meta{display:flex;align-items:center;justify-content:space-between;margin-top:12px;
padding-top:12px;border-top:1px solid var(--border);font-size:11px;color:var(--text-mute)}
.empty{text-align:center;padding:40px 20px;color:var(--text-mute);font-size:13px}
.empty svg{width:32px;height:32px;margin-bottom:12px;opacity:0.4}
.alert{padding:12px 14px;border-radius:10px;font-size:12px;display:flex;gap:10px;
align-items:flex-start;line-height:1.5}
.alert.warn{background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2);color:#fbbf24}
.alert.err{background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.2);color:#fca5a5;
margin-top:16px;display:none}
.alert.err.on{display:flex}
.alert.info{background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);color:#a5b4fc;
margin-bottom:16px}
.toasts{position:fixed;bottom:24px;right:24px;z-index:2000;display:flex;flex-direction:column;gap:8px}
.toast{padding:12px 16px;border-radius:10px;background:var(--surface-solid);border:1px solid var(--border-strong);
box-shadow:0 12px 32px rgba(0,0,0,0.5);display:flex;align-items:center;gap:10px;font-size:13px;
font-weight:500;min-width:240px;animation:toastIn 0.3s cubic-bezier(0.16,1,0.3,1)}
.toast.success{border-color:rgba(16,185,129,0.3)}.toast.error{border-color:rgba(239,68,68,0.3)}
.toast .dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.toast.success .dot{background:var(--green);box-shadow:0 0 8px var(--green)}
.toast.error .dot{background:var(--red);box-shadow:0 0 8px var(--red)}
.toast.info .dot{background:var(--brand);box-shadow:0 0 8px var(--brand)}
@keyframes toastIn{from{opacity:0;transform:translateX(20px)}to{opacity:1;transform:translateX(0)}}
.score-container{background:#fff;border-radius:12px;padding:20px;overflow-x:auto;display:none;margin-top:16px}
.score-container.on{display:block}.score-container svg{display:block;max-width:100%;height:auto}
.settings-row{display:flex;align-items:center;justify-content:space-between;padding:16px 0;border-bottom:1px solid var(--border)}
.settings-row:last-child{border-bottom:none}
.settings-label{font-size:13px;font-weight:600}
.settings-desc{font-size:12px;color:var(--text-dim);margin-top:2px}
.toggle{width:40px;height:22px;border-radius:100px;background:rgba(255,255,255,0.08);position:relative;cursor:pointer;transition:all .2s}
.toggle.on{background:var(--brand)}
.toggle::after{content:'';position:absolute;top:2px;left:2px;width:18px;height:18px;border-radius:50%;background:#fff;transition:all .2s}
.toggle.on::after{left:20px}
@media (max-width:900px){.sidebar{position:fixed;z-index:100;transform:translateX(-100%);transition:transform .3s}
.sidebar.open{transform:translateX(0)}.mode-grid{grid-template-columns:1fr}.nav-links{display:none}
.content{padding:20px}.community-grid{grid-template-columns:1fr}}
</style>
</head>
<body>

<div id="view-landing" class="view active">
  <div class="landing">
    <nav class="nav">
      <div class="nav-brand" onclick="location.reload()"><div class="logo-mark">♪</div> PianoForge</div>
      <div class="nav-links">
        <a onclick="openAuth('login')">로그인</a>
        <a onclick="if(State.user)navigate('community');else toast('로그인이 필요합니다','info')">커뮤니티</a>
      </div>
      <div class="nav-actions">
        <button class="btn-ghost" onclick="openAuth('login')">로그인</button>
        <button class="btn-solid" onclick="openAuth('signup')">무료로 시작</button>
      </div>
    </nav>
    <section class="hero">
      <div class="badge"><span class="dot"></span> AI Transcription + Community Sharing</div>
      <h1>음악을 악보로,<br>그리고 세상과 공유하세요.</h1>
      <p>파일 업로드 또는 YouTube 링크만 붙여넣으면 정밀한 피아노 MIDI와 악보로 변환됩니다. 마음에 들면 공유 버튼 한 번으로 누구나 볼 수 있게 게시하세요.</p>
      <div class="hero-cta">
        <button class="btn-hero primary" onclick="openAuth('signup')">
          무료로 시작하기
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
        </button>
        <button class="btn-hero secondary" onclick="openAuth('login')">로그인</button>
      </div>
      <div class="features">
        <div class="feat">
          <div class="feat-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg></div>
          <h3>YouTube 링크 변환</h3>
          <p>URL만 붙여넣으면 자동으로 오디오 추출 → 악보 변환까지 한 번에 처리합니다.</p>
        </div>
        <div class="feat">
          <div class="feat-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg></div>
          <h3>커뮤니티 공유</h3>
          <p>공유 버튼으로 나만의 악보를 사이트에 게시하고, 다른 사람들의 작품도 감상하세요.</p>
        </div>
        <div class="feat">
          <div class="feat-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg></div>
          <h3>전문가급 품질</h3>
          <p>성부 분리, 그랜드 스태프, 비트 양자화까지 자동 적용된 악보를 받습니다.</p>
        </div>
        <div class="feat">
          <div class="feat-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg></div>
          <h3>다양한 포맷</h3>
          <p>MIDI, MusicXML, SVG 악보, MP3까지 원하는 형태로 즉시 다운로드.</p>
        </div>
      </div>
    </section>
  </div>
</div>

<div id="view-app" class="view-flex">
  <aside class="sidebar" id="sidebar">
    <div class="sidebar-head">
      <div class="logo-mark">♪</div>
      <div style="font-weight:700;font-size:14px;">PianoForge</div>
    </div>
    <nav class="sidebar-nav">
      <div class="nav-section">
        <div class="nav-section-title">Workspace</div>
        <div class="nav-item active" data-page="dashboard" onclick="navigate('dashboard')">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>
          대시보드
        </div>
        <div class="nav-item" data-page="convert" onclick="navigate('convert')">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
          새 변환
        </div>
        <div class="nav-item" data-page="history" onclick="navigate('history')">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
          내 히스토리
          <span class="badge-count" id="hist-count">0</span>
        </div>
      </div>
      <div class="nav-section">
        <div class="nav-section-title">Community</div>
        <div class="nav-item" data-page="community" onclick="navigate('community')">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg>
          공유된 곡
          <span class="badge-count" id="comm-count">0</span>
        </div>
      </div>
      <div class="nav-section">
        <div class="nav-section-title">Library</div>
        <div class="nav-item" data-page="settings" onclick="navigate('settings')">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
          설정
        </div>
      </div>
    </nav>
    <div class="sidebar-foot">
      <div class="user-card" onclick="logout()">
        <div class="avatar" id="user-avatar">U</div>
        <div class="user-info">
          <div class="user-name" id="user-name">User</div>
          <div class="user-plan" id="user-plan">Free Plan</div>
        </div>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" style="color:var(--text-mute)"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/></svg>
      </div>
    </div>
  </aside>

  <div class="main">
    <div class="topbar">
      <div style="display:flex;align-items:center;gap:12px;">
        <button class="icon-btn" onclick="document.getElementById('sidebar').classList.toggle('open')" style="display:none;" id="menu-btn">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="18" x2="21" y2="18"/></svg>
        </button>
        <div style="font-size:13px;color:var(--text-dim);" id="breadcrumb">대시보드</div>
      </div>
      <div class="topbar-actions">
        <button class="icon-btn" onclick="toast('알림이 없습니다','info')" title="알림">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/></svg>
        </button>
      </div>
    </div>

    <div class="content">
      <div class="page-view" data-view="dashboard">
        <div class="page-title">안녕하세요, <span id="hello-name">User</span>님 👋</div>
        <div class="page-sub">오늘도 음악을 악보로 만들어 보세요.</div>
        <div class="stats-row">
          <div class="stat"><div class="stat-head"><div class="stat-label">총 변환</div></div><div class="stat-value" id="stat-total">0</div></div>
          <div class="stat"><div class="stat-head"><div class="stat-label">총 노트</div></div><div class="stat-value" id="stat-notes">0</div></div>
          <div class="stat"><div class="stat-head"><div class="stat-label">평균 시간</div></div><div class="stat-value" id="stat-time">0s</div></div>
          <div class="stat"><div class="stat-head"><div class="stat-label">요금제</div></div><div class="stat-value" style="font-size:18px;">Free</div></div>
        </div>
        <div class="card">
          <div class="card-head">
            <div><div class="card-title">최근 변환</div><div class="card-desc">최근에 변환한 파일</div></div>
            <button class="btn-ghost" onclick="navigate('history')">전체 보기 →</button>
          </div>
          <div id="recent-list"></div>
        </div>
        <div class="card">
          <div class="card-head">
            <div><div class="card-title">커뮤니티 신작</div><div class="card-desc">다른 사람들이 공유한 최신 곡들</div></div>
            <button class="btn-ghost" onclick="navigate('community')">더 보기 →</button>
          </div>
          <div id="dash-community"></div>
        </div>
      </div>

      <div class="page-view" data-view="convert" style="display:none;">
        <div class="page-title">새 변환</div>
        <div class="page-sub">파일을 업로드하거나 YouTube 링크를 붙여넣으세요.</div>
        <div class="card">
          <div class="source-tabs">
            <div class="source-tab active" data-src="file" onclick="switchSource('file')">📁 파일 업로드</div>
            <div class="source-tab" data-src="yt" onclick="switchSource('yt')">▶️ YouTube 링크</div>
          </div>
          <div class="source-pane active" id="pane-file">
            <label class="dropzone" id="drop">
              <input type="file" id="file" accept="audio/*,.mp3,.wav,.m4a,.flac,.ogg">
              <div class="dz-icon"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg></div>
              <div class="dz-main" id="main">음원 파일을 드래그 앤 드롭</div>
              <div class="dz-hint">MP3 · WAV · M4A · FLAC · OGG · 최대 5분</div>
            </label>
          </div>
          <div class="source-pane" id="pane-yt">
            <div class="alert info">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="flex-shrink:0;margin-top:2px;"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
              <span><strong>YouTube 봇 감지 우회:</strong> 쿠키 파일(<code style="background:rgba(0,0,0,0.3);padding:2px 6px;border-radius:4px;">cookies.txt</code>)을 프로젝트 폴더에 넣으면 인증됩니다. 브라우저에서 YouTube 로그인 후 'Get cookies.txt LOCALLY' 확장으로 내보내세요.</span>
            </div>
            <div class="yt-row">
              <input type="text" id="yt-url" placeholder="https://www.youtube.com/watch?v=...">
              <button id="yt-go" onclick="startYoutube()">변환 시작</button>
            </div>
            <div class="yt-hint">💡 최대 5분. 쿠키 없으면 tv/mweb 클라이언트로 자동 시도합니다.</div>
          </div>
          <div class="mode-grid">
            <div class="mode-card active" data-mode="auto"><div class="m-name">자동 감지</div><div class="m-desc">AI가 최적 모드 선택</div></div>
            <div class="mode-card" data-mode="piano"><div class="m-name">피아노</div><div class="m-desc">피아노 전용 정밀 전사</div></div>
            <div class="mode-card" data-mode="universal"><div class="m-name">만능</div><div class="m-desc">다악기 → 피아노</div></div>
          </div>
          <div class="alert warn" style="margin-top:16px;">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="flex-shrink:0;margin-top:2px;"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
            <span>만능 모드는 MuScriptor(CC BY-NC 4.0)를 사용하므로 <strong>비상업적 용도로만</strong> 사용 가능합니다. YouTube 영상은 저작권자의 허락을 받은 경우에만 사용하세요.</span>
          </div>
          <button class="btn-primary" id="go" disabled>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>
            변환 시작하기
          </button>
          <div class="progress-card" id="progress">
            <div class="prog-bar-bg"><div class="prog-bar-fill"></div></div>
            <div class="prog-status"><div class="spinner"></div><span id="status">대기 중...</span></div>
          </div>
          <div class="alert err" id="err"></div>
        </div>
        <div class="card" id="result-card" style="display:none;">
          <div class="card-head">
            <div><div class="card-title">✨ 변환 완료</div><div class="card-desc" id="result-desc">결과물을 다운로드하세요.</div></div>
          </div>
          <div class="stats-row" id="result-stats"></div>
          <div class="result-grid" id="files"></div>
          <div class="share-bar">
            <div>
              <div class="sb-title">🌐 커뮤니티에 공유하기</div>
              <div class="sb-desc">공유하면 누구나 볼 수 있는 고유 링크가 생성됩니다.</div>
            </div>
            <button id="share-btn" onclick="openShareModal()">공유</button>
          </div>
          <div class="card-title" style="margin-top:24px;margin-bottom:12px;">악보 미리보기</div>
          <div class="score-container" id="score-view"></div>
        </div>
      </div>

      <div class="page-view" data-view="history" style="display:none;">
        <div class="page-title">내 히스토리</div>
        <div class="page-sub">지금까지 변환한 모든 파일</div>
        <div class="card"><div id="history-list"></div></div>
      </div>

      <div class="page-view" data-view="community" style="display:none;">
        <div class="page-title">🌐 커뮤니티</div>
        <div class="page-sub">다른 사용자들이 공유한 곡들</div>
        <div class="card">
          <div class="card-head">
            <div><div class="card-title">최신 공유</div><div class="card-desc" id="comm-desc">총 0곡</div></div>
            <button class="btn-ghost" onclick="loadCommunity()">새로고침 ↻</button>
          </div>
          <div class="community-grid" id="community-grid"></div>
        </div>
      </div>

      <div class="page-view" data-view="settings" style="display:none;">
        <div class="page-title">설정</div>
        <div class="page-sub">계정 및 앱 환경설정</div>
        <div class="card">
          <div class="card-title" style="margin-bottom:16px;">일반</div>
          <div class="settings-row">
            <div><div class="settings-label">이메일 알림</div><div class="settings-desc">변환 완료 시 알림</div></div>
            <div class="toggle on" onclick="this.classList.toggle('on')"></div>
          </div>
          <div class="settings-row">
            <div><div class="settings-label">자동 다운로드</div><div class="settings-desc">변환 완료 시 자동 다운로드</div></div>
            <div class="toggle" onclick="this.classList.toggle('on')"></div>
          </div>
        </div>
        <div class="card">
          <div class="card-title" style="margin-bottom:16px;">계정</div>
          <div class="settings-row">
            <div><div class="settings-label">로그아웃</div><div class="settings-desc">현재 기기에서 로그아웃</div></div>
            <button class="btn-ghost" style="background:rgba(239,68,68,0.1);color:#fca5a5;" onclick="logout()">로그아웃</button>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<div class="modal-backdrop" id="auth-modal">
  <div class="modal">
    <button class="modal-close" onclick="closeAuth()">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
    </button>
    <h2 id="auth-title">다시 만나서 반가워요</h2>
    <p class="sub" id="auth-sub">계정에 로그인하세요.</p>
    <div class="tabs">
      <div class="tab active" data-tab="login" onclick="switchAuth('login')">로그인</div>
      <div class="tab" data-tab="signup" onclick="switchAuth('signup')">회원가입</div>
    </div>
    <form id="auth-form" onsubmit="return handleAuth(event)">
      <div class="field" id="name-field" style="display:none;">
        <label>이름</label><input type="text" id="auth-name" placeholder="홍길동">
      </div>
      <div class="field"><label>이메일</label><input type="email" id="auth-email" placeholder="you@example.com" required></div>
      <div class="field"><label>비밀번호</label><input type="password" id="auth-pw" placeholder="••••••••" required minlength="4"></div>
      <button type="submit" class="btn-submit" id="auth-submit">로그인</button>
    </form>
    <div class="auth-switch">
      <span id="auth-switch-text">계정이 없으신가요?</span>
      <a onclick="switchAuth(document.querySelector('.tab.active').dataset.tab === 'login' ? 'signup' : 'login')" id="auth-switch-link">회원가입</a>
    </div>
  </div>
</div>

<div class="modal-backdrop" id="share-modal">
  <div class="modal">
    <button class="modal-close" onclick="closeShareModal()">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
    </button>
    <h2>🌐 커뮤니티에 공유</h2>
    <p class="sub">곡 정보를 입력하세요. 누구나 볼 수 있는 링크가 생성됩니다.</p>
    <div class="field"><label>곡 제목</label><input type="text" id="share-title" placeholder="예: HIKARI - Piano Arrangement"></div>
    <div class="field"><label>설명 (선택)</label><textarea id="share-desc" placeholder="이 곡에 대한 설명이나 메모..."></textarea></div>
    <div class="field"><label>작성자</label><input type="text" id="share-author" readonly></div>
    <button class="btn-submit" id="share-submit" onclick="submitShare()">커뮤니티에 게시</button>
  </div>
</div>

<div class="modal-backdrop" id="success-modal">
  <div class="modal" style="text-align:center;">
    <div style="width:64px;height:64px;border-radius:50%;background:rgba(16,185,129,0.15);
      border:1px solid rgba(16,185,129,0.3);display:grid;place-items:center;margin:0 auto 20px;color:#6ee7b7;">
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><polyline points="20 6 9 17 4 12"/></svg>
    </div>
    <h2 style="margin-bottom:8px;">공유 완료!</h2>
    <p class="sub" style="margin-bottom:20px;">아래 링크로 누구나 이 곡을 볼 수 있습니다.</p>
    <div style="display:flex;gap:8px;background:rgba(0,0,0,0.3);border:1px solid var(--border);
      border-radius:10px;padding:12px;margin-bottom:16px;align-items:center;">
      <input type="text" id="share-url" readonly style="flex:1;background:none;border:none;color:#a5b4fc;
        font-size:12px;font-family:var(--mono);outline:none;">
    </div>
    <div style="display:flex;gap:8px;">
      <button class="btn-submit" style="margin-top:0;flex:1;" onclick="copyShareUrl()">링크 복사</button>
      <button class="btn-submit" style="margin-top:0;flex:1;background:rgba(255,255,255,0.06);box-shadow:none;"
        onclick="window.open(document.getElementById('share-url').value,'_blank')">새 창에서 보기</button>
    </div>
    <button class="btn-ghost" style="width:100%;margin-top:12px;" onclick="closeSuccessModal();navigate('community')">커뮤니티로 이동</button>
  </div>
</div>

<div class="toasts" id="toasts"></div>

<script>
const State={user:null,history:[],currentJob:null,currentFiles:null,currentMeta:null,mode:'auto',file:null,source:'file'};
const LS={get:(k,d)=>{try{return JSON.parse(localStorage.getItem(k))??d}catch{return d}},
set:(k,v)=>localStorage.setItem(k,JSON.stringify(v)),del:k=>localStorage.removeItem(k)};

function toast(msg,type='info',dur=3000){
  const el=document.createElement('div');el.className=`toast ${type}`;
  el.innerHTML=`<span class="dot"></span><span>${escapeHtml(msg)}</span>`;
  document.getElementById('toasts').appendChild(el);
  setTimeout(()=>{el.style.transition='all 0.3s';el.style.opacity='0';el.style.transform='translateX(20px)';setTimeout(()=>el.remove(),300)},dur);
}
function escapeHtml(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}

function openAuth(m='login'){document.getElementById('auth-modal').classList.add('on');switchAuth(m)}
function closeAuth(){document.getElementById('auth-modal').classList.remove('on')}
function switchAuth(m){
  document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('active',t.dataset.tab===m));
  const isL=m==='login';
  document.getElementById('auth-title').textContent=isL?'다시 만나서 반가워요':'계정 만들기';
  document.getElementById('auth-sub').textContent=isL?'계정에 로그인하세요.':'몇 초면 시작할 수 있습니다.';
  document.getElementById('auth-submit').textContent=isL?'로그인':'회원가입';
  document.getElementById('name-field').style.display=isL?'none':'block';
  document.getElementById('auth-name').required=!isL;
  document.getElementById('auth-switch-text').textContent=isL?'계정이 없으신가요?':'이미 계정이 있으신가요?';
  document.getElementById('auth-switch-link').textContent=isL?'회원가입':'로그인';
}
function handleAuth(e){
  e.preventDefault();
  const isL=document.querySelector('.tab.active').dataset.tab==='login';
  const email=document.getElementById('auth-email').value.trim();
  const pw=document.getElementById('auth-pw').value;
  const name=document.getElementById('auth-name').value.trim();
  if(!email||!pw){toast('모든 필드를 입력해주세요','error');return false}
  if(!isL&&!name){toast('이름을 입력해주세요','error');return false}
  const users=LS.get('pf_users',{});
  if(isL){
    if(!users[email]){users[email]={name:email.split('@')[0],pw};LS.set('pf_users',users)}
    else if(users[email].pw!==pw){toast('비밀번호가 일치하지 않습니다','error');return false}
    State.user={email,name:users[email].name};
  }else{
    if(users[email]){toast('이미 가입된 이메일입니다','error');return false}
    users[email]={name,pw};LS.set('pf_users',users);State.user={email,name};
  }
  LS.set('pf_user',State.user);
  toast(`환영합니다, ${State.user.name}님!`,'success');
  closeAuth();enterApp();return false;
}
function logout(){
  LS.del('pf_user');State.user=null;toast('로그아웃되었습니다','info');
  document.getElementById('view-app').classList.remove('active');
  document.getElementById('view-landing').classList.add('active');
}

function navigate(page){
  document.querySelectorAll('.nav-item').forEach(n=>n.classList.toggle('active',n.dataset.page===page));
  document.querySelectorAll('.page-view').forEach(v=>v.style.display=v.dataset.view===page?'block':'none');
  const titles={dashboard:'대시보드',convert:'새 변환',history:'내 히스토리',community:'커뮤니티',settings:'설정'};
  document.getElementById('breadcrumb').textContent=titles[page]||page;
  if(window.innerWidth<=900)document.getElementById('sidebar').classList.remove('open');
  if(page==='community')loadCommunity();
  if(page==='dashboard'){loadCommunity(true);renderHistory();updateStats()}
}
function enterApp(){
  document.getElementById('view-landing').classList.remove('active');
  document.getElementById('view-app').classList.add('active');
  const name=State.user.name;
  document.getElementById('user-name').textContent=name;
  document.getElementById('user-plan').textContent='Free Plan';
  document.getElementById('user-avatar').textContent=name.charAt(0).toUpperCase();
  document.getElementById('hello-name').textContent=name;
  if(window.innerWidth<=900)document.getElementById('menu-btn').style.display='grid';
  renderHistory();updateStats();navigate('dashboard');
}

function addHistory(meta,files,jobId){
  State.history.unshift({id:Date.now(),jobId,name:meta.filename,notes:meta.notes,bpm:meta.bpm,
    duration:meta.duration_sec,files,date:new Date().toISOString()});
  State.history=State.history.slice(0,50);
  LS.set('pf_history',State.history);renderHistory();updateStats();
}
function renderHistory(){
  State.history=LS.get('pf_history',[]);
  document.getElementById('hist-count').textContent=State.history.length;
  const hl=document.getElementById('history-list');
  hl.innerHTML=State.history.length===0?`<div class="empty">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
    <div>아직 변환 기록이 없습니다</div></div>`:
    State.history.map((h,i)=>`<div style="display:flex;align-items:center;gap:12px;padding:12px 14px;border-radius:12px;background:rgba(0,0,0,0.2);border:1px solid var(--border);margin-bottom:8px;">
      <div style="width:28px;height:28px;border-radius:8px;background:var(--grad-subtle);display:grid;place-items:center;font-size:11px;font-weight:700;color:#a5b4fc;">${State.history.length-i}</div>
      <div style="flex:1;min-width:0;">
        <div style="font-size:13px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${escapeHtml(h.name)}</div>
        <div style="font-size:11px;color:var(--text-mute);">${h.notes} 노트 · ${h.bpm} BPM · ${h.duration}초 · ${fmtDate(h.date)}</div>
      </div>
      ${h.files?.midi?`<a href="${h.files.midi}" download style="font-size:10px;padding:3px 8px;border-radius:100px;background:rgba(16,185,129,0.1);color:var(--green);font-weight:600;text-decoration:none;">MIDI</a>`:''}
    </div>`).join('');
  const rl=document.getElementById('recent-list');
  rl.innerHTML=State.history.length===0?`<div class="empty"><div>아직 변환한 파일이 없습니다</div></div>`:
    State.history.slice(0,4).map((h,i)=>`<div style="display:flex;align-items:center;gap:12px;padding:12px 14px;border-radius:12px;background:rgba(0,0,0,0.2);border:1px solid var(--border);margin-bottom:8px;">
      <div style="width:28px;height:28px;border-radius:8px;background:var(--grad-subtle);display:grid;place-items:center;font-size:11px;font-weight:700;color:#a5b4fc;">${i+1}</div>
      <div style="flex:1;min-width:0;">
        <div style="font-size:13px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${escapeHtml(h.name)}</div>
        <div style="font-size:11px;color:var(--text-mute);">${h.notes} 노트 · ${h.bpm} BPM</div>
      </div>
    </div>`).join('');
}
function updateStats(){
  State.history=LS.get('pf_history',[]);
  document.getElementById('stat-total').textContent=State.history.length;
  document.getElementById('stat-notes').textContent=State.history.reduce((a,h)=>a+(h.notes||0),0).toLocaleString();
  const avg=State.history.length>0?Math.round(State.history.reduce((a,h)=>a+(h.duration||0),0)/State.history.length):0;
  document.getElementById('stat-time').textContent=avg+'s';
}
function fmtDate(iso){
  const d=new Date(iso),diff=(Date.now()-d.getTime())/1000;
  if(diff<60)return'방금 전';
  if(diff<3600)return Math.floor(diff/60)+'분 전';
  if(diff<86400)return Math.floor(diff/3600)+'시간 전';
  return d.toLocaleDateString('ko-KR');
}

function switchSource(s){
  State.source=s;
  document.querySelectorAll('.source-tab').forEach(t=>t.classList.toggle('active',t.dataset.src===s));
  document.querySelectorAll('.source-pane').forEach(p=>p.classList.toggle('active',p.id==='pane-'+s));
  document.getElementById('go').disabled=true;
  if(s==='file'&&State.file)document.getElementById('go').disabled=false;
}
const drop=document.getElementById('drop');
document.querySelectorAll('.mode-card').forEach(c=>c.addEventListener('click',()=>{
  document.querySelectorAll('.mode-card').forEach(x=>x.classList.remove('active'));
  c.classList.add('active');State.mode=c.dataset.mode;
}));
drop.addEventListener('dragover',e=>{e.preventDefault();drop.classList.add('active')});
drop.addEventListener('dragleave',()=>drop.classList.remove('active'));
drop.addEventListener('drop',e=>{e.preventDefault();drop.classList.remove('active');if(e.dataTransfer.files.length)pickFile(e.dataTransfer.files[0])});
document.getElementById('file').addEventListener('change',e=>{if(e.target.files.length)pickFile(e.target.files[0])});

function pickFile(f){
  State.file=f;drop.classList.add('has-file');
  document.getElementById('main').textContent='✓ '+f.name;
  document.getElementById('go').disabled=false;
  document.getElementById('result-card').style.display='none';
  document.getElementById('err').classList.remove('on');
}

document.getElementById('go').addEventListener('click',async()=>{
  if(State.source==='yt')return startYoutube();
  if(!State.file)return;
  const btn=document.getElementById('go');btn.disabled=true;
  document.getElementById('progress').classList.add('on');
  document.getElementById('result-card').style.display='none';
  document.getElementById('err').classList.remove('on');
  document.getElementById('status').textContent='파일 업로드 중...';
  const fd=new FormData();
  fd.append('file',State.file);fd.append('mode',State.mode);
  try{
    const r=await fetch('/upload',{method:'POST',body:fd});
    const j=await r.json();
    if(j.error)throw new Error(j.error);
    pollJob(j.job_id);
  }catch(e){failJob(e.message)}
});

async function startYoutube(){
  const url=document.getElementById('yt-url').value.trim();
  if(!url)return;
  const btn=document.getElementById('yt-go');btn.disabled=true;
  document.getElementById('go').disabled=true;
  document.getElementById('progress').classList.add('on');
  document.getElementById('result-card').style.display='none';
  document.getElementById('err').classList.remove('on');
  document.getElementById('status').textContent='YouTube 다운로드 중... (최대 2분)';
  try{
    const r=await fetch('/youtube',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({url,mode:State.mode})});
    const j=await r.json();
    if(j.error)throw new Error(j.error);
    const m=j.method?` (${j.method})`:'';
    toast(`"${j.title}" 다운로드 완료${m}`,'success');
    pollJob(j.job_id);
  }catch(e){
    failJob(e.message);
    btn.disabled=false;document.getElementById('go').disabled=false;
    document.getElementById('progress').classList.remove('on');
  }
}

let pollTimer=null;
function pollJob(id){
  State.currentJob=id;
  pollTimer=setInterval(async()=>{
    try{
      const r=await fetch('/status/'+id);
      const j=await r.json();
      if(j.error)throw new Error(j.error);
      if(j.message)document.getElementById('status').textContent=j.message;
      if(j.status==='done'){
        clearInterval(pollTimer);
        document.getElementById('progress').classList.remove('on');
        document.getElementById('go').disabled=false;
        document.getElementById('yt-go').disabled=false;
        showResult(j.result,j);
      }else if(j.status==='error'){
        clearInterval(pollTimer);
        document.getElementById('progress').classList.remove('on');
        document.getElementById('go').disabled=false;
        document.getElementById('yt-go').disabled=false;
        failJob(j.message);
      }
    }catch(e){
      clearInterval(pollTimer);
      document.getElementById('progress').classList.remove('on');
      document.getElementById('go').disabled=false;
      document.getElementById('yt-go').disabled=false;
      failJob(e.message);
    }
  },1000);
}
function failJob(m){
  const el=document.getElementById('err');
  el.classList.add('on');
  el.innerHTML=`<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="flex-shrink:0;margin-top:2px;"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg><span>오류: ${escapeHtml(m)}</span>`;
  toast('변환 중 오류가 발생했습니다','error');
}

const FILE_META={
  midi:{icon:'♪',name:'MIDI 파일',desc:'DAW · 신디사이저'},
  piano_mp3:{icon:'♫',name:'피아노 MP3',desc:'FluidSynth 렌더링'},
  musicxml:{icon:'𝄞',name:'MusicXML 악보',desc:'MuseScore · Finale'},
  svg:{icon:'◈',name:'SVG 악보',desc:'벡터 · 확대 가능'},
  pdf:{icon:'▤',name:'PDF 악보',desc:'인쇄용'},
  meta:{icon:'▦',name:'정보',desc:'변환 통계'},
};

async function showResult(files,jobInfo){
  State.currentFiles=files;
  document.getElementById('result-card').style.display='block';
  const statsEl=document.getElementById('result-stats');
  const filesEl=document.getElementById('files');
  const scoreEl=document.getElementById('score-view');
  statsEl.innerHTML='';filesEl.innerHTML='';scoreEl.classList.remove('on');scoreEl.innerHTML='';

  let meta=null;
  if(files.meta){
    try{
      const r=await fetch(files.meta);meta=await r.json();State.currentMeta=meta;
      const items=[['노트',meta.notes],['BPM',meta.bpm],['성부',meta.voices],['길이',meta.duration_sec+'초']];
      for(const [l,v] of items){
        const el=document.createElement('div');el.className='stat';
        el.innerHTML=`<div class="stat-head"><div class="stat-label">${l}</div></div><div class="stat-value">${v}</div>`;
        statsEl.appendChild(el);
      }
      let desc='결과물을 다운로드하세요.';
      if(jobInfo&&jobInfo.yt_title)desc=`YouTube: ${jobInfo.yt_title}`;
      else desc=`${meta.filename} · ${meta.genre}`;
      document.getElementById('result-desc').textContent=desc;
    }catch(e){}
  }

  const order=['midi','piano_mp3','musicxml','svg','pdf','meta'];
  for(const k of order){
    if(!files[k])continue;
    const m=FILE_META[k]||{icon:'◉',name:k,desc:''};
    const a=document.createElement('a');a.className='result-item';a.href=files[k];
    a.setAttribute('download','');
    a.innerHTML=`<div class="ri-icon">${m.icon}</div><div><div class="ri-name">${m.name}</div><div class="ri-desc">${m.desc}</div></div><div class="ri-arrow">↓</div>`;
    filesEl.appendChild(a);
  }
  if(files.svg){
    try{
      const r=await fetch(files.svg);const svg=await r.text();
      scoreEl.innerHTML=svg;scoreEl.classList.add('on');
    }catch(e){}
  }
  if(meta)addHistory(meta,files,State.currentJob);
  toast('변환 완료! 결과물을 확인하세요.','success');
}

function openShareModal(){
  if(!State.user){toast('로그인이 필요합니다','error');openAuth('login');return}
  if(!State.currentFiles){toast('먼저 변환을 완료해주세요','error');return}
  document.getElementById('share-author').value=State.user.name;
  document.getElementById('share-title').value=State.currentMeta?.filename||'Untitled';
  document.getElementById('share-modal').classList.add('on');
}
function closeShareModal(){document.getElementById('share-modal').classList.remove('on')}
async function submitShare(){
  const title=document.getElementById('share-title').value.trim()||'Untitled';
  const author=document.getElementById('share-author').value.trim()||'익명';
  const desc=document.getElementById('share-desc').value.trim();
  const btn=document.getElementById('share-submit');btn.disabled=true;btn.textContent='공유 중...';
  try{
    const r=await fetch('/api/share',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({job_id:State.currentJob,title,author,description:desc})});
    const j=await r.json();
    if(j.error)throw new Error(j.error);
    closeShareModal();
    const fullUrl=location.origin+j.url;
    document.getElementById('share-url').value=fullUrl;
    document.getElementById('success-modal').classList.add('on');
    toast('커뮤니티에 공유되었습니다!','success');
    btn.disabled=false;btn.textContent='커뮤니티에 게시';
  }catch(e){
    toast('공유 실패: '+e.message,'error');
    btn.disabled=false;btn.textContent='커뮤니티에 게시';
  }
}
function closeSuccessModal(){document.getElementById('success-modal').classList.remove('on')}
function copyShareUrl(){
  const inp=document.getElementById('share-url');
  inp.select();inp.setSelectionRange(0,99999);
  try{document.execCommand('copy');toast('링크가 복사되었습니다!','success')}catch(e){
    navigator.clipboard.writeText(inp.value).then(()=>toast('링크가 복사되었습니다!','success'));
  }
}

async function loadCommunity(forDash=false){
  try{
    const r=await fetch('/api/community');const j=await r.json();
    const items=j.items||[];
    document.getElementById('comm-count').textContent=items.length;
    document.getElementById('comm-desc').textContent=`총 ${items.length}곡`;
    const html=items.length===0?`<div class="empty" style="grid-column:1/-1;">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/></svg>
      <div>아직 공유된 곡이 없습니다. 첫 번째로 공유해보세요!</div></div>`:
      items.slice(0,forDash?3:100).map(it=>`
      <a class="community-card" href="/s/${it.id}" target="_blank">
        <div class="cc-top">
          <div class="cc-avatar">${escapeHtml((it.author||'?').charAt(0).toUpperCase())}</div>
          <div>
            <div class="cc-author">${escapeHtml(it.author||'익명')}</div>
            <div class="cc-date">${fmtDate(new Date(it.date*1000).toISOString())}</div>
          </div>
        </div>
        <div class="cc-title">${escapeHtml(it.title||'Untitled')}</div>
        <div class="cc-tags">
          ${it.meta?.bpm?`<span class="cc-tag">${it.meta.bpm} BPM</span>`:''}
          ${it.meta?.notes?`<span class="cc-tag">${it.meta.notes} 노트</span>`:''}
          ${it.meta?.genre?`<span class="cc-tag">${escapeHtml(it.meta.genre)}</span>`:''}
        </div>
        <div class="cc-meta"><span>👁 ${it.views||0}회</span><span>${it.has_svg?'악보 ✓':''} ${it.has_mp3?'MP3 ✓':''}</span></div>
      </a>`).join('');
    if(forDash)document.getElementById('dash-community').innerHTML=html;
    else document.getElementById('community-grid').innerHTML=html;
  }catch(e){console.error(e)}
}

document.addEventListener('keydown',e=>{
  if(e.key==='Escape'){closeAuth();closeShareModal();closeSuccessModal()}
});

(function init(){
  const saved=LS.get('pf_user',null);
  State.history=LS.get('pf_history',[]);
  if(saved){State.user=saved;enterApp()}
})();
</script>
</body>
</html>
"""

# ═════════════════════════════════════════════════════════════════════════════
# 15. Shared Song Page Template
# ═════════════════════════════════════════════════════════════════════════════
SHARED_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{ item.title }} — PianoForge</title>
<meta property="og:title" content="{{ item.title }}">
<meta property="og:description" content="PianoForge에서 공유한 곡입니다. 악보와 MIDI를 다운로드하세요.">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:"Pretendard Variable",-apple-system,system-ui,sans-serif;background:#08080a;color:#fafafa;min-height:100vh;padding:24px;line-height:1.5}
body::before{content:'';position:fixed;inset:0;pointer-events:none;z-index:0;background:radial-gradient(1000px 600px at 20% -10%,rgba(99,102,241,0.10),transparent 60%),radial-gradient(800px 500px at 90% 10%,rgba(168,85,247,0.08),transparent 60%)}
.wrap{max-width:960px;margin:0 auto;position:relative;z-index:1}
.top{display:flex;align-items:center;gap:12px;padding:16px 0;margin-bottom:20px}
.logo{display:flex;align-items:center;gap:10px;font-weight:700;text-decoration:none;color:inherit}
.logo-mark{width:28px;height:28px;border-radius:8px;background:linear-gradient(135deg,#6366f1,#a855f7,#ec4899);display:grid;place-items:center;font-size:14px;color:#fff;box-shadow:0 0 20px rgba(99,102,241,0.35)}
.btn{padding:8px 16px;border-radius:8px;font-size:13px;font-weight:600;background:rgba(255,255,255,0.06);color:#fafafa;text-decoration:none;border:1px solid rgba(255,255,255,0.12);transition:all .2s;cursor:pointer}
.btn:hover{background:rgba(255,255,255,0.1)}
.btn-primary{background:linear-gradient(135deg,#6366f1,#a855f7);border:none;box-shadow:0 8px 24px -8px rgba(99,102,241,0.4)}
.hero{background:rgba(23,23,28,0.65);border:1px solid rgba(255,255,255,0.07);border-radius:24px;padding:32px;backdrop-filter:blur(12px);margin-bottom:24px}
.hero-top{display:flex;align-items:center;gap:16px;margin-bottom:20px}
.avatar{width:56px;height:56px;border-radius:16px;background:linear-gradient(135deg,#6366f1,#a855f7);display:grid;place-items:center;font-size:22px;font-weight:700;color:#fff;flex-shrink:0;box-shadow:0 8px 24px -8px rgba(99,102,241,0.5)}
.author{font-size:13px;color:#a1a1aa;font-weight:600}
.date{font-size:11px;color:#52525b;margin-top:2px}
h1{font-size:clamp(24px,4vw,36px);font-weight:800;letter-spacing:-0.02em;margin-bottom:12px;background:linear-gradient(180deg,#fff 30%,#a1a1aa);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.tags{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:20px}
.tag{font-size:11px;padding:5px 12px;border-radius:100px;background:rgba(99,102,241,0.1);color:#a5b4fc;font-weight:600;border:1px solid rgba(99,102,241,0.2)}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px}
.stat{padding:14px;background:rgba(0,0,0,0.3);border:1px solid rgba(255,255,255,0.07);border-radius:12px;text-align:center}
.stat-v{font-size:20px;font-weight:800;letter-spacing:-0.02em;margin-bottom:4px}
.stat-l{font-size:10px;text-transform:uppercase;letter-spacing:0.08em;color:#52525b;font-weight:600}
.section-title{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;color:#52525b;margin-bottom:16px;display:flex;align-items:center;gap:8px}
.section-title::after{content:'';flex:1;height:1px;background:rgba(255,255,255,0.07)}
.files{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px;margin-bottom:24px}
.file{display:flex;align-items:center;gap:12px;padding:14px;border-radius:12px;text-decoration:none;color:inherit;background:rgba(0,0,0,0.2);border:1px solid rgba(255,255,255,0.07);transition:all .2s}
.file:hover{border-color:#6366f1;background:rgba(99,102,241,0.06);transform:translateX(2px)}
.file-icon{width:36px;height:36px;border-radius:10px;background:linear-gradient(135deg,rgba(99,102,241,0.15),rgba(168,85,247,0.1));display:grid;place-items:center;color:#a5b4fc;font-size:16px;flex-shrink:0}
.file-name{font-size:13px;font-weight:600}
.file-desc{font-size:11px;color:#52525b}
.file-arrow{margin-left:auto;color:#52525b}
.score{background:#fff;border-radius:12px;padding:20px;overflow-x:auto}
.score svg{display:block;max-width:100%;height:auto}
.footer{text-align:center;padding:40px 0 20px;color:#52525b;font-size:12px}
.footer a{color:#a5b4fc;text-decoration:none}
@media (max-width:600px){.stats{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <a class="logo" href="/"><div class="logo-mark">♪</div>PianoForge</a>
    <div style="flex:1"></div>
    <a class="btn btn-primary" href="/">나도 만들어보기</a>
  </div>

  <div class="hero">
    <div class="hero-top">
      <div class="avatar">{{ item.author[0]|upper }}</div>
      <div>
        <div class="author">{{ item.author }}</div>
        <div class="date">공유됨 · 👁 {{ item.views }}회 조회</div>
      </div>
    </div>
    <h1>{{ item.title }}</h1>
    <div class="tags">
      {% if item.meta.bpm %}<span class="tag">{{ item.meta.bpm }} BPM</span>{% endif %}
      {% if item.meta.notes %}<span class="tag">{{ item.meta.notes }} 노트</span>{% endif %}
      {% if item.meta.voices %}<span class="tag">{{ item.meta.voices }} 성부</span>{% endif %}
      {% if item.meta.genre %}<span class="tag">{{ item.meta.genre }}</span>{% endif %}
      {% if item.meta.duration_sec %}<span class="tag">{{ item.meta.duration_sec }}초</span>{% endif %}
    </div>
    <div class="stats">
      <div class="stat"><div class="stat-v">{{ item.meta.notes|default('—') }}</div><div class="stat-l">노트</div></div>
      <div class="stat"><div class="stat-v">{{ item.meta.bpm|default('—') }}</div><div class="stat-l">BPM</div></div>
      <div class="stat"><div class="stat-v">{{ item.meta.voices|default('—') }}</div><div class="stat-l">성부</div></div>
      <div class="stat"><div class="stat-v">{{ item.meta.duration_sec|default('—') }}</div><div class="stat-l">초</div></div>
    </div>
  </div>

  <div class="section-title">다운로드</div>
  <div class="files">
    {% for k, fn in item.files.items() %}
      {% if k == 'mid' %}<a class="file" href="/shared_dl/{{ sid }}/{{ fn }}"><div class="file-icon">♪</div><div><div class="file-name">MIDI 파일</div><div class="file-desc">DAW · 신디사이저</div></div><div class="file-arrow">↓</div></a>{% endif %}
      {% if k == 'mp3' %}<a class="file" href="/shared_dl/{{ sid }}/{{ fn }}"><div class="file-icon">♫</div><div><div class="file-name">피아노 MP3</div><div class="file-desc">FluidSynth 렌더링</div></div><div class="file-arrow">↓</div></a>{% endif %}
      {% if k == 'musicxml' %}<a class="file" href="/shared_dl/{{ sid }}/{{ fn }}"><div class="file-icon">𝄞</div><div><div class="file-name">MusicXML</div><div class="file-desc">MuseScore · Finale</div></div><div class="file-arrow">↓</div></a>{% endif %}
      {% if k == 'svg' %}<a class="file" href="/shared_dl/{{ sid }}/{{ fn }}"><div class="file-icon">◈</div><div><div class="file-name">SVG 악보</div><div class="file-desc">벡터 · 확대 가능</div></div><div class="file-arrow">↓</div></a>{% endif %}
      {% if k == 'pdf' %}<a class="file" href="/shared_dl/{{ sid }}/{{ fn }}"><div class="file-icon">▤</div><div><div class="file-name">PDF 악보</div><div class="file-desc">인쇄용</div></div><div class="file-arrow">↓</div></a>{% endif %}
    {% endfor %}
  </div>

  {% if item.has_svg %}
  <div class="section-title">악보 미리보기</div>
  <div class="score" id="score-view">로딩 중...</div>
  <script>
    (function(){
      fetch('/shared/{{ sid }}/{{ item.files.svg }}')
        .then(r=>r.text()).then(t=>{document.getElementById('score-view').innerHTML=t})
        .catch(e=>{document.getElementById('score-view').textContent='악보를 불러올 수 없습니다'});
    })();
  </script>
  {% endif %}

  <div class="footer">Powered by <a href="/">PianoForge Studio</a> — AI Audio to Score Platform</div>
</div>
</body>
</html>
"""


# ═════════════════════════════════════════════════════════════════════════════
# 16. Run
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\n  PianoForge v15 → http://localhost:{port}\n")
    try:
        from waitress import serve
        serve(app, host="0.0.0.0", port=port, threads=4)
    except ImportError:
        app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
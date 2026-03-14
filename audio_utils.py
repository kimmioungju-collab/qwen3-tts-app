"""
[Senior Developer] 오디오 유틸리티
- 마이크 녹음 (sounddevice)
- 포맷 변환 (m4a / mp3 → wav, wav → m4a)
- 임시 파일 관리
"""
from __future__ import annotations

import io
import shutil
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 44_100
CHANNELS = 1
DTYPE = "float32"


# ── 녹음 ───────────────────────────────────────────────────────────────────


class Recorder:
    """스레드 안전 마이크 녹음기."""

    def __init__(self) -> None:
        self._frames: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None
        self._recording = False

    @property
    def is_recording(self) -> bool:
        return self._recording

    def start(self) -> None:
        if self._recording:
            return
        with self._lock:
            self._frames.clear()
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=self._callback,
        )
        self._stream.start()
        self._recording = True

    def stop(self) -> np.ndarray:
        if not self._recording:
            return np.array([], dtype=np.float32)
        self._stream.stop()
        self._stream.close()
        self._recording = False
        with self._lock:
            if not self._frames:
                return np.array([], dtype=np.float32)
            return np.concatenate(self._frames, axis=0).flatten()

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,  # noqa: ARG002
        time_info: object,  # noqa: ARG002
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            print(f"[Recorder] 경고: {status}")
        with self._lock:
            self._frames.append(indata.copy())

    def save_wav(self, audio: np.ndarray, path: str | Path) -> Path:
        out = Path(path)
        sf.write(str(out), audio, SAMPLE_RATE)
        return out


# ── 포맷 변환 ──────────────────────────────────────────────────────────────


def _require_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg가 설치되지 않았습니다.\n"
            "  brew install ffmpeg  로 설치 후 재시도하세요."
        )


def any_to_wav(src: str | Path, dst: str | Path | None = None) -> Path:
    """m4a / mp3 / wav → wav 변환.  dst=None이면 임시파일 생성."""
    src = Path(src)
    ext = src.suffix.lower()

    if dst is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        dst = Path(tmp.name)
        tmp.close()
    else:
        dst = Path(dst)

    if ext == ".wav":
        # 이미 wav — 그냥 복사
        shutil.copy2(src, dst)
        return dst

    # m4a / mp3 → ffmpeg 필요
    _require_ffmpeg()
    try:
        from pydub import AudioSegment

        seg = AudioSegment.from_file(str(src))
        seg = seg.set_channels(1).set_frame_rate(SAMPLE_RATE)
        seg.export(str(dst), format="wav")
    except ImportError:
        # pydub 없으면 ffmpeg 직접 호출
        import subprocess

        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(src),
                "-ac", "1", "-ar", str(SAMPLE_RATE), str(dst),
            ],
            check=True,
            capture_output=True,
        )
    return dst


def wav_to_m4a(src: str | Path, dst: str | Path | None = None) -> Path:
    """wav → m4a 변환 (다운로드용)."""
    _require_ffmpeg()
    src = Path(src)
    if dst is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".m4a", delete=False)
        dst = Path(tmp.name)
        tmp.close()
    else:
        dst = Path(dst)

    try:
        from pydub import AudioSegment

        seg = AudioSegment.from_wav(str(src))
        seg.export(str(dst), format="mp4", codec="aac")
    except ImportError:
        import subprocess

        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), "-c:a", "aac", "-b:a", "192k", str(dst)],
            check=True,
            capture_output=True,
        )
    return dst


def numpy_to_wav(audio: np.ndarray, sample_rate: int) -> Path:
    """numpy array → 임시 wav 파일."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    sf.write(tmp.name, audio, sample_rate)
    return Path(tmp.name)


# ── 재생 ───────────────────────────────────────────────────────────────────


class Player:
    """비동기 오디오 재생기."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def play(self, wav_path: str | Path) -> None:
        self.stop()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._play_worker, args=(str(wav_path),), daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            sd.stop()
            self._thread.join(timeout=2)

    def _play_worker(self, path: str) -> None:
        try:
            data, sr = sf.read(path, dtype="float32")
            sd.play(data, sr)
            sd.wait()
        except Exception as e:
            print(f"[Player] 재생 오류: {e}")

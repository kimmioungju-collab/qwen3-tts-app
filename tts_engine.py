"""
[Senior Developer] Qwen3-TTS 엔진 래퍼
공식 README 기반 구현 — Mac M-series 최적화
핵심 변경: flash_attention_2 → sdpa, cuda → mps/cpu
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

import numpy as np
import torch

# ── 디바이스 / dtype 자동 감지 ─────────────────────────────────────────────


def detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def detect_dtype(device: str) -> torch.dtype:
    """
    [QA] Mac MPS에서 bfloat16은 PyTorch 2.3+ 필요.
    안전하게 float16 사용. CPU는 float32.
    """
    if device == "cuda:0":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def detect_attn() -> str:
    """
    [QA CRITICAL] flash_attention_2는 CUDA 전용.
    Mac은 반드시 'sdpa' 사용.
    """
    try:
        import flash_attn  # noqa: F401
        if torch.cuda.is_available():
            return "flash_attention_2"
    except ImportError:
        pass
    return "sdpa"


# ── 모델 상수 ──────────────────────────────────────────────────────────────

MODELS: dict[str, str] = {
    "0.6B-Base (경량 클론)":        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "0.6B-CustomVoice (경량 프리셋)": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "1.7B-Base (고품질 클론)":       "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "1.7B-CustomVoice (고품질 프리셋)": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "1.7B-VoiceDesign (보이스 디자인)": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}

SPEAKERS: list[str] = [
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
    "Ryan", "Aiden", "Ono_Anna", "Sohee",
]

LANGUAGES: list[str] = [
    "Chinese", "English", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese", "Spanish", "Italian",
]


# ── 엔진 ───────────────────────────────────────────────────────────────────


class Qwen3TTSEngine:
    """
    Qwen3-TTS 공식 API 래퍼.
    스레드 안전 — 한 번에 하나의 생성 작업만 허용.
    """

    def __init__(self) -> None:
        self._model = None
        self._current_model_id: str | None = None
        self._lock = threading.Lock()
        self.device = detect_device()
        self.dtype = detect_dtype(self.device)
        self.attn = detect_attn()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def device_info(self) -> str:
        labels = {
            "mps": "Apple Silicon (MPS)",
            "cuda:0": "NVIDIA GPU (CUDA)",
            "cpu": "CPU (느림 — 0.6B 권장)",
        }
        return labels.get(self.device, self.device)

    def load_model(
        self,
        model_key: str,
        progress_cb: Callable[[str], None] | None = None,
    ) -> None:
        """
        모델 로드. 동일 모델이면 스킵.
        progress_cb: UI 진행 상황 콜백
        """
        model_id = MODELS[model_key]
        if self._current_model_id == model_id and self._model is not None:
            return

        def _report(msg: str) -> None:
            if progress_cb:
                progress_cb(msg)

        with self._lock:
            _report(f"모델 로드 중: {model_id}")
            _report(f"  디바이스: {self.device_info()}")
            _report(f"  dtype: {self.dtype}")
            _report(f"  attn: {self.attn}")

            try:
                from qwen_tts import Qwen3TTSModel

                self._model = Qwen3TTSModel.from_pretrained(
                    model_id,
                    device_map=self.device,
                    dtype=self.dtype,
                    attn_implementation=self.attn,
                )
                self._current_model_id = model_id
                _report("✅ 모델 로드 완료")
            except Exception as e:
                self._model = None
                self._current_model_id = None
                raise RuntimeError(f"모델 로드 실패: {e}") from e

    def generate_custom_voice(
        self,
        text: str,
        language: str,
        speaker: str,
        instruct: str = "",
    ) -> tuple[np.ndarray, int]:
        """공식 API: generate_custom_voice"""
        self._assert_loaded()
        with self._lock:
            wavs, sr = self._model.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker,
                instruct=instruct or None,
            )
            return np.array(wavs[0]), int(sr)

    def generate_voice_design(
        self,
        text: str,
        language: str,
        instruct: str,
    ) -> tuple[np.ndarray, int]:
        """공식 API: generate_voice_design"""
        self._assert_loaded()
        with self._lock:
            wavs, sr = self._model.generate_voice_design(
                text=text,
                language=language,
                instruct=instruct,
            )
            return np.array(wavs[0]), int(sr)

    def generate_voice_clone(
        self,
        text: str,
        language: str,
        ref_audio: str,
        ref_text: str,
    ) -> tuple[np.ndarray, int]:
        """공식 API: generate_voice_clone"""
        self._assert_loaded()
        with self._lock:
            wavs, sr = self._model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_audio,
                ref_text=ref_text,
            )
            return np.array(wavs[0]), int(sr)

    def _assert_loaded(self) -> None:
        if self._model is None:
            raise RuntimeError("모델이 로드되지 않았습니다. 먼저 load_model()을 호출하세요.")

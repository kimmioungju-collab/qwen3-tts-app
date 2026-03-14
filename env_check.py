"""
[QA Specialist] 환경 사전 검증 모듈
실행 시 Mac 호환성 전체 체크 후 결과 반환
"""
from __future__ import annotations
import sys
import shutil
import importlib
from dataclasses import dataclass, field
from typing import Literal

Status = Literal["OK", "WARN", "FAIL"]


@dataclass
class CheckResult:
    name: str
    status: Status
    message: str
    fix: str = ""


@dataclass
class EnvReport:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, status: Status, message: str, fix: str = "") -> None:
        self.results.append(CheckResult(name, status, message, fix))

    @property
    def has_critical(self) -> bool:
        return any(r.status == "FAIL" for r in self.results)

    def summary(self) -> str:
        lines = ["=" * 52, "  Qwen3-TTS Mac 환경 검증 결과", "=" * 52]
        for r in self.results:
            icon = {"OK": "✅", "WARN": "⚠️ ", "FAIL": "❌"}[r.status]
            lines.append(f"  {icon} {r.name}: {r.message}")
            if r.fix:
                lines.append(f"      → 수정: {r.fix}")
        lines.append("=" * 52)
        ok = sum(1 for r in self.results if r.status == "OK")
        warn = sum(1 for r in self.results if r.status == "WARN")
        fail = sum(1 for r in self.results if r.status == "FAIL")
        lines.append(f"  결과: ✅{ok}  ⚠️ {warn}  ❌{fail}")
        return "\n".join(lines)


def run_checks() -> EnvReport:
    report = EnvReport()

    # ── Python 버전 ─────────────────────────────────────────
    major, minor = sys.version_info.major, sys.version_info.minor
    if major == 3 and minor >= 10:
        report.add("Python", "OK", f"{major}.{minor} (권장: 3.12)")
    else:
        report.add("Python", "FAIL", f"{major}.{minor} — 3.10 이상 필요",
                   "conda create -n qwen3-tts python=3.12")

    # ── PyTorch & MPS ───────────────────────────────────────
    try:
        import torch
        mps = torch.backends.mps.is_available()
        cuda = torch.cuda.is_available()
        if mps:
            report.add("PyTorch/MPS", "OK",
                       f"v{torch.__version__} — Apple Silicon MPS ✅")
        elif cuda:
            report.add("PyTorch/CUDA", "OK",
                       f"v{torch.__version__} — CUDA ✅")
        else:
            report.add("PyTorch/CPU", "WARN",
                       f"v{torch.__version__} — CPU 모드 (느림)",
                       "0.6B 모델 사용 권장")

        # bfloat16 MPS 지원 (PyTorch 2.3+)
        if mps:
            major_pt = int(torch.__version__.split(".")[0])
            minor_pt = int(torch.__version__.split(".")[1])
            if major_pt >= 2 and minor_pt >= 3:
                report.add("bfloat16/MPS", "OK", "PyTorch 2.3+ — MPS bfloat16 지원")
            else:
                report.add("bfloat16/MPS", "WARN",
                           f"PyTorch {torch.__version__} — float32 fallback 사용",
                           "pip install -U torch")
    except ImportError:
        report.add("PyTorch", "FAIL", "미설치", "pip install torch")

    # ── flash-attn (Mac에서는 없어야 정상) ──────────────────
    try:
        import flash_attn  # noqa: F401
        report.add("flash-attn", "WARN",
                   "설치됨 — sdpa로 강제 override 됩니다")
    except ImportError:
        report.add("flash-attn", "OK", "미설치 (Mac에서 정상) → sdpa 사용")

    # ── qwen-tts ────────────────────────────────────────────
    try:
        import qwen_tts  # noqa: F401
        report.add("qwen-tts", "OK", "설치됨")
    except ImportError:
        report.add("qwen-tts", "FAIL", "미설치", "pip install -U qwen-tts")

    # ── PySide6 ─────────────────────────────────────────────
    try:
        import PySide6
        report.add("PySide6", "OK", f"v{PySide6.__version__}")
    except ImportError:
        report.add("PySide6", "FAIL", "미설치", "pip install PySide6")

    # ── sounddevice ─────────────────────────────────────────
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        input_devs = [d for d in devices if d["max_input_channels"] > 0]
        report.add("sounddevice", "OK",
                   f"v{sd.__version__} — 입력장치 {len(input_devs)}개")
    except ImportError:
        report.add("sounddevice", "FAIL", "미설치", "pip install sounddevice")
    except Exception as e:
        report.add("sounddevice", "WARN", f"장치 조회 오류: {e}")

    # ── soundfile ───────────────────────────────────────────
    if importlib.util.find_spec("soundfile"):
        report.add("soundfile", "OK", "설치됨")
    else:
        report.add("soundfile", "FAIL", "미설치", "pip install soundfile")

    # ── pydub ───────────────────────────────────────────────
    if importlib.util.find_spec("pydub"):
        report.add("pydub", "OK", "설치됨")
    else:
        report.add("pydub", "WARN", "미설치 — m4a/mp3 변환 불가",
                   "pip install pydub")

    # ── ffmpeg ──────────────────────────────────────────────
    if shutil.which("ffmpeg"):
        report.add("ffmpeg", "OK", f"경로: {shutil.which('ffmpeg')}")
    else:
        report.add("ffmpeg", "WARN", "미설치 — m4a 변환 불가",
                   "brew install ffmpeg")

    return report


if __name__ == "__main__":
    report = run_checks()
    print(report.summary())
    if report.has_critical:
        print("\n❌ 필수 항목 오류 — 위 수정 사항 먼저 적용 후 재실행")
        sys.exit(1)
    else:
        print("\n🚀 실행 가능 상태입니다.")

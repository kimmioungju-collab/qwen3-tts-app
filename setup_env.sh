#!/bin/bash
# ============================================================
# Qwen3-TTS Mac App — 환경 구축 스크립트
# [Build Engineer] Mac M1/M2/M3 최적화
# ============================================================
set -e

echo "🔧 [1/5] ffmpeg 설치 확인..."
if ! command -v ffmpeg &>/dev/null; then
    echo "  → ffmpeg 없음. Homebrew로 설치 중..."
    brew install ffmpeg
else
    echo "  ✅ ffmpeg 이미 설치됨: $(ffmpeg -version 2>&1 | head -1)"
fi

echo ""
echo "🐍 [2/5] conda 환경 생성 (python 3.12)..."
conda create -n qwen3-tts python=3.12 -y 2>/dev/null || echo "  → 이미 존재하는 환경"

echo ""
echo "📦 [3/5] 패키지 설치..."
conda run -n qwen3-tts pip install -U pip
conda run -n qwen3-tts pip install -r requirements.txt

echo ""
echo "⚠️  [4/5] flash-attn 설치 스킵 (Mac = CUDA 없음, sdpa 사용)"

echo ""
echo "🔍 [5/5] 환경 검증..."
conda run -n qwen3-tts python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  MPS 가용: {torch.backends.mps.is_available()}')
print(f'  CPU: fallback 준비됨')
import sounddevice as sd
print(f'  sounddevice: OK ({sd.__version__})')
import PySide6
print(f'  PySide6: OK ({PySide6.__version__})')
print('')
print('✅ 환경 구축 완료!')
print('   실행: conda activate qwen3-tts && python app.py')
"

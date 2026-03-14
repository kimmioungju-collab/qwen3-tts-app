#!/bin/bash
# Qwen3-TTS Voice Studio — 실행 스크립트
set -e

cd "$(dirname "$0")"

# conda 환경 활성화 후 실행
if conda info --envs | grep -q "qwen3-tts"; then
    echo "🚀 conda env 'qwen3-tts' 로 실행..."
    conda run -n qwen3-tts python app.py
else
    echo "⚠️  conda env 'qwen3-tts' 없음 — 현재 Python으로 실행..."
    python app.py
fi

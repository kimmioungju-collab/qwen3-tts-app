#!/opt/homebrew/bin/python3
"""
Qwen3-TTS CLI
음성 복제 도구

사용법:
  tts --voice eunwoo "아빠 일어나세요"
  tts --list
  tts --add myvoice audio.wav "레퍼런스 텍스트"
  tts --model "1.7B-Base (고품질 클론)" --voice eunwoo "안녕하세요"
"""
import argparse
import json
import shutil
from pathlib import Path

import soundfile as sf

from tts_engine import Qwen3TTSEngine, MODELS

VOICES_DIR = Path(__file__).parent / "voices"
DEFAULT_MODEL = "0.6B-Base (경량 클론)"


def load_voice_config(voice_name: str) -> dict:
    """음성 설정 로드"""
    config_path = VOICES_DIR / voice_name / "config.json"
    if not config_path.exists():
        raise ValueError(f"음성 '{voice_name}'을(를) 찾을 수 없습니다")

    with open(config_path) as f:
        config = json.load(f)

    config["reference_audio"] = str(VOICES_DIR / voice_name / config["reference_audio"])
    return config


def list_voices() -> None:
    """등록된 음성 목록"""
    print("\n등록된 음성:\n")

    for voice_dir in sorted(VOICES_DIR.glob("*")):
        if not voice_dir.is_dir():
            continue
        try:
            config = load_voice_config(voice_dir.name)
            print(f"  - {config['name']:12} - {config['display_name']}")
            print(f"    {config['description']}")
        except Exception:
            pass

    print()


def list_models() -> None:
    """사용 가능한 모델 목록"""
    print("\n사용 가능한 모델:\n")
    for key, model_id in MODELS.items():
        marker = " (기본)" if key == DEFAULT_MODEL else ""
        print(f"  - {key}{marker}")
        print(f"    {model_id}")
    print()


def add_voice(name: str, audio_path: str, ref_text: str) -> None:
    """새 음성 추가"""
    voice_dir = VOICES_DIR / name
    voice_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(audio_path, voice_dir / "reference.wav")

    config = {
        "name": name,
        "display_name": name,
        "reference_audio": "reference.wav",
        "reference_text": ref_text,
        "language": "Korean",
        "description": f"{name} 목소리",
    }

    with open(voice_dir / "config.json", "w") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"음성 '{name}' 추가 완료!")


def generate_tts(
    voice_name: str,
    text: str,
    model_key: str,
    output_path: str | None = None,
) -> str:
    """TTS 생성 (Qwen3TTSEngine 사용)"""
    config = load_voice_config(voice_name)

    print(f"\nTTS 생성")
    print(f"  음성: {config['display_name']}")
    print(f"  텍스트: {text}")
    print(f"  모델: {model_key}\n")

    engine = Qwen3TTSEngine()
    print(f"디바이스: {engine.device_info()}")

    def on_progress(msg: str) -> None:
        print(f"  {msg}")

    engine.load_model(model_key, progress_cb=on_progress)

    print("\n음성 합성 중...\n")
    wav, sr = engine.generate_voice_clone(
        text=text,
        language=config.get("language", "Korean"),
        ref_audio=config["reference_audio"],
        ref_text=config["reference_text"],
    )

    if not output_path:
        output_path = f"/tmp/tts_output_{voice_name}.wav"

    sf.write(output_path, wav, sr)

    print(f"완료!")
    print(f"  저장: {output_path}\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-TTS CLI (음성 복제)")
    parser.add_argument("--voice", "-v", help="음성 이름 (eunwoo, yeonwoo, ...)")
    parser.add_argument("text", nargs="?", help="합성할 텍스트")
    parser.add_argument("--output", "-o", help="출력 파일 경로")
    parser.add_argument("--list", "-l", action="store_true", help="음성 목록")
    parser.add_argument("--models", action="store_true", help="모델 목록")
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        choices=list(MODELS.keys()),
        help=f"모델 선택 (기본: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--add",
        nargs=3,
        metavar=("NAME", "AUDIO", "TEXT"),
        help="새 음성 추가",
    )

    args = parser.parse_args()

    if args.list:
        list_voices()
        return

    if args.models:
        list_models()
        return

    if args.add:
        name, audio_path, ref_text = args.add
        add_voice(name, audio_path, ref_text)
        return

    if not args.voice or not args.text:
        parser.print_help()
        return

    generate_tts(args.voice, args.text, args.model, args.output)


if __name__ == "__main__":
    main()

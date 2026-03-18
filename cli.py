#!/opt/homebrew/bin/python3
"""
Qwen3-TTS CLI
음성 복제 도구

사용법:
  tts --voice eunwoo "아빠 일어나세요"
  tts --list
  tts --add myvoice audio.wav "레퍼런스 텍스트"
"""
import argparse
import json
import os
import shutil
from pathlib import Path
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

VOICES_DIR = Path(__file__).parent / "voices"
MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"  # 1.7B 모델 (고품질)


def load_voice_config(voice_name):
    """음성 설정 로드"""
    config_path = VOICES_DIR / voice_name / "config.json"
    if not config_path.exists():
        raise ValueError(f"음성 '{voice_name}'을(를) 찾을 수 없습니다")
    
    with open(config_path) as f:
        config = json.load(f)
    
    # 레퍼런스 오디오 경로
    config["reference_audio"] = str(VOICES_DIR / voice_name / config["reference_audio"])
    
    return config


def list_voices():
    """등록된 음성 목록"""
    print("\n📢 등록된 음성:\n")
    
    for voice_dir in sorted(VOICES_DIR.glob("*")):
        if not voice_dir.is_dir():
            continue
        
        try:
            config = load_voice_config(voice_dir.name)
            print(f"  • {config['name']:12} - {config['display_name']}")
            print(f"    {config['description']}")
        except:
            pass
    
    print()


def add_voice(name, audio_path, ref_text):
    """새 음성 추가"""
    voice_dir = VOICES_DIR / name
    voice_dir.mkdir(parents=True, exist_ok=True)
    
    # 오디오 복사
    shutil.copy2(audio_path, voice_dir / "reference.wav")
    
    # 설정 파일 생성
    config = {
        "name": name,
        "display_name": name,
        "reference_audio": "reference.wav",
        "reference_text": ref_text,
        "language": "Korean",
        "description": f"{name} 목소리"
    }
    
    with open(voice_dir / "config.json", "w") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 음성 '{name}' 추가 완료!")


def generate_tts(voice_name, text, output_path=None):
    """TTS 생성"""
    # 음성 설정 로드
    config = load_voice_config(voice_name)
    
    print(f"\n🎙️ TTS 생성")
    print(f"   음성: {config['display_name']}")
    print(f"   텍스트: {text}\n")
    
    # 모델 로드
    print(f"⏳ 모델 로드 중... ({MODEL_NAME})")
    
    # MPS 비활성화 (안정성)
    torch.backends.mps.is_available = lambda: False
    
    model = Qwen3TTSModel.from_pretrained(
        MODEL_NAME,
        device_map="cpu",
        dtype=torch.float32,
        attn_implementation="sdpa",
    )
    print("✅ 모델 로드 완료!\n")
    
    # 음성 생성
    print("🎤 음성 합성 중...\n")
    wavs, sr = model.generate_voice_clone(
        text=text,
        language=config["language"],
        ref_audio=config["reference_audio"],
        ref_text=config["reference_text"],
    )
    
    # 저장
    if not output_path:
        output_path = f"/tmp/tts_output_{voice_name}.wav"
    
    sf.write(output_path, wavs[0], sr)
    
    print(f"✅ 완료!")
    print(f"   저장: {output_path}\n")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS CLI")
    parser.add_argument("--voice", "-v", help="음성 이름 (eunwoo, yeonwoo, hymermyung)")
    parser.add_argument("text", nargs="?", help="합성할 텍스트")
    parser.add_argument("--output", "-o", help="출력 파일 경로")
    parser.add_argument("--list", "-l", action="store_true", help="음성 목록")
    parser.add_argument("--add", nargs=3, metavar=("NAME", "AUDIO", "TEXT"), help="새 음성 추가")
    
    args = parser.parse_args()
    
    # 음성 목록
    if args.list:
        list_voices()
        return
    
    # 음성 추가
    if args.add:
        name, audio_path, ref_text = args.add
        add_voice(name, audio_path, ref_text)
        return
    
    # TTS 생성
    if not args.voice or not args.text:
        parser.print_help()
        return
    
    generate_tts(args.voice, args.text, args.output)


if __name__ == "__main__":
    main()

#!/opt/homebrew/bin/python3
"""
YouTube → 레퍼런스 오디오 + STT ref_text 자동 준비

사용법:
  # YouTube URL로 직접 지정
  python3 prepare_ref.py --url "https://youtube.com/watch?v=..." --name trump

  # 검색어로 자동 찾기
  python3 prepare_ref.py --search "트럼프 한국어 더빙" --name trump

  # 시작/끝 구간 지정 (초)
  python3 prepare_ref.py --url "..." --name trump --start 5 --end 15

결과:
  voices/{name}/
    ├── reference.wav      (10~15초 레퍼런스 오디오)
    ├── config.json         (ref_text 포함)
    └── source.json         (원본 YouTube 정보)
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

VOICES_DIR = Path(__file__).parent
MAX_REF_SECONDS = 15
MIN_REF_SECONDS = 5
DEFAULT_DURATION = 10


def search_youtube(query: str, max_results: int = 5) -> list[dict]:
    """yt-dlp로 YouTube 검색, URL + 제목 반환"""
    result = subprocess.run(
        [
            "yt-dlp",
            f"ytsearch{max_results}:{query}",
            "--print", "%(id)s\t%(title)s\t%(duration)s",
            "--no-download",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        print(f"[ERROR] 검색 실패: {result.stderr}", file=sys.stderr)
        return []

    entries = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            entries.append({
                "id": parts[0],
                "title": parts[1],
                "duration": parts[2],
                "url": f"https://www.youtube.com/watch?v={parts[0]}",
            })
    return entries


def download_audio(url: str, output_path: str, start: float = 0, duration: float = DEFAULT_DURATION) -> str:
    """YouTube에서 오디오 다운로드 + 구간 추출"""
    tmp_audio = "/tmp/_yt_ref_full.wav"

    # 1) yt-dlp로 오디오 다운로드
    print(f"[1/3] 다운로드 중: {url}")
    result = subprocess.run(
        [
            "yt-dlp",
            "-x",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "-o", tmp_audio,
            "--force-overwrites",
            url,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp 실패: {result.stderr}")

    # 2) ffmpeg로 구간 추출 + mono 16kHz 변환
    print(f"[2/3] 구간 추출: {start}초 ~ {start + duration}초")
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", tmp_audio,
            "-ss", str(start),
            "-t", str(duration),
            "-ac", "1",
            "-ar", "16000",
            output_path,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 실패: {result.stderr}")

    # 정리
    Path(tmp_audio).unlink(missing_ok=True)

    # 길이 확인
    dur = get_duration(output_path)
    print(f"   → {dur:.1f}초 추출 완료")
    return output_path


def get_duration(path: str) -> float:
    """ffprobe로 오디오 길이 반환"""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True,
        text=True,
        timeout=10,
    )
    try:
        return float(result.stdout.strip())
    except (ValueError, AttributeError):
        return 0.0


def run_whisper(audio_path: str) -> str:
    """Whisper STT로 ref_text 추출"""
    print("[3/3] Whisper STT 실행 중...")
    result = subprocess.run(
        ["whisper", audio_path, "--language", "Korean", "--model", "medium", "--output_format", "txt"],
        capture_output=True,
        text=True,
        timeout=120,
    )

    texts = []
    for line in result.stdout.strip().split("\n"):
        if "-->" in line:
            parts = line.split("]", 1)
            if len(parts) > 1:
                texts.append(parts[1].strip())

    ref_text = " ".join(texts).strip()
    if not ref_text:
        # fallback: stdout 전체
        ref_text = result.stdout.strip()

    print(f"   → STT 결과: {ref_text[:80]}...")
    return ref_text


def create_voice_profile(
    name: str,
    ref_audio_path: str,
    ref_text: str,
    source_info: dict,
    display_name: str = "",
    description: str = "",
) -> Path:
    """voices/{name}/ 프로필 생성"""
    voice_dir = VOICES_DIR / name
    voice_dir.mkdir(parents=True, exist_ok=True)

    # reference.wav 복사
    import shutil
    dest_wav = voice_dir / "reference.wav"
    shutil.copy2(ref_audio_path, dest_wav)

    # config.json
    config = {
        "name": name,
        "display_name": display_name or name,
        "reference_audio": "reference.wav",
        "reference_text": ref_text,
        "language": "Korean",
        "description": description or f"{name} 목소리 (YouTube 자동 추출)",
    }
    with open(voice_dir / "config.json", "w") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # source.json (출처 기록)
    source_info["created_at"] = datetime.now().isoformat()
    source_info["ref_text"] = ref_text
    with open(voice_dir / "source.json", "w") as f:
        json.dump(source_info, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 보이스 프로필 생성: {voice_dir}")
    print(f"   ref_audio: {dest_wav}")
    print(f"   ref_text:  {ref_text[:60]}...")
    return voice_dir


def main():
    parser = argparse.ArgumentParser(description="YouTube → Voice Reference 자동 준비")
    parser.add_argument("--name", required=True, help="보이스 프로필 이름 (예: trump)")
    parser.add_argument("--url", help="YouTube URL")
    parser.add_argument("--search", help="YouTube 검색어 (URL 미지정 시)")
    parser.add_argument("--start", type=float, default=0, help="시작 시간(초)")
    parser.add_argument("--end", type=float, default=0, help="끝 시간(초). 0이면 start+10초")
    parser.add_argument("--display-name", default="", help="표시 이름")
    parser.add_argument("--description", default="", help="설명")
    args = parser.parse_args()

    url = args.url
    source_info = {}

    # URL 없으면 검색
    if not url:
        if not args.search:
            print("--url 또는 --search 중 하나는 필수입니다.", file=sys.stderr)
            sys.exit(1)

        print(f"🔍 YouTube 검색: {args.search}\n")
        results = search_youtube(args.search)
        if not results:
            print("검색 결과 없음", file=sys.stderr)
            sys.exit(1)

        for i, r in enumerate(results):
            print(f"  [{i}] {r['title']} ({r['duration']}초)")
            print(f"      {r['url']}")

        # 첫 번째 자동 선택
        url = results[0]["url"]
        source_info = results[0]
        print(f"\n→ 자동 선택: [{0}] {results[0]['title']}")
    else:
        source_info = {"url": url}

    # 구간 계산
    duration = DEFAULT_DURATION
    if args.end > 0:
        duration = min(args.end - args.start, MAX_REF_SECONDS)
    duration = max(MIN_REF_SECONDS, min(MAX_REF_SECONDS, duration))

    # 다운로드
    tmp_ref = f"/tmp/_ref_{args.name}.wav"
    download_audio(url, tmp_ref, start=args.start, duration=duration)

    # STT
    ref_text = run_whisper(tmp_ref)

    # 프로필 생성
    create_voice_profile(
        name=args.name,
        ref_audio_path=tmp_ref,
        ref_text=ref_text,
        source_info=source_info,
        display_name=args.display_name,
        description=args.description,
    )

    # 정리
    Path(tmp_ref).unlink(missing_ok=True)


if __name__ == "__main__":
    main()

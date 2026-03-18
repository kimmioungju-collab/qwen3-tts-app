# Qwen3-TTS Voice Studio

음성 복제 & TTS 생성 도구 (GUI + CLI)

## 🎯 기능

- **GUI 앱**: PySide6 기반 Mac 앱
- **CLI 도구**: 터미널에서 빠른 음성 생성
- **음성 라이브러리**: 은우, 연우, 하이머명거

## 📦 설치

```bash
cd ~/projects/qwen3-tts-app
pip3 install -r requirements.txt
```

## 🚀 사용법

### GUI 앱

```bash
~/projects/qwen3-tts-app/gui
```

또는:

```bash
cd ~/projects/qwen3-tts-app
./gui
```

### CLI

```bash
# 음성 목록
~/projects/qwen3-tts-app/tts --list

# TTS 생성
~/projects/qwen3-tts-app/tts --voice eunwoo "아빠 일어나세요"
~/projects/qwen3-tts-app/tts --voice yeonwoo "배고파요"
~/projects/qwen3-tts-app/tts --voice hymermyung "좋은 아침입니다"

# 출력 파일 지정
~/projects/qwen3-tts-app/tts --voice eunwoo "안녕" --output ~/Desktop/test.wav
```

### 별칭 설정 (추천!)

```bash
echo 'alias tts="~/projects/qwen3-tts-app/tts"' >> ~/.zshrc
echo 'alias qwen-gui="~/projects/qwen3-tts-app/gui"' >> ~/.zshrc
source ~/.zshrc
```

그럼:

```bash
tts --voice eunwoo "안녕"
qwen-gui
```

## 📁 프로젝트 구조

```
~/projects/qwen3-tts-app/
├── app.py              # GUI 메인
├── cli.py              # CLI 메인
├── tts                 # CLI 실행 스크립트
├── gui                 # GUI 실행 스크립트
├── tts_engine.py       # TTS 엔진
├── audio_utils.py      # 오디오 유틸
├── voices/
│   ├── eunwoo/         # 김은우 (큰아들)
│   ├── yeonwoo/        # 김연우 (둘째)
│   └── hymermyung/     # 하이머명거 (아빠)
└── README.md
```

## 🎤 등록된 음성

- **eunwoo** - 김은우 (큰아들)
- **yeonwoo** - 김연우 (둘째)
- **hymermyung** - 하이머명거 (아빠)

## 💡 모델

- **1.7B-Base** (기본) - 고품질 음성 복제
- **0.6B-Base** - 빠른 생성 (품질 낮음)

## 📝 예시

### CLI

```bash
# 아침 인사
tts --voice eunwoo "좋은 아침이에요, 아빠!"

# 모닝콜
tts --voice yeonwoo "일어나세요! 학교 가야해요!"

# 날씨 안내
tts --voice hymermyung "오늘은 맑고 화창한 날씨입니다"
```

### GUI

1. 앱 실행: `./gui`
2. 모델 자동 로드 대기 (1.7B-Base)
3. 텍스트 입력
4. Voice Clone 탭 → 음성 선택 (eunwoo/yeonwoo/hymermyung)
5. "음성 생성" 클릭
6. 재생!

## 🔧 기술 스택

- **모델**: Qwen3-TTS-12Hz-1.7B-Base
- **GUI**: PySide6
- **디바이스**: CPU (Apple Silicon 호환)
- **언어**: 한국어
- **출력**: WAV (44.1kHz, mono)

## 📱 Twilio 연동

전화로 음성 보내기:

```bash
cd ~/projects/qwen3-tts-app
python3 twilio_call.py --voice eunwoo "아침이에요!" --to +821077218854
```

(twilio_call.py 예정)

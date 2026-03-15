# Qwen3-TTS Voice Studio 사용 가이드

Mac M-series 최적화 TTS 앱. 목소리 복제, 커스텀 보이스, 보이스 디자인 3가지 모드 지원.

---

## 실행

```bash
cd /Users/kmj/projects/qwen3-tts-app
python3 app.py
```

또는 `./run.sh`

---

## 모델 선택

| 모델 | 용도 | 크기 |
|------|------|------|
| 0.6B-Base | 경량 Voice Clone | ~1.2GB |
| 0.6B-CustomVoice | 경량 프리셋 스피커 | ~1.2GB |
| 1.7B-Base | 고품질 Voice Clone | ~3.4GB |
| 1.7B-CustomVoice | 고품질 프리셋 스피커 | ~3.4GB |
| 1.7B-VoiceDesign | 텍스트로 목소리 디자인 | ~3.4GB |

- 처음 로드 시 HuggingFace에서 자동 다운로드
- MPS(Apple Silicon) 로드: 30초~2분

---

## 3가지 모드

### 1. Custom Voice (프리셋 스피커)

미리 정의된 스피커로 음성 합성.

1. 탭: `Custom Voice`
2. 모델: `0.6B-CustomVoice` 또는 `1.7B-CustomVoice` 로드
3. 스피커 선택: Vivian, Serena, Dylan, Eric, Ryan, Aiden, Sohee 등
4. 텍스트 입력 → `음성 생성`

### 2. Voice Design (목소리 디자인)

텍스트 설명으로 원하는 목소리를 디자인.

1. 탭: `Voice Design`
2. 모델: `1.7B-VoiceDesign` 로드
3. 디자인 설명 입력 (예: "따뜻하고 부드러운 30대 여성 목소리")
4. 텍스트 입력 → `음성 생성`

### 3. Voice Clone (목소리 복제)

3초 이상의 음성 샘플로 목소리 복제.

1. 탭: `Voice Clone`
2. 모델: `0.6B-Base` 또는 `1.7B-Base` 로드
3. 레퍼런스 오디오 입력 (3가지 방법):
   - **파일 업로드**: wav/mp3/m4a 파일
   - **YouTube URL**: URL 입력 → `YouTube 다운로드` (자동 Whisper 전사)
   - **마이크 녹음**: 마이크 버튼 클릭
4. 레퍼런스 텍스트 입력 (오디오 내용과 일치해야 함)
5. 합성할 텍스트 입력 → `음성 생성`

---

## 목소리 저장/로드

한 번 클론한 목소리를 프로필로 저장하여 재사용 가능.

### 저장

1. Voice Clone 탭에서 레퍼런스 오디오 + 텍스트 설정
2. `현재 목소리 저장` 버튼 클릭
3. 프로필 이름 입력 (예: `eunwoo`)
4. 표시 이름 입력 (예: `김은우 (큰아들)`)

### 로드

1. Voice Clone 탭 상단 `저장된 목소리` 콤보박스에서 선택
2. 레퍼런스 오디오 + 텍스트 자동 채움
3. 바로 합성 가능

### 삭제

- 콤보박스에서 프로필 선택 → 🗑 버튼 클릭

### 저장 구조

```
voices/
├── eunwoo/
│   ├── config.json       # 메타데이터
│   └── reference.wav     # 음성 샘플
├── yeonwoo/
│   ├── config.json
│   └── reference.wav
└── ...
```

---

## CLI 사용법

```bash
# 저장된 목소리로 TTS
python3 cli.py --voice eunwoo "아빠 일어나세요"

# 목소리 목록
python3 cli.py --list

# 새 목소리 추가
python3 cli.py --add myvoice audio.wav "레퍼런스 텍스트"
```

---

## 결과 내보내기

- `m4a 저장`: 생성된 음성을 m4a 파일로 다운로드
- `재생`: 앱 내에서 바로 재생

---

## 요구사항

- Python 3.11+
- PyTorch 2.3+ (MPS 지원)
- PySide6
- ffmpeg (`brew install ffmpeg`)
- yt-dlp (`brew install yt-dlp`) — YouTube 다운로드용
- openai-whisper (`pip install openai-whisper`) — 자동 전사용

---

## 지원 언어

Korean, Chinese, English, Japanese, German, French, Russian, Portuguese, Spanish, Italian

---

## 트러블슈팅

| 문제 | 해결 |
|------|------|
| MPS 세그폴트 | 모델 로드/생성은 메인 스레드에서만 실행됨 (자동 처리) |
| "Placeholder storage not allocated on MPS" | tts_engine.py에서 model.device 자동 업데이트됨 |
| OMP 충돌 | `KMP_DUPLICATE_LIB_OK=TRUE` 자동 설정됨 |
| yt-dlp 오류 | `brew install yt-dlp` 후 재시도 |
| ffmpeg 없음 | `brew install ffmpeg` |

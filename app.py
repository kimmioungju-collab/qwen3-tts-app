"""
Qwen3-TTS Mac App
[Senior Developer] PySide6 GUI — Mac M-series 최적화
공식 README 기반: Custom Voice / Voice Design / Voice Clone
"""
from __future__ import annotations

import os
# Mac: PyTorch + numpy + Qt가 각각 다른 libomp를 링크하여 충돌하는 문제 방지
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import shutil
import sys
import tempfile
import time
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFrame, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QScrollArea, QSizePolicy, QStatusBar,
    QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)

import numpy as np

from audio_utils import (
    SAMPLE_RATE, Player, Recorder, any_to_wav,
    download_youtube_audio, numpy_to_wav, transcribe_audio, wav_to_m4a,
)
from tts_engine import (
    LANGUAGES, MODE_DEFAULT_MODEL, MODE_MODELS, MODELS, SPEAKERS,
    Qwen3TTSEngine,
)


# ── 스타일 상수 ─────────────────────────────────────────────────────────────

DARK_BG = "#1E1E2E"
PANEL_BG = "#2A2A3E"
ACCENT   = "#7C6AF7"
TEXT     = "#CDD6F4"
MUTED    = "#6C7086"
SUCCESS  = "#A6E3A1"
WARNING  = "#FAB387"
DANGER   = "#F38BA8"

STYLE = f"""
QMainWindow, QWidget {{
    background-color: {DARK_BG};
    color: {TEXT};
    font-family: "Helvetica Neue";
    font-size: 13px;
}}
QTabWidget::pane {{
    border: 1px solid #45475A;
    border-radius: 8px;
    background: {PANEL_BG};
}}
QTabBar::tab {{
    background: #313244;
    color: {MUTED};
    padding: 8px 20px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
}}
QTabBar::tab:selected {{
    background: {ACCENT};
    color: white;
}}
QGroupBox {{
    border: 1px solid #45475A;
    border-radius: 8px;
    margin-top: 14px;
    padding: 12px;
    font-weight: bold;
    color: {TEXT};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 4px;
}}
QTextEdit, QLineEdit {{
    background: #181825;
    border: 1px solid #45475A;
    border-radius: 6px;
    padding: 8px;
    color: {TEXT};
}}
QComboBox {{
    background: #313244;
    border: 1px solid #45475A;
    border-radius: 6px;
    padding: 6px 10px;
    color: {TEXT};
}}
QComboBox::drop-down {{ border: none; }}
QComboBox QAbstractItemView {{
    background: #313244;
    color: {TEXT};
    selection-background-color: {ACCENT};
}}
QPushButton {{
    background: #313244;
    border: 1px solid #45475A;
    border-radius: 6px;
    padding: 8px 16px;
    color: {TEXT};
}}
QPushButton:hover {{ background: #45475A; }}
QPushButton:pressed {{ background: {ACCENT}; }}
QPushButton:disabled {{ color: {MUTED}; }}
QPushButton#primary {{
    background: {ACCENT};
    color: white;
    font-weight: bold;
    border: none;
}}
QPushButton#primary:hover {{ background: #6A58E0; }}
QPushButton#primary:disabled {{ background: #45475A; color: {MUTED}; }}
QPushButton#danger {{
    background: {DANGER};
    color: #1E1E2E;
    font-weight: bold;
    border: none;
}}
QPushButton#record_btn {{
    background: #313244;
    color: {TEXT};
    border-radius: 20px;
    min-width: 40px;
    min-height: 40px;
    font-size: 18px;
}}
QPushButton#record_btn[recording="true"] {{
    background: {DANGER};
    color: white;
}}
QProgressBar {{
    background: #313244;
    border-radius: 4px;
    text-align: center;
    color: white;
    height: 8px;
}}
QProgressBar::chunk {{
    background: {ACCENT};
    border-radius: 4px;
}}
QLabel#status_ok  {{ color: {SUCCESS}; }}
QLabel#status_warn {{ color: {WARNING}; }}
QLabel#status_err  {{ color: {DANGER}; }}
QStatusBar {{ color: {MUTED}; }}
"""


# ── 워커 ────────────────────────────────────────────────────────────────────
# NOTE: 스레드(QThread/threading) + transformers의 _load_state_dict_into_meta_model이
#       MPS에서 세그폴트 유발. 메인 스레드 동기 실행 + processEvents()로 UI 응답 유지.



# ── 탭 위젯들 ──────────────────────────────────────────────────────────────


class TextInputBox(QGroupBox):
    def __init__(self) -> None:
        super().__init__("📝 텍스트 입력")
        layout = QVBoxLayout(self)

        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText(
            "합성할 텍스트를 입력하세요...\n"
            "예) Hello, this is a test of Qwen3 TTS voice synthesis."
        )
        self.text_edit.setMinimumHeight(100)
        layout.addWidget(self.text_edit)

        lang_row = QHBoxLayout()
        lang_row.addWidget(QLabel("언어:"))
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(LANGUAGES)
        self.lang_combo.setCurrentText("Korean")
        lang_row.addWidget(self.lang_combo)
        lang_row.addStretch()
        layout.addLayout(lang_row)

    @property
    def text(self) -> str:
        return self.text_edit.toPlainText().strip()

    @property
    def language(self) -> str:
        return self.lang_combo.currentText()


class CustomVoiceTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # 화자 선택
        spk_group = QGroupBox("🎤 화자 선택")
        spk_layout = QHBoxLayout(spk_group)
        spk_layout.addWidget(QLabel("화자:"))
        self.speaker_combo = QComboBox()
        self.speaker_combo.addItems(SPEAKERS)
        spk_layout.addWidget(self.speaker_combo)
        spk_layout.addStretch()
        layout.addWidget(spk_group)

        # 스타일 지시
        style_group = QGroupBox("🎭 스타일 지시 (선택)")
        style_layout = QVBoxLayout(style_group)
        self.instruct_edit = QLineEdit()
        self.instruct_edit.setPlaceholderText(
            "예) 부드럽고 따뜻한 목소리로, 천천히 말해주세요"
        )
        style_layout.addWidget(self.instruct_edit)
        layout.addWidget(style_group)

        # 설명
        info = QLabel(
            "ℹ️ <b>CustomVoice</b> — 9가지 프리미엄 화자 + 감정/톤 제어\n"
            "모델: 0.6B-CustomVoice 또는 1.7B-CustomVoice 사용"
        )
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        layout.addWidget(info)
        layout.addStretch()

    def get_kwargs(self, text: str, language: str) -> dict:
        return {
            "text": text,
            "language": language,
            "speaker": self.speaker_combo.currentText(),
            "instruct": self.instruct_edit.text().strip(),
        }


class VoiceDesignTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        design_group = QGroupBox("🎨 보이스 디자인 설명")
        design_layout = QVBoxLayout(design_group)
        self.design_edit = QTextEdit()
        self.design_edit.setPlaceholderText(
            "원하는 목소리를 자유롭게 설명하세요.\n"
            "예) 20대 여성의 밝고 활기찬 목소리, 약간 빠른 속도"
        )
        self.design_edit.setMinimumHeight(120)
        design_layout.addWidget(self.design_edit)
        layout.addWidget(design_group)

        info = QLabel(
            "ℹ️ <b>VoiceDesign</b> — 자연어로 원하는 목소리 특성 기술\n"
            "모델: 1.7B-VoiceDesign 사용 (0.6B 없음)"
        )
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        layout.addWidget(info)
        layout.addStretch()

    def get_kwargs(self, text: str, language: str) -> dict:
        return {
            "text": text,
            "language": language,
            "instruct": self.design_edit.toPlainText().strip(),
        }


class YouTubeWorker(QThread):
    """YouTube 다운로드 + Whisper 전사를 백그라운드 스레드에서 실행."""

    progress = Signal(str)
    finished = Signal(str, str)  # (wav_path, transcribed_text)
    error = Signal(str)

    def __init__(self, url: str, max_duration: int = 30) -> None:
        super().__init__()
        self._url = url
        self._max_duration = max_duration

    def run(self) -> None:
        try:
            wav_path = download_youtube_audio(
                self._url,
                max_duration=self._max_duration,
                progress_cb=lambda msg: self.progress.emit(msg),
            )
            self.progress.emit("Whisper 전사 시작...")
            text = transcribe_audio(
                wav_path,
                progress_cb=lambda msg: self.progress.emit(msg),
            )
            self.finished.emit(str(wav_path), text or "")
        except Exception as e:
            self.error.emit(str(e))


class VoiceCloneTab(QWidget):
    ref_audio_path: str | None = None

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        self.recorder = Recorder()
        self._rec_timer = QTimer()
        self._rec_timer.timeout.connect(self._update_rec_time)
        self._rec_start = 0.0
        self._yt_worker: YouTubeWorker | None = None

        # 레퍼런스 오디오
        ref_group = QGroupBox("🎵 레퍼런스 오디오 (3초 이상 권장)")
        ref_layout = QVBoxLayout(ref_group)

        # 파일 업로드 행
        file_row = QHBoxLayout()
        self.file_label = QLabel("파일 없음")
        self.file_label.setStyleSheet(f"color: {MUTED};")
        file_row.addWidget(self.file_label, 1)
        self.upload_btn = QPushButton("📂 파일 업로드")
        self.upload_btn.clicked.connect(self._upload_file)
        file_row.addWidget(self.upload_btn)
        self.clear_btn = QPushButton("✕ 취소")
        self.clear_btn.setEnabled(False)
        self.clear_btn.clicked.connect(self._clear_file)
        file_row.addWidget(self.clear_btn)
        ref_layout.addLayout(file_row)

        # YouTube URL 입력
        yt_row = QHBoxLayout()
        self.yt_url_edit = QLineEdit()
        self.yt_url_edit.setPlaceholderText("YouTube URL 붙여넣기 (예: https://youtu.be/...)")
        yt_row.addWidget(self.yt_url_edit, 1)
        self.yt_download_btn = QPushButton("▶ YouTube 다운로드")
        self.yt_download_btn.clicked.connect(self._download_youtube)
        yt_row.addWidget(self.yt_download_btn)
        ref_layout.addLayout(yt_row)

        self.yt_status_label = QLabel("")
        self.yt_status_label.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        ref_layout.addWidget(self.yt_status_label)

        # 녹음 행
        rec_row = QHBoxLayout()
        self.record_btn = QPushButton("🎙")
        self.record_btn.setObjectName("record_btn")
        self.record_btn.setFixedSize(40, 40)
        self.record_btn.clicked.connect(self._toggle_record)
        rec_row.addWidget(self.record_btn)
        self.rec_label = QLabel("클릭하여 마이크 녹음")
        self.rec_label.setStyleSheet(f"color: {MUTED};")
        rec_row.addWidget(self.rec_label)
        rec_row.addStretch()
        ref_layout.addLayout(rec_row)
        layout.addWidget(ref_group)

        # 레퍼런스 텍스트
        text_group = QGroupBox("📄 레퍼런스 텍스트 (오디오의 내용)")
        text_layout = QVBoxLayout(text_group)
        self.ref_text_edit = QTextEdit()
        self.ref_text_edit.setPlainText(
            "저는 매일 아침 따뜻한 커피 한 잔으로 하루를 시작합니다. "
            "파란 하늘 아래 초록빛 나무들이 바람에 살랑살랑 흔들리고 있어요. "
            "혹시 내일 오후에 시간이 괜찮으신가요? 같이 점심 먹으면 좋겠는데요. "
            "정말 놀랍네요! 이렇게 빨리 완성될 줄은 전혀 몰랐습니다. "
            "가족과 함께 보내는 저녁 시간이 하루 중 가장 행복한 순간입니다."
        )
        self.ref_text_edit.setMaximumHeight(80)
        text_layout.addWidget(self.ref_text_edit)
        layout.addWidget(text_group)

        info = QLabel(
            "ℹ️ <b>VoiceClone</b> — 3초 샘플로 목소리 복제\n"
            "YouTube URL / 파일 업로드 / 마이크 녹음 → 자동 텍스트 인식"
        )
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        layout.addWidget(info)
        layout.addStretch()

    def _upload_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "레퍼런스 오디오 선택", "",
            "오디오 파일 (*.wav *.mp3 *.m4a);;모든 파일 (*)"
        )
        if not path:
            return
        try:
            wav_path = any_to_wav(path)
            self.ref_audio_path = str(wav_path)
            self.file_label.setText(Path(path).name)
            self.file_label.setStyleSheet(f"color: {SUCCESS};")
            self.clear_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.warning(self, "변환 오류", str(e))

    def _clear_file(self) -> None:
        self.ref_audio_path = None
        self.file_label.setText("파일 없음")
        self.file_label.setStyleSheet(f"color: {MUTED};")
        self.clear_btn.setEnabled(False)

    def _toggle_record(self) -> None:
        if not self.recorder.is_recording:
            self.recorder.start()
            self.record_btn.setProperty("recording", "true")
            self.record_btn.setText("⏹")
            self.record_btn.setStyle(self.record_btn.style())
            self.rec_label.setText("녹음 중... (다시 클릭하여 중지)")
            self.rec_label.setStyleSheet(f"color: {DANGER};")
            self._rec_start = time.time()
            self._rec_timer.start(1000)
        else:
            self._rec_timer.stop()
            audio = self.recorder.stop()
            self.record_btn.setProperty("recording", "false")
            self.record_btn.setText("🎙")
            self.record_btn.setStyle(self.record_btn.style())
            if len(audio) > 0:
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.close()
                import soundfile as sf
                sf.write(tmp.name, audio, SAMPLE_RATE)
                self.ref_audio_path = tmp.name
                self.rec_label.setText(f"녹음 완료 ({len(audio)/SAMPLE_RATE:.1f}초)")
                self.rec_label.setStyleSheet(f"color: {SUCCESS};")
            else:
                self.rec_label.setText("녹음 없음")
                self.rec_label.setStyleSheet(f"color: {MUTED};")

    def _download_youtube(self) -> None:
        """YouTube URL에서 오디오 다운로드 + Whisper 자동 전사 (백그라운드 스레드)."""
        url = self.yt_url_edit.text().strip()
        if not url:
            QMessageBox.warning(self, "URL 필요", "YouTube URL을 입력하세요.")
            return

        self.yt_download_btn.setEnabled(False)
        self.yt_status_label.setText("다운로드 중...")
        self.yt_status_label.setStyleSheet(f"color: {WARNING};")

        self._yt_worker = YouTubeWorker(url, max_duration=30)
        self._yt_worker.progress.connect(self._on_yt_progress)
        self._yt_worker.finished.connect(self._on_yt_finished)
        self._yt_worker.error.connect(self._on_yt_error)
        self._yt_worker.start()

    def _on_yt_progress(self, msg: str) -> None:
        self.yt_status_label.setText(msg)

    def _on_yt_finished(self, wav_path: str, text: str) -> None:
        self.ref_audio_path = wav_path
        self.file_label.setText(f"YouTube: {Path(wav_path).stem[:30]}")
        self.file_label.setStyleSheet(f"color: {SUCCESS};")
        self.clear_btn.setEnabled(True)
        self.yt_download_btn.setEnabled(True)
        if text:
            self.ref_text_edit.setPlainText(text)
            self.yt_status_label.setText("완료 — 오디오 + 텍스트 자동 입력됨")
            self.yt_status_label.setStyleSheet(f"color: {SUCCESS};")
        else:
            self.yt_status_label.setText("오디오 다운로드 완료 (텍스트 인식 실패 — 수동 입력)")
            self.yt_status_label.setStyleSheet(f"color: {WARNING};")

    def _on_yt_error(self, err: str) -> None:
        self.yt_status_label.setText(f"오류: {err}")
        self.yt_status_label.setStyleSheet(f"color: {DANGER};")
        self.yt_download_btn.setEnabled(True)

    def _update_rec_time(self) -> None:
        elapsed = int(time.time() - self._rec_start)
        self.rec_label.setText(f"🔴 녹음 중... {elapsed}초")

    def get_kwargs(self, text: str, language: str) -> dict:
        return {
            "text": text,
            "language": language,
            "ref_audio": self.ref_audio_path or "",
            "ref_text": self.ref_text_edit.toPlainText().strip(),
        }


# ── 메인 윈도우 ────────────────────────────────────────────────────────────


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.engine = Qwen3TTSEngine()
        self.player = Player()
        self._last_wav: Path | None = None

        self.setWindowTitle("Qwen3-TTS Voice Studio")
        self.setMinimumSize(800, 600)
        self.setStyleSheet(STYLE)

        self._build_ui()
        self._update_device_label()
        # 초기 탭에 맞는 모델 목록으로 시작
        self._on_tab_changed(self.tabs.currentIndex())

    # ── UI 구성 ────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # 스크롤 가능한 중앙 위젯
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setCentralWidget(scroll)

        central = QWidget()
        scroll.setWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(12)
        root.setContentsMargins(16, 16, 16, 16)

        # ── 헤더 ──────────────────────────────────────────────────────────
        header = QHBoxLayout()
        title = QLabel("🎙 Qwen3-TTS Voice Studio")
        title.setFont(QFont("-apple-system", 18, QFont.Weight.Bold))
        header.addWidget(title)
        header.addStretch()
        self.device_label = QLabel()
        self.device_label.setObjectName("status_ok")
        header.addWidget(self.device_label)
        root.addLayout(header)

        # ── 모델 선택 행 ───────────────────────────────────────────────────
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("모델:"))
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(260)
        model_row.addWidget(self.model_combo)

        self.load_btn = QPushButton("⬇ 모델 로드")
        self.load_btn.setObjectName("primary")
        self.load_btn.clicked.connect(self._load_model)
        model_row.addWidget(self.load_btn)

        self.model_status_label = QLabel("")
        self.model_status_label.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        model_row.addWidget(self.model_status_label)
        model_row.addStretch()
        root.addLayout(model_row)

        # 로드 진행바
        self.load_progress = QProgressBar()
        self.load_progress.setRange(0, 0)
        self.load_progress.setVisible(False)
        self.load_progress.setMaximumHeight(6)
        root.addWidget(self.load_progress)

        self.load_log = QLabel("")
        self.load_log.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        root.addWidget(self.load_log)

        # ── 텍스트 입력 ────────────────────────────────────────────────────
        self.text_box = TextInputBox()
        root.addWidget(self.text_box)

        # ── 모드 탭 ────────────────────────────────────────────────────────
        self.tabs = QTabWidget()
        self.custom_tab  = CustomVoiceTab()
        self.design_tab  = VoiceDesignTab()
        self.clone_tab   = VoiceCloneTab()
        self.tabs.addTab(self.custom_tab,  "🎤 Custom Voice")
        self.tabs.addTab(self.design_tab,  "🎨 Voice Design")
        self.tabs.addTab(self.clone_tab,   "🔄 Voice Clone")
        root.addWidget(self.tabs)

        # 탭 인덱스 → 모드 이름 (시그널 연결 전에 정의)
        self._tab_modes = {0: "custom", 1: "design", 2: "clone"}

        # ── 생성 버튼 ──────────────────────────────────────────────────────
        gen_row = QHBoxLayout()
        self.gen_btn = QPushButton("▶ 음성 생성")
        self.gen_btn.setObjectName("primary")
        self.gen_btn.setMinimumHeight(42)
        self.gen_btn.setEnabled(False)
        self.gen_btn.clicked.connect(self._generate)
        gen_row.addWidget(self.gen_btn, 1)

        # gen_btn 생성 후 탭 시그널 연결 & 초기 탭 설정
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.tabs.setCurrentIndex(2)  # Voice Clone 기본 선택

        self.gen_progress = QProgressBar()
        self.gen_progress.setRange(0, 0)
        self.gen_progress.setVisible(False)
        self.gen_progress.setMaximumHeight(6)
        root.addWidget(self.gen_progress)
        root.addLayout(gen_row)

        # ── 결과 ───────────────────────────────────────────────────────────
        result_group = QGroupBox("🔊 생성 결과")
        result_layout = QHBoxLayout(result_group)

        self.result_label = QLabel("아직 생성된 음성이 없습니다")
        self.result_label.setStyleSheet(f"color: {MUTED};")
        result_layout.addWidget(self.result_label, 1)

        self.play_btn = QPushButton("▶ 재생")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self._play_result)
        result_layout.addWidget(self.play_btn)

        self.download_btn = QPushButton("⬇ m4a 저장")
        self.download_btn.setEnabled(False)
        self.download_btn.clicked.connect(self._download_m4a)
        result_layout.addWidget(self.download_btn)

        root.addWidget(result_group)

        # ── 상태바 ─────────────────────────────────────────────────────────
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("모드 탭을 선택하고 모델을 로드하세요.")

    # ── 디바이스 레이블 ────────────────────────────────────────────────────

    def _update_device_label(self) -> None:
        info = self.engine.device_info()
        self.device_label.setText(f"🖥 {info}")

    # ── 탭-모델 연동 ──────────────────────────────────────────────────────

    def _current_mode(self) -> str:
        return self._tab_modes.get(self.tabs.currentIndex(), "custom")

    def _on_tab_changed(self, index: int) -> None:
        """탭 전환 시 호환 모델 목록으로 콤보 교체 + 자동 선택."""
        mode = self._tab_modes.get(index, "custom")
        compatible = MODE_MODELS.get(mode, list(MODELS.keys()))
        default = MODE_DEFAULT_MODEL.get(mode, compatible[0])

        # 로드된 모델이 새 탭에도 호환되면 그 모델을 선택, 아니면 기본값
        selected = default
        if self.engine.is_loaded:
            loaded_id = self.engine._current_model_id
            for key in compatible:
                if MODELS[key] == loaded_id:
                    selected = key
                    break

        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItems(compatible)
        self.model_combo.setCurrentText(selected)
        self.model_combo.blockSignals(False)

        # 이미 로드된 모델이 호환되는지 표시
        if self.engine.is_loaded:
            loaded_id = self.engine._current_model_id
            compatible_ids = [MODELS[k] for k in compatible]
            if loaded_id in compatible_ids:
                self.model_status_label.setText("✅ 로드된 모델 호환")
                self.model_status_label.setStyleSheet(f"color: {SUCCESS};")
                self.gen_btn.setEnabled(True)
            else:
                self.model_status_label.setText("⚠️ 로드된 모델 비호환 — 다시 로드 필요")
                self.model_status_label.setStyleSheet(f"color: {WARNING};")
                self.gen_btn.setEnabled(False)
        else:
            self.model_status_label.setText("")

    # ── 모델 로드 ──────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """메인 스레드에서 동기 로드 (스레드 + transformers = MPS 세그폴트)."""
        model_key = self.model_combo.currentText()
        self.load_btn.setEnabled(False)
        self.gen_btn.setEnabled(False)
        self.load_progress.setVisible(True)
        self.load_log.setText("로드 준비 중...")
        QApplication.processEvents()

        def _progress(msg: str) -> None:
            self.load_log.setText(msg)
            QApplication.processEvents()

        try:
            self.engine.load_model(model_key, _progress)
            # 성공 — 콤보를 로드된 모델로 유지 (_on_tab_changed 호출 시 기본값으로 리셋되므로 직접 설정)
            self.load_progress.setVisible(False)
            self.load_btn.setEnabled(True)
            self.load_log.setText(f"✅ 모델 로드 완료: {model_key}")
            self.load_log.setStyleSheet(f"color: {SUCCESS};")
            self.status_bar.showMessage("준비됨 — 텍스트 입력 후 음성 생성을 클릭하세요.")

            # 콤보를 로드된 모델로 설정 (리셋 방지)
            self.model_combo.blockSignals(True)
            self.model_combo.setCurrentText(model_key)
            self.model_combo.blockSignals(False)

            # 호환성 검증 후 생성 버튼 활성화
            mode = self._current_mode()
            compatible_ids = [MODELS[k] for k in MODE_MODELS.get(mode, [])]
            is_compatible = self.engine._current_model_id in compatible_ids
            self.gen_btn.setEnabled(is_compatible)
            if is_compatible:
                self.model_status_label.setText("✅ 로드된 모델 호환")
                self.model_status_label.setStyleSheet(f"color: {SUCCESS};")
            else:
                self.model_status_label.setText("⚠️ 로드된 모델 비호환 — 다시 로드 필요")
                self.model_status_label.setStyleSheet(f"color: {WARNING};")
        except Exception as e:
            self.load_progress.setVisible(False)
            self.load_btn.setEnabled(True)
            self.load_log.setText(f"❌ {e}")
            self.load_log.setStyleSheet(f"color: {DANGER};")
            QMessageBox.critical(self, "모델 로드 실패", str(e))

    # ── 음성 생성 ──────────────────────────────────────────────────────────

    def _generate(self) -> None:
        text = self.text_box.text
        language = self.text_box.language

        if not text:
            QMessageBox.warning(self, "입력 필요", "합성할 텍스트를 입력하세요.")
            return

        if not self.engine.is_loaded:
            QMessageBox.warning(self, "모델 미로드", "먼저 모델을 로드하세요.")
            return

        # 모델-모드 호환성 검증
        mode = self._current_mode()
        compatible_ids = [MODELS[k] for k in MODE_MODELS.get(mode, [])]
        if self.engine._current_model_id not in compatible_ids:
            mode_labels = {"custom": "Custom Voice", "design": "Voice Design", "clone": "Voice Clone"}
            compatible_names = "\n".join(f"  • {k}" for k in MODE_MODELS.get(mode, []))
            QMessageBox.warning(
                self, "모델 비호환",
                f"{mode_labels.get(mode, mode)} 모드에는 다음 모델이 필요합니다:\n\n"
                f"{compatible_names}\n\n"
                f"모델을 다시 로드해주세요.",
            )
            return

        tab = self.tabs.currentIndex()
        if tab == 0:
            mode   = "custom"
            kwargs = self.custom_tab.get_kwargs(text, language)
        elif tab == 1:
            mode   = "design"
            kwargs = self.design_tab.get_kwargs(text, language)
            if not kwargs.get("instruct"):
                QMessageBox.warning(self, "입력 필요", "보이스 디자인 설명을 입력하세요.")
                return
        else:
            mode   = "clone"
            kwargs = self.clone_tab.get_kwargs(text, language)
            if not kwargs.get("ref_audio"):
                QMessageBox.warning(self, "입력 필요",
                                    "레퍼런스 오디오를 업로드하거나 녹음하세요.")
                return

        self.gen_btn.setEnabled(False)
        self.gen_progress.setVisible(True)
        self.status_bar.showMessage("음성 생성 중... (모델 크기에 따라 수 분 소요)")
        self.result_label.setText("생성 중...")
        self.play_btn.setEnabled(False)
        self.download_btn.setEnabled(False)
        QApplication.processEvents()

        try:
            if mode == "custom":
                audio, sr = self.engine.generate_custom_voice(**kwargs)
            elif mode == "design":
                audio, sr = self.engine.generate_voice_design(**kwargs)
            elif mode == "clone":
                audio, sr = self.engine.generate_voice_clone(**kwargs)
            else:
                raise ValueError(f"알 수 없는 모드: {mode}")

            audio_arr = np.array(audio)
            self._last_wav = numpy_to_wav(audio_arr, sr)
            duration = len(audio_arr) / sr

            self.gen_progress.setVisible(False)
            self.gen_btn.setEnabled(True)
            self.result_label.setText(f"✅ 생성 완료 — {duration:.1f}초")
            self.result_label.setStyleSheet(f"color: {SUCCESS};")
            self.play_btn.setEnabled(True)
            self.download_btn.setEnabled(True)
            self.status_bar.showMessage(f"음성 생성 완료 ({duration:.1f}초)")
        except Exception as e:
            self.gen_progress.setVisible(False)
            self.gen_btn.setEnabled(True)
            self.result_label.setText("❌ 생성 실패 — 아래 오류 확인")
            self.result_label.setStyleSheet(f"color: {DANGER};")
            self.status_bar.showMessage("생성 실패")
            QMessageBox.critical(self, "생성 오류", f"음성 생성 중 오류가 발생했습니다:\n\n{e}")

    # ── 재생 & 다운로드 ────────────────────────────────────────────────────

    def _play_result(self) -> None:
        if self._last_wav:
            self.player.play(self._last_wav)

    def _download_m4a(self) -> None:
        if not self._last_wav:
            return
        if not shutil.which("ffmpeg"):
            QMessageBox.warning(self, "ffmpeg 없음",
                                "m4a 변환에 ffmpeg가 필요합니다.\n"
                                "  brew install ffmpeg  실행 후 재시도하세요.")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "m4a 저장", "output.m4a", "M4A 오디오 (*.m4a)"
        )
        if not save_path:
            return

        try:
            wav_to_m4a(self._last_wav, save_path)
            self.status_bar.showMessage(f"저장 완료: {save_path}")
        except Exception as e:
            QMessageBox.critical(self, "저장 오류", str(e))

    def closeEvent(self, event: object) -> None:
        self.player.stop()
        if self.clone_tab.recorder.is_recording:
            self.clone_tab.recorder.stop()
        super().closeEvent(event)  # type: ignore[arg-type]


# ── 진입점 ─────────────────────────────────────────────────────────────────


def main() -> None:
    # 환경 사전 체크 (GUI 실행은 차단하지 않음)
    from env_check import run_checks
    report = run_checks()
    print(report.summary())

    app = QApplication(sys.argv)
    app.setApplicationName("Qwen3-TTS Voice Studio")

    win = MainWindow()
    win.show()

    # critical 오류가 있으면 GUI 안에서 경고 표시 (앱은 띄운 상태로)
    if report.has_critical:
        missing = [r for r in report.results if r.status == "FAIL"]
        details = "\n".join(f"  • {r.name}: {r.message}\n    → {r.fix}" for r in missing)
        QMessageBox.warning(
            win, "환경 설정 필요",
            f"일부 필수 패키지가 누락되었습니다:\n\n{details}\n\n"
            f"설치 후 앱을 재시작하세요.\n"
            f"(모델 로드 시 오류가 발생할 수 있습니다)",
        )

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

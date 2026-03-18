#!/opt/homebrew/bin/python3
"""
Qwen3-TTS 웹 GUI (Gradio)
PySide6 대신 브라우저 기반 인터페이스
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import gradio as gr
from pathlib import Path
from tts_engine import Qwen3TTSEngine

# 음성 라이브러리
VOICES_DIR = Path(__file__).parent / "voices"

def load_voices():
    """등록된 음성 목록"""
    voices = {}
    for voice_dir in sorted(VOICES_DIR.glob("*")):
        if not voice_dir.is_dir():
            continue
        config_path = voice_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                voices[config["name"]] = {
                    "display_name": config["display_name"],
                    "reference_audio": str(voice_dir / config["reference_audio"]),
                    "reference_text": config["reference_text"]
                }
    return voices

VOICES = load_voices()
engine = None

def get_engine():
    global engine
    if engine is None:
        engine = Qwen3TTSEngine()
    return engine

def generate_tts(voice_name, text, model_name="0.6B-Base (경량 클론)"):
    """TTS 생성"""
    if not voice_name or not text:
        return None, "❌ 음성과 텍스트를 입력하세요"
    
    try:
        import soundfile as sf
        import tempfile
        
        eng = get_engine()
        
        # 모델 로드 (모델 변경 시에도 재로드)
        yield None, f"⏳ 모델 로드 중... ({model_name})"
        eng.load_model(model_name)
        
        yield None, "🎤 음성 합성 중..."
        
        # 음성 생성
        voice = VOICES[voice_name]
        wav, sr = eng.generate_voice_clone(
            text=text,
            language="Korean",
            ref_audio=voice["reference_audio"],
            ref_text=voice["reference_text"]
        )
        
        # 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            output_path = f.name
        sf.write(output_path, wav, sr)
        
        yield output_path, f"✅ 생성 완료! ({voice['display_name']})"
        
    except Exception as e:
        import traceback
        error_msg = f"❌ 에러:\n{str(e)}\n\n{traceback.format_exc()}"
        yield None, error_msg

# Gradio 인터페이스
with gr.Blocks(title="Qwen3-TTS Voice Studio") as demo:
    gr.Markdown("# 🎙 Qwen3-TTS Voice Studio")
    gr.Markdown("음성 복제 & TTS 생성")
    
    with gr.Row():
        with gr.Column():
            model_dropdown = gr.Dropdown(
                choices=["0.6B-Base (경량 클론)", "1.7B-Base (고품질 클론)"],
                value="0.6B-Base (경량 클론)",
                label="모델"
            )
            
            voice_dropdown = gr.Dropdown(
                choices=[(v["display_name"], k) for k, v in VOICES.items()],
                label="음성"
            )
            
            text_input = gr.Textbox(
                label="텍스트",
                placeholder="합성할 텍스트를 입력하세요...",
                lines=3
            )
            
            generate_btn = gr.Button("▶ 음성 생성", variant="primary")
        
        with gr.Column():
            status_text = gr.Textbox(label="상태", interactive=False)
            audio_output = gr.Audio(label="생성된 음성", type="filepath")
    
    generate_btn.click(
        fn=generate_tts,
        inputs=[voice_dropdown, text_input, model_dropdown],
        outputs=[audio_output, status_text]
    )
    
    gr.Markdown("---")
    gr.Markdown("**사용법:** 음성 선택 → 텍스트 입력 → 생성")

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True
    )

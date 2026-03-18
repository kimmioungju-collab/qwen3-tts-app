#!/opt/homebrew/bin/python3
"""간단한 Qwen3-TTS 웹 인터페이스"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import gradio as gr
import json
from pathlib import Path

# voices 로드
VOICES_DIR = Path(__file__).parent / "voices"
voices = {}
for voice_dir in sorted(VOICES_DIR.glob("*")):
    if voice_dir.is_dir():
        config = voice_dir / "config.json"
        if config.exists():
            with open(config) as f:
                data = json.load(f)
                voices[data["display_name"]] = {
                    "ref_audio": str(voice_dir / data["reference_audio"]),
                    "ref_text": data["reference_text"]
                }

def generate(voice, text):
    if not voice or not text:
        return None, "음성과 텍스트를 입력하세요"
    
    try:
        import torch
        torch.backends.mps.is_available = lambda: False
        
        import soundfile as sf
        from qwen_tts import Qwen3TTSModel
        
        model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            device_map="cpu",
            dtype=torch.float32,
            attn_implementation="sdpa"
        )
        
        v = voices[voice]
        wavs, sr = model.generate_voice_clone(
            text=text,
            language="Korean",
            ref_audio=v["ref_audio"],
            ref_text=v["ref_text"]
        )
        
        output = "/tmp/tts_out.wav"
        sf.write(output, wavs[0], sr)
        
        return output, f"✅ 완료! ({voice})"
    except Exception as e:
        return None, f"❌ {e}"

with gr.Blocks() as demo:
    gr.Markdown("# 🎙 Qwen3-TTS")
    
    with gr.Row():
        with gr.Column():
            voice = gr.Dropdown(choices=list(voices.keys()), label="음성")
            text = gr.Textbox(label="텍스트", lines=3)
            btn = gr.Button("생성", variant="primary")
        with gr.Column():
            status = gr.Textbox(label="상태", interactive=False)
            audio = gr.Audio(label="결과", type="filepath")
    
    btn.click(generate, inputs=[voice, text], outputs=[audio, status])

demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)

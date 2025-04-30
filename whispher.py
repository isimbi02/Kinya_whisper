import os
import torch
import torchaudio
import numpy as np
import gradio as gr
import difflib
import json
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import tempfile
import time
import shutil
import soundfile as sf
from pathlib import Path

def load_tts_safely():
    """Load TTS model with comprehensive error handling for PyTorch 2.6+"""
    print("Loading TTS model...")
    
    try:
        from TTS.api import TTS as TTSClass
        tts = TTSClass("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, 
                      torch_options={"weights_only": False})
        print("TTS model loaded successfully with weights_only=False!")
        return tts
    except Exception as e:
        print(f"Error loading TTS with weights_only=False: {str(e)}")
    
    try:
        import torch._C._onnx as _
        
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import XttsAudioConfig, Xtts, XttsArgs
        from TTS.config.shared_configs import BaseDatasetConfig, BaseAudioConfig
        from TTS.utils.audio import AudioProcessor
        from TTS.tts.utils.speakers import SpeakerManager
        
        classes_to_add = [
            XttsConfig, 
            XttsAudioConfig,
            XttsArgs,
            BaseDatasetConfig,
            Xtts,
            AudioProcessor,
            SpeakerManager,
            BaseAudioConfig
        ]
        
        print("Adding all TTS classes to safe globals...")
        torch.serialization.add_safe_globals(classes_to_add)
        
        modules_to_add = [
            'TTS.tts.configs.xtts_config',
            'TTS.tts.models.xtts',
            'TTS.config.shared_configs',
            'TTS.utils.audio',
            'TTS.tts.utils.speakers'
        ]
        
        for module_path in modules_to_add:
            module = __import__(module_path, fromlist=['*'])
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type):
                    torch.serialization.add_safe_globals([attr])
        
        from TTS.api import TTS as TTSClass
        tts = TTSClass("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)
        print("TTS loaded successfully with comprehensive safe globals!")
        return tts
    except Exception as e:
        print(f"Failed with comprehensive safe globals approach: {str(e)}")
    
    try:
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import XttsAudioConfig, Xtts, XttsArgs
        from TTS.config.shared_configs import BaseDatasetConfig, BaseAudioConfig
        
        safe_classes = [XttsConfig, XttsAudioConfig, Xtts, XttsArgs, BaseDatasetConfig, BaseAudioConfig]
        
        with torch.serialization.safe_globals(safe_classes):
            from TTS.api import TTS as TTSClass
            tts = TTSClass("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)
            print("TTS loaded successfully with context manager for safe globals!")
            return tts
    except Exception as e:
        print(f"Failed with context manager approach: {str(e)}")
    
    print("All TTS loading methods failed. Will use text-only mode without speech synthesis.")
    return None

class KinyarwandaQASystem:
    def __init__(self):
        print("Initializing Kinyarwanda Q&A System...")
        
        os.makedirs("audio_samples", exist_ok=True)
        os.makedirs("transcriptions", exist_ok=True)
        os.makedirs("answers", exist_ok=True)
        
        self.asr_model = WhisperForConditionalGeneration.from_pretrained("benax-rw/KinyaWhisper")
        self.processor = WhisperProcessor.from_pretrained("benax-rw/KinyaWhisper")
        
        if hasattr(self.asr_model.generation_config, "forced_decoder_ids"):
            self.asr_model.generation_config.forced_decoder_ids = None
        
        self.tts = load_tts_safely()
        
        self.qa_pairs = {
            "Mwiriwe neza?": "Mwiriwe, Nkufashe iki?",
            "Wiga hehe?": "Nyabihu.",
            "Ese urankunda?": "Yego!",
            
        }
        
        with open("qa_pairs.json", "w", encoding="utf-8") as f:
            json.dump(self.qa_pairs, f, ensure_ascii=False, indent=4)
        
        print("System initialized and ready!")

    def transcribe_audio(self, audio_path):
        """Transcribe audio file using KinyaWhisper"""
        print(f"Transcribing audio file: {audio_path}")
        
        waveform, sample_rate = torchaudio.load(audio_path)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        
        waveform_np = waveform.squeeze().numpy()
        
        inputs = self.processor(waveform_np, sampling_rate=sample_rate, return_tensors="pt")
        
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_features"][:, :, 0])
        
        predicted_ids = self.asr_model.generate(
            inputs["input_features"],
            attention_mask=inputs.get("attention_mask", None)
        )
        
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        base_filename = os.path.splitext(os.path.basename(audio_path))[0]
        transcription_path = f"transcriptions/{base_filename}.txt"
        with open(transcription_path, "w", encoding="utf-8") as f:
            f.write(transcription)
        
        print(f"🗣️ Transcription: {transcription}")
        return transcription

    def find_best_match(self, query):
        """Find the closest match to the query in the QA pairs"""
        if not query:
            return None
            
        matches = difflib.get_close_matches(query, self.qa_pairs.keys(), n=1, cutoff=0.6)
        
        if matches:
            best_match = matches[0]
            print(f"Matched query: '{best_match}'")
            return self.qa_pairs[best_match], best_match
        else:
            print("No match found in the QA database")
            default_response = "Mbabarira, sinumvise neza ikibazo cyawe."  # I'm sorry, I didn't understand your question
            return default_response, None

    def text_to_speech(self, text, output_path=None):
        """Convert text to speech using TTS and return the path to the audio file"""
        if not text:
            return None
            
        print(f"Converting to speech: '{text}'")
        
        if output_path is None:
            timestamp = int(time.time())
            output_path = f"answers/answer_{timestamp}.wav"
        
        if self.tts:
            try:
                reference_wav = "ref_voice.wav"
                if not os.path.exists(reference_wav):
                    print(f"Warning: Reference voice file {reference_wav} not found. Using default voice.")
                    self.tts.tts_to_file(text=text, 
                                      file_path=output_path,
                                      language="en")
                else:
                    self.tts.tts_to_file(text=text, 
                                      file_path=output_path, 
                                      speaker_wav=reference_wav,
                                      language="en")
                
                print(f"Speech saved to {output_path}")
                return output_path
            except Exception as e:
                print(f"TTS failed: {str(e)}")
                print("Falling back to text-only response")
                
                sr = 22050
                empty_audio = np.zeros(sr)
                sf.write(output_path, empty_audio, sr)
                
                return output_path
        else:
            print("TTS not available, returning empty audio")
            sr = 22050
            empty_audio = np.zeros(sr)
            sf.write(output_path, empty_audio, sr)
            
            return output_path

    def process_audio(self, audio_path):
        """Process audio file and return transcription, answer, and audio response"""
        base_filename = os.path.splitext(os.path.basename(audio_path))[0]
        sample_path = f"audio_samples/{base_filename}.wav"
        
        if audio_path != sample_path:
            data, samplerate = sf.read(audio_path)
            sf.write(sample_path, data, samplerate)
        
        transcription = self.transcribe_audio(audio_path)
        
        answer, matched_question = self.find_best_match(transcription)
        
        answer_text_path = f"answers/{base_filename}_answer.txt"
        with open(answer_text_path, "w", encoding="utf-8") as f:
            f.write(f"Question: {transcription}\n")
            f.write(f"Matched Question: {matched_question if matched_question else 'No match'}\n")
            f.write(f"Answer: {answer}")
        
        answer_audio_path = f"answers/{base_filename}_answer.wav"
        self.text_to_speech(answer, answer_audio_path)
        
        return transcription, answer, answer_audio_path, matched_question

def process_audio_file(audio_file):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        temp_audio_path = temp_audio.name
        shutil.copyfile(audio_file, temp_audio_path)
    
    transcription, answer, answer_audio, matched_question = qa_system.process_audio(temp_audio_path)
    
    matched_info = f"Matched to: '{matched_question}'" if matched_question else "No close match found"
    return transcription, matched_info, answer, answer_audio

def record_and_process(audio):
    timestamp = int(time.time())
    audio_path = f"audio_samples/recording_{timestamp}.wav"
    sf.write(audio_path, audio[1], audio[0])
    
    transcription, answer, answer_audio, matched_question = qa_system.process_audio(audio_path)
    
    matched_info = f"Matched to: '{matched_question}'" if matched_question else "No close match found"
    return transcription, matched_info, answer, answer_audio

qa_system = KinyarwandaQASystem()

with gr.Blocks(title="Kinyarwanda Speech Q&A System") as demo:
    gr.Markdown("# Kinyarwanda Speech Q&A System")
    gr.Markdown("""
    This system transcribes Kinyarwanda speech, matches it to a question in the database,
    and responds with a spoken answer. You can either upload an audio file or record directly.
    """)
    
    with gr.Tab("Upload Audio"):
        with gr.Row():
            audio_input = gr.Audio(type="filepath", label="Upload Audio File (Kinyarwanda)")
        
        submit_btn = gr.Button("Process Audio")
        
        with gr.Row():
            transcription_output = gr.Textbox(label="Transcription")
            matched_question_output = gr.Textbox(label="Matched Question")
        
        with gr.Row():
            answer_text_output = gr.Textbox(label="Answer Text")
            answer_audio_output = gr.Audio(label="Answer Audio")
        
        submit_btn.click(
            fn=process_audio_file,
            inputs=[audio_input],
            outputs=[transcription_output, matched_question_output, answer_text_output, answer_audio_output]
        )
    
    with gr.Tab("Record Audio"):
        with gr.Row():
            audio_recorder = gr.Audio(sources=["microphone"], type="numpy", label="Record Audio")
        
        record_btn = gr.Button("Process Recording")
        
        with gr.Row():
            rec_transcription_output = gr.Textbox(label="Transcription")
            rec_matched_question_output = gr.Textbox(label="Matched Question")
        
        with gr.Row():
            rec_answer_text_output = gr.Textbox(label="Answer Text")
            rec_answer_audio_output = gr.Audio(label="Answer Audio")
        
        record_btn.click(
            fn=record_and_process,
            inputs=[audio_recorder],
            outputs=[rec_transcription_output, rec_matched_question_output, rec_answer_text_output, rec_answer_audio_output]
        )
    
    gr.Markdown("""
    ## Available Questions
    
    The system can answer questions about:
    - Mwiriwe neza
    - Wiga hehe
    - Ese urankunda
    
    
    Feel free to ask these questions in Kinyarwanda!
    """)

if __name__ == "__main__":
    demo.launch()
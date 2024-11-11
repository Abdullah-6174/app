import gradio as gr
import sounddevice as sd
import whisper
from groq import Groq
from gtts import gTTS
import os

# Set up the API key for Groq
os.environ["GROQ_API_KEY"] = "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"
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load Whisper model for voice-to-text
model = whisper.load_model("base")

# Functions for chatbot pipeline
def record_audio(duration=5, sample_rate=16000):
    print("Recording...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("Recording complete")
    return recording

def transcribe_audio(audio):
    transcription = model.transcribe(audio)
    return transcription['text']

def get_groq_response(user_input):
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": user_input}],
        model="llama3-8b-8192"
    )
    return response.choices[0].message.content

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    tts.save("/app/response.mp3")  # Hugging Face will handle files under /app directory
    return "/app/response.mp3"

# Define the chatbot function for Gradio interface
def gradio_chatbot():
    # Step 1: Record voice and transcribe to text
    audio = record_audio(duration=5)
    user_text = transcribe_audio(audio)
    print("User:", user_text)
    
    # Step 2: Get response from Groq API
    response_text = get_groq_response(user_text)
    print("Bot:", response_text)
    
    # Step 3: Convert response to audio and provide playback
    response_audio_path = text_to_speech(response_text)
    return response_text, response_audio_path

# Create Gradio Interface
interface = gr.Interface(
    fn=gradio_chatbot,
    inputs=gr.Audio(type="numpy"),  # Automatically uses the device's microphone for recording
    outputs=[
        gr.Textbox(label="Chatbot Response"),  # Display chatbot's response text
        gr.Audio(type="filepath", autoplay=True)  # Automatically play response audio
    ],
    live=False,
    title="Voice-to-Voice Chatbot",
    description="Press the microphone button to start recording and interacting with the chatbot."
)

# Launch the interface
interface.launch(share=True)

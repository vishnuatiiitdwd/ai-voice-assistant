import streamlit as st
from audio_recorder_streamlit import audio_recorder
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import requests
import os
import subprocess
import warnings
import tempfile
import whisper
import torch
import soundfile as sf


# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434"

def setup_ollama():
    """Ensure required models are pulled and server is running"""
    required_models = {
        "mistral": "LLM",
        "llama2": "Alternative LLM"
    }
    
    with st.status("Setting up Ollama models...") as status:
        for model, purpose in required_models.items():
            try:
                st.write(f"Checking {model} ({purpose}) model...")
                response = requests.head(f"{OLLAMA_BASE_URL}/api/show", json={"name": model})
                if response.status_code == 404:
                    st.write(f"Pulling {model} model...")
                    subprocess.run(["ollama", "pull", model], check=True)
                status.update(label=f"{model} model ready", state="complete")
            except Exception as e:
                st.error(f"Error setting up {model}: {str(e)}")
                return False
    return True


def transcribe_with_whisper(audio_bytes):
    """Transcribe audio using local Whisper"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        # Try using the whisper Python module
        model = whisper.load_model("small")
        trans_text = model.transcribe(tmp_path)
        return trans_text["text"]

    except ImportError:
        try:
            # Fallback to CLI whisper (if installed via `pip install openai-whisper`)
            result = subprocess.run(
                ["whisper", tmp_path, "--model", "tiny", "--language", "en", "--output_format", "txt", "--output_dir", os.path.dirname(tmp_path)],
                capture_output=True,
                text=True
            )

            # Log error if subprocess fails
            if result.returncode != 0:
                raise RuntimeError(result.stderr)

            txt_path = tmp_path.replace(".wav", ".txt")
            if os.path.exists(txt_path):
                with open(txt_path, "r") as f:
                    return f.read()
            else:
                raise FileNotFoundError("Whisper CLI transcription output not found.")

        except Exception as e:
            st.error(f"Transcription failed: {str(e)}")
            return ""

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        txt_path = tmp_path.replace(".wav", ".txt")
        if os.path.exists(txt_path):
            os.unlink(txt_path)


def generate_with_ollama(prompt):
    """Generate response using updated LangChain syntax"""
    llm = Ollama(
        base_url=OLLAMA_BASE_URL,
        model="mistral",
        temperature=0.7
    )
    
    prompt_template = PromptTemplate.from_template(
        "You are a helpful AI assistant. Respond to the user in a friendly manner.\n\n"
        "User: {input}\n"
        "Assistant:"
    )
    
    chain = {"input": RunnablePassthrough()} | prompt_template | llm
    return chain.invoke(prompt)


# Load Silero TTS model once globally
device = torch.device('cpu')
model, example_text = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_tts',
    language='en',
    speaker='v3_en',
    device=device
)

def text_to_speech(text):
    """Convert text to speech using Silero TTS"""
    temp_path = None
    try:
        # Generate audio using Silero
        audio = model.apply_tts(
            text=text,
            speaker='en_53',  # You can try other speaker IDs from the Silero repo
            sample_rate=48000
        )

        # Save to a temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_path = temp_file.name
        temp_file.close()
        sf.write(temp_path, audio, 48000)

        # Return binary content
        with open(temp_path, "rb") as f:
            return f.read()

    except Exception as e:
        st.error(f"TTS failed: {str(e)}")
        return None

    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def main():
    st.title("🎙️ Local Voice Assistant")
    st.caption("A fully local assistant using Whisper + Ollama")
    
    if not setup_ollama():
        st.error("Ollama setup failed. Please ensure Ollama is running.")
        return
    
    audio_bytes = audio_recorder(
        pause_threshold=2.0,
        sample_rate=16000,
        text="Click to speak"
    )
    
    if audio_bytes:
        col1, col2 = st.columns(2)
        
        with col1:
            st.audio(audio_bytes, format="audio/wav")
            with st.spinner("Transcribing..."):
                transcription = transcribe_with_whisper(audio_bytes)
                if transcription:
                    st.text_area("You said:", transcription, height=100)
                else:
                    st.warning("Failed to transcribe audio")
        
        if transcription:
            with col2:
                with st.spinner("Generating response..."):
                    response = generate_with_ollama(transcription)
                    if response:
                        st.text_area("AI Response:", response, height=100)
                    else:
                        st.warning("Failed to generate response")
                
                if response:
                    with st.spinner("Converting to speech..."):
                        audio_response = text_to_speech(response)
                        if audio_response:
                            st.audio(audio_response, format="audio/wav")
                        else:
                            st.warning("Failed to generate speech")

if __name__ == "__main__":
    # Workaround for Streamlit event loop issue
    import asyncio
    import nest_asyncio
    nest_asyncio.apply()
    print("I have made a change")
    main()
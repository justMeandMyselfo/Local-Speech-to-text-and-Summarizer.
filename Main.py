import streamlit as st
import subprocess
import os
import tempfile
import sys
import glob
import chardet
import os

os.environ["PATH"] += os.pathsep + os.path.abspath(".")
st.set_page_config(page_title="Meeting Summarizer", layout="centered")
st.title("🎙️ Local Meeting Summarizer")
st.write("Upload a meeting audio file. This app will transcribe and summarize it entirely **offline** for privacy.")

uploaded_file = st.file_uploader("Upload audio file (.wav or .mp3)", type=["wav", "mp3"])

if uploaded_file:
    if st.button("🧠 Process Meeting"):
        with st.spinner("Saving audio file..."):
            suffix = ".mp3" if uploaded_file.name.endswith(".mp3") else ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp.flush()
                os.fsync(tmp.fileno())
                audio_path = tmp.name

            st.success(f"✅ Saved to: {audio_path}")

        if not os.path.exists(audio_path):
            st.error(f"Audio file not found at {audio_path}")
            st.stop()

        with st.spinner("Transcribing with Whisper..."):
            try:
                temp_output_dir = tempfile.gettempdir()
                python_executable = sys.executable

                result = subprocess.run(
                    [python_executable, "-m", "whisper", audio_path, "--language", "French", "--output_format", "txt", "--output_dir", temp_output_dir],
                    capture_output=True,
                    text=True
                )

                st.text("Whisper stdout:\n" + result.stdout)
                st.text("Whisper stderr:\n" + result.stderr)

                if result.returncode != 0:
                    st.error("Whisper failed:\n" + result.stderr)
                    st.stop()

                # Find latest .txt file
                txt_files = glob.glob(os.path.join(temp_output_dir, "*.txt"))
                if not txt_files:
                    st.error("No transcript file found.")
                    st.stop()

                transcript_file = max(txt_files, key=os.path.getmtime)

                # Read with encoding detection
                with open(transcript_file, 'rb') as f:
                    rawdata = f.read()
                encoding = chardet.detect(rawdata)['encoding']
                transcript = rawdata.decode(encoding or 'utf-8', errors='ignore')

                st.success("✅ Transcription complete!")

                with st.spinner("Summarizing with Mistral via Ollama..."):
                    prompt = f"""
You are a meeting assistant. Read the following transcript and summarize the key points and list the action items clearly.

Transcript:
{transcript}

Please provide:
- A bullet list of key points
- A bullet list of action items
"""

                    result = subprocess.run(
                        ["ollama", "run", "mistral"],
                        input=prompt,
                        capture_output=True,
                        text=True
                    )

                    if result.returncode != 0:
                        st.error("Ollama error:\n" + result.stderr)
                        st.stop()

                    st.subheader("📝 Summary")
                    st.code(result.stdout, language="markdown")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

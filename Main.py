import streamlit as st
import whisper
import tempfile
import os
import chardet
import subprocess

# Add local ffmpeg to PATH
ffmpeg_path = os.path.abspath("./ffmpeg.exe")
if os.path.exists(ffmpeg_path):
    os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)

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
                model = whisper.load_model("base")
                result = model.transcribe(audio_path, language="fr")

                transcript = result.get("text", "").strip()
                if not transcript:
                    st.error("❌ No transcript was generated.")
                    st.stop()

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
                st.error(f"❌ An error occurred: {str(e)}")

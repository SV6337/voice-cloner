import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import streamlit as st
import tempfile
import torch
from TTS.api import TTS
import uuid

def main():
    st.set_page_config(page_title="Kannada Voice → English Speech", page_icon="🎤")
    st.title("🎤 Kannada Voice → English Speech")
    st.write("Use your Kannada voice sample to say English text!")
    
    # Info about the process
    st.info("🎯 **Process**: Upload Kannada voice → Enter English text → Get English speech in your voice")
    
    # Upload audio (WAV is most reliable)
    audio_file = st.file_uploader(
        "Upload your Kannada voice sample (WAV format recommended)",
        type=["wav", "mp3"],
        help="Convert your Kannada MP3 to WAV online first for best results"
    )
    
    if audio_file:
        st.success(f"✅ Uploaded: {audio_file.name}")
        st.audio(audio_file)
        
        # Show file info
        file_size = len(audio_file.getvalue()) / 1024 / 1024
        st.info(f"📊 File size: {file_size:.2f} MB")
    
    # Pre-fill the birthday message
    text = st.text_area(
        "English text to synthesize",
        value="Happy Birthday Natesh!",
        height=100,
        help="The voice will say this in English using your Kannada voice characteristics"
    )
    
    # Language settings
    st.markdown("### 🌐 Language Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Voice Sample Language:**")
        st.write("🇮🇳 Kannada (your uploaded audio)")
    
    with col2:
        st.markdown("**Output Language:**")
        output_lang = st.selectbox(
            "Choose output language",
            ["en", "hi"],  # English or Hindi options
            format_func=lambda x: "🇺🇸 English" if x == "en" else "🇮🇳 Hindi",
            index=0
        )
    
    # Generate button
    if st.button("🎉 Generate  Message", disabled=not (audio_file and text.strip()), type="primary"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"💻 Using device: {device}")
        
        # Save uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav" if audio_file.type == "audio/wav" else ".mp3") as tmp:
            tmp.write(audio_file.read())
            ref_path = tmp.name
        
        # Output path
        out_path = tempfile.mktemp(suffix=".wav")
        unique_id = str(uuid.uuid4())[:8]
        
        try:
            with st.spinner("🤖 Loading voice cloning model..."):
                tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
            
            with st.spinner("🎤 Cloning your Kannada voice to say English..."):
                tts.tts_to_file(
                    text=text,
                    speaker_wav=ref_path,
                    language=output_lang,  # English output
                    file_path=out_path
                )
            
            # Read generated audio
            with open(out_path, "rb") as f:
                audio_bytes = f.read()
            
            # Success!
            st.success("🎉  message generated successfully!")
            st.balloons()  # Fun animation for birthday!
            
            # Display results
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("🎵 Your Message")
                st.audio(audio_bytes, format="audio/wav")
                
                # Audio info
                audio_size = len(audio_bytes) / 1024 / 1024
                st.info(f"📊 Generated: {audio_size:.2f} MB")
            
            with col2:
                st.subheader("💾 Download & Share")
                st.download_button(
                    "🎁 Download  Audio",
                    data=audio_bytes,
                    file_name=f"audio{unique_id}.wav",
                    mime="audio/wav",
                    use_container_width=True
                )
                
                st.markdown("### 🎊 Share Options")
                st.write("• Send via WhatsApp")
                st.write("• Share on social media") 
                st.write("• Play at birthday party!")

        except Exception as e:
            st.error(f"❌ Generation failed: {e}")
            
            # Helpful error messages
            if "format not recognised" in str(e).lower():
                st.info("💡 **Tip**: Convert your Kannada MP3 to WAV format online first")
            elif "speaker_wav" in str(e).lower():
                st.info("💡 **Tip**: Make sure your audio file has clear speech (5+ seconds)")
            else:
                st.info("💡 **Tip**: Try a different audio file or shorter text")
        
        finally:
            # Cleanup
            try:
                os.remove(ref_path)
                os.remove(out_path)
            except:
                pass

    # Help section specifically for Kannada → English
    with st.expander("📝 Instructions for Kannada Voice Cloning"):
        st.markdown("""
        ### How this works:
        1. **Upload Kannada Audio**: Your voice speaking in Kannada
        2. **AI Analysis**: The system learns your voice characteristics 
        3. **English Synthesis**: Applies your voice to English text
        4. **Result**: "Happy Birthday Natesh" in your voice, but in English!
        
        ### Tips for best results:
        - **Clear Kannada speech**: Use a recording with good pronunciation
        - **5-10 seconds**: Longer samples give better voice cloning
        - **No background noise**: Record in a quiet environment
        - **Natural speech**: Speak normally, not too fast or slow
        
        ### File format tips:
        - **Best**: WAV format (uncompressed)
        - **Okay**: High-quality MP3
        - **If MP3 fails**: Convert to WAV online first
        
        ### Language note:
        - Input: Any language (your Kannada voice)
        - Output: English or Hindi text in your voice
        """)
    
    # Example section
    with st.expander("🎤 Example Audio Samples"):
        st.markdown("""
        ### What to upload:
        **Good examples:**
        - You speaking a few sentences in Kannada clearly
        - A voice message you recorded
        - Any clear audio of your voice (language doesn't matter!)
        
        **What you'll get:**
        - The same voice saying "Happy Birthday Natesh" in English
        - Perfect for birthday wishes in your voice!
        """)

if __name__ == "__main__":
    main()

"""Browser UI (extra `ui`). The browser records the audio with `st.audio_input`, so the
microphone of the user is used, not the microphone of the server. Models load once (`st.cache_resource`).
Run: `speakright ui` or `streamlit run src/speakright/ui/streamlit_app.py`.
"""
import streamlit as st

from speakright.assess import Assessor, to_markdown
from speakright.audio import AudioError, prepare
from speakright.config import Settings
from speakright.g2p_en import EnglishG2P
from speakright.llm import build_llm
from speakright.phones import parse_phones
from speakright.pinyin import parse_pinyin
from speakright.privacy import store
from speakright.recognize import Recognition
from speakright.samples import samples


@st.cache_resource
def settings() -> Settings:
    return Settings.from_env()


@st.cache_resource
def assessor() -> Assessor:
    s = settings()
    return Assessor(llm=build_llm(s.llm_base_url, s.llm_api_key, s.llm_model), g2p=EnglishG2P(s.cmudict))


@st.cache_resource
def recognizer():
    try:
        from speakright.recognize import get_recognizer

        return get_recognizer(settings().recognizer_model, settings().device)
    except Exception as exc:  # the extra "recognizer" is not installed, or the model is not available
        return exc


st.set_page_config(page_title="speakright", page_icon="🗣️")
st.title("speakright")
st.caption("Phone-level pronunciation feedback. No recording is kept unless SPEAKRIGHT_STORE_AUDIO=yes.")
lang = st.radio("Language", ["en", "zh"], format_func={"en": "English", "zh": "Mandarin"}.get, horizontal=True)
target = st.selectbox("Practice text", samples(lang))
target = st.text_input("Or type your own text", value=target)
clip = st.audio_input("Record")
manual = st.text_input("Heard units (optional): ARPAbet for English, pinyin for Mandarin")

if st.button("Assess"):
    try:
        audio = prepare(clip.getvalue())[0] if clip is not None else None
        if manual.strip():
            units = parse_phones(manual) if lang == "en" else parse_pinyin(manual)
            rec = Recognition(tuple(units), source="manual")
        elif lang == "en" and audio is not None:
            r = recognizer()
            if isinstance(r, Exception):
                st.error(f"No phone recogniser: {r}. Install the extra 'recognizer' or type the heard phones.")
                st.stop()
            rec = r.recognize(audio, 16_000)
        else:
            st.warning("Record audio (English), or type the heard units. Mandarin needs the heard syllables.")
            st.stop()
        result = assessor().assess(lang, target, rec, audio)
        if audio is not None:
            store(audio, 16_000, settings().audio_dir, settings().store_audio, settings().retention_days)
        st.markdown(to_markdown(result))
    except (AudioError, ValueError) as exc:
        st.error(str(exc))

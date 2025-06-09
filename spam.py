import streamlit as st
import pickle
from PIL import Image

# Load model dan vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# --- UI Setup ---
st.set_page_config(page_title="Deteksi Spam", page_icon=":robot_face:", layout="centered")
st.markdown("""
    <style>
        .main {
            background-color: #f7f7f7;
            padding: 2rem;
            border-radius: 1rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }
        .stTextArea > div > textarea {
            border-radius: 0.5rem;
            font-size: 1.1rem;
        }
        .title {
            font-size: 2.5rem;
            color: #4a4a4a;
            text-align: center;
            margin-bottom: 2rem;
        }
        .result {
            font-size: 1.4rem;
            color: #2c3e50;
            text-align: center;
            background-color: #e0f7fa;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-top: 1.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown('<div class="title">\n    \U0001F916 Deteksi Pesan Spam\n</div>', unsafe_allow_html=True)

# --- Text Input ---
user_input = st.text_area("Masukkan pesan teks Anda di bawah ini:", height=150)

# --- Detection Button ---
if st.button("\u2705 Deteksi"):  # ✅
    if user_input.strip():
        input_vect = vectorizer.transform([user_input])
        prediction = model.predict(input_vect)[0]

        if prediction.lower() == "spam":
            st.markdown(f'<div class="result" style="background-color: #ffe6e6; color: #c0392b;">\n                \u26A0\ufe0f <strong>Hasil Deteksi:</strong> {prediction.upper()}\n            </div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result" style="background-color: #e6ffed; color: #27ae60;">\n                \U0001F4E2 <strong>Hasil Deteksi:</strong> {prediction.upper()}\n            </div>', unsafe_allow_html=True)
    else:
        st.warning("Silakan masukkan pesan terlebih dahulu.")

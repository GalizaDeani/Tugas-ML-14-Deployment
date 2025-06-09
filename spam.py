import streamlit as st
import pickle

# Load model dan vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.title("Deteksi Pesan Spam")

# Input dari user
user_input = st.text_area("Masukkan pesan teks:")

if st.button("Deteksi"):
    if user_input:
        input_vect = vectorizer.transform([user_input])
        prediction = model.predict(input_vect)
        st.write(f"**Hasil Deteksi:** {prediction[0]}")
    else:
        st.warning("Silakan masukkan pesan terlebih dahulu.")

import streamlit as st
import model as mod

def input_text():
    text = st.text_area('Masukkan Isi Email')
    if st.button('Prediksi') :
        return text

def main() :
    st.markdown("<h1 style='text-align: center; color: white;'>📧 Demo Aplikasi Spam Email 📧</h1>", unsafe_allow_html=True)
    st.markdown("Demo aplikasi ini digunakan untuk mendeteksi spam email dengan penambahan fitur emosi menggunakan metode LSTM (Long Short-Term Memory)")
    st.markdown(
"""
- Akurasi model yang digunakan adalah 99.1%
- Deteksi ini hanya menggunakan body email tanpa header
- Input yang digunakan pada model ini merupakan email dalam bahasa inggris
- Keluaran yang dihasilkan model terdiri dari 2 macam yakni email spam atau email ham (non-spam)
"""
    )
    text = input_text()
    if text is None:
        return

    label = mod.predict(text)
    if label == 1:
        st.warning("Email yang anda masukkan merupakan SPAM")
    elif label == -1:
        st.error("Email yang anda masukkan merupakan stopword")
    else :
        st.success("Email yang anda masukkan merupakan BUKAN SPAM")

if __name__ == '__main__':
    main()

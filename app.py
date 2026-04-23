import streamlit as st
import requests

# Takım arkadaşının hazırladığı API'nin adresi (Lokalde çalışırken genelde 8000 portudur)
# Eğer arkadaşın farklı bir port kullanıyorsa burayı güncellemelisin.
API_URL = "http://localhost:8000/api/upload"

# Sayfa Ayarları
st.set_page_config(page_title="Doküman Sohbet", page_icon="📄")
st.title("Kendi Dokümanların ile Sohbet Et 🤖")

# Sol Menü - Dosya Yükleme Alanı
with st.sidebar:
    st.header("1. Dokümanlarını Yükle")
    uploaded_files = st.file_uploader(
        "PDF, DOC, DOCX veya TXT dosyalarınızı seçin",
        type=["pdf", "docx", "txt", "doc"],
        accept_multiple_files=True 
    )

    # Dosyaları yüklemek için bir buton ekliyoruz
    if st.button("Dosyaları Gönder"):
        if uploaded_files:
            with st.spinner("Arka plana (FastAPI) gönderiliyor..."):
                # Streamlit'teki dosyaları API'nin okuyabileceği formata çeviriyoruz
                # (API birden fazla dosya beklediği için list comprehension kullanıyoruz)
                files_to_send = [
                    ("files", (file.name, file.getvalue(), file.type)) 
                    for file in uploaded_files
                ]
                
                try:
                    # Hazırlanan API'ye POST isteği atıyoruz
                    response = requests.post(API_URL, files=files_to_send)
                    
                    if response.status_code == 200:
                        st.success(f"{len(uploaded_files)} dosya başarıyla işlendi!")
                    else:
                        st.error(f"Yükleme sırasında bir hata oluştu. Hata Kodu: {response.status_code}")
                except requests.exceptions.ConnectionError:
                    st.error("API'ye ulaşılamadı. Takım arkadaşının FastAPI sunucusunun (backend) arka planda çalıştığından emin ol!")
        else:
            st.warning("Lütfen yüklemek için önce bir dosya seçin.")


# Sağ Alan - Chat Ekranı (Şimdilik iskelet olarak kalıyor, Kişi 2'yi bekliyoruz)
st.header("2. Sorularını Sor")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Dokümanlarla ilgili ne öğrenmek istersin?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Burası Kişi 2'nin RAG modeli bittiğinde gerçek API'ye bağlanacak
    asistan_cevabi = f"Sisteme henüz RAG motoru bağlanmadı. '{prompt}' sorusuna şu an cevap veremiyorum."
    
    with st.chat_message("assistant"):
        st.markdown(asistan_cevabi)
    st.session_state.messages.append({"role": "assistant", "content": asistan_cevabi})
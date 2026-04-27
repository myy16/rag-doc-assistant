import json
import os
import streamlit as st
import requests

# --- Dynamic Backend URL Discovery ---
def get_backend_url():
    """Determine the correct backend URL based on availability."""
    if os.getenv("BASE_URL"):
        return os.getenv("BASE_URL")
    
    # Try common ports: 8000 (Internal/Compose), 8010 (Agent Host Port)
    urls = ["http://localhost:8000/api", "http://localhost:8010/api", "http://backend:8000/api"]
    for url in urls:
        try:
            # Quick check (1s timeout)
            requests.get(url.replace("/api", "/"), timeout=1.0)
            return url
        except:
            continue
    return "http://localhost:8010/api" # Default fallback

BASE_URL = get_backend_url()

st.set_page_config(page_title="P2P YZTA — Expert RAG", page_icon="🧬", layout="wide")

# ── Tasarım ve CSS ────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Ana Arka Plan */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Glassmorphism Sidebar - Light & Dark support */
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.7) !important;
        backdrop-filter: blur(15px);
        border-right: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    @media (prefers-color-scheme: dark) {
        section[data-testid="stSidebar"] {
            background: rgba(20, 20, 20, 0.8) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }
        section[data-testid="stSidebar"] * {
            color: #E0E0E0 !important;
        }
        .stMarkdown p {
            color: #E0E0E0 !important;
        }
    }
    
    /* Modern Başlıklar */
    h1 {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
        letter-spacing: -1px;
    }
    
    /* Kart Yapısı */
    .stChatMessage {
        border-radius: 15px;
        margin-bottom: 10px;
        border: 1px solid rgba(0,0,0,0.05);
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }
    
    /* Akıl Yürütme Expander */
    .stExpander {
        border: none !important;
        background: rgba(30, 60, 114, 0.08) !important;
        border-radius: 12px !important;
        margin-bottom: 10px;
    }
    
    /* Buton İyileştirmeleri */
    .stButton>button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        border: 1px solid rgba(0,0,0,0.1) !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        background-color: #1e3c72 !important;
        color: white !important;
    }
    
    /* Custom Title Bar */
    .expert-badge {
        background: #1e3c72;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        vertical-align: middle;
        margin-left: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# ── Session state başlangıç ───────────────────────────────────────────────────
for key, default in [
    ("messages", []),
    ("uploaded_files_info", []),
    ("username", ""),
    ("username_set", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Kullanıcı adı girişi ──────────────────────────────────────────────────────
def _load_user_files(username: str):
    """Backend'den kullanıcının daha önce yüklediği dosyaları çek."""
    try:
        resp = requests.get(f"{BASE_URL}/files", params={"username": username}, timeout=30)
        if resp.status_code == 200:
            return resp.json().get("files", [])
    except requests.exceptions.ConnectionError:
        pass
    except requests.exceptions.Timeout:
        pass
    except requests.exceptions.RequestException:
        pass
    return []


if not st.session_state.username_set:
    st.title("Kendi Dokümanların ile Sohbet Et 📄")
    st.markdown("### Başlamak için kullanıcı adını gir")
    col1, col2 = st.columns([3, 1])
    with col1:
        name_input = st.text_input("Kullanıcı adı", label_visibility="collapsed")
    with col2:
        if st.button("Giriş Yap", use_container_width=True) and name_input.strip():
            st.session_state.username = name_input.strip()
            st.session_state.username_set = True
            st.session_state.uploaded_files_info = []
            st.rerun()
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"**Kullanıcı:** {st.session_state.username}")
    if st.button("Çıkış Yap", use_container_width=True):
        for key in ["messages", "uploaded_files_info", "username", "username_set"]:
            st.session_state[key] = [] if key in ("messages", "uploaded_files_info") else ("" if key == "username" else False)
        st.rerun()

    if st.button("Kayıtlı dosyaları yükle", use_container_width=True):
        try:
            with st.spinner("Kayıtlı dosyalar alınıyor..."):
                st.session_state.uploaded_files_info = _load_user_files(st.session_state.username)
            if not st.session_state.uploaded_files_info:
                st.info("Bu kullanıcı için kayıtlı dosya bulunamadı.")
            st.rerun()
        except Exception:
            st.warning("Kayıtlı dosyalar şu anda alınamadı. Sonradan tekrar deneyebilirsin.")

    st.divider()
    st.header("1. Doküman Yükle")
    uploaded_files = st.file_uploader(
        "PDF, DOC, DOCX veya TXT",
        type=["pdf", "docx", "txt", "doc"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if st.button("Yükle ve İndeksle", disabled=not uploaded_files, use_container_width=True):
        files_to_send = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]
        try:
            stage_box = st.empty()
            progress_box = st.empty()
            progress_bar = progress_box.progress(0.0)
            completed = 0
            total_files = len(uploaded_files)
            uploaded_results = []
            upload_errors = []

            with st.spinner("İşleniyor..."):
                with requests.post(
                    f"{BASE_URL}/upload/stream",
                    files=files_to_send,
                    data={"username": st.session_state.get("username", "")},
                    stream=True,
                    timeout=(10, None),
                ) as resp:
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        line = line.decode("utf-8")
                        if not line.startswith("data:"):
                            continue
                        raw = line[len("data:"):].strip()
                        try:
                            chunk = json.loads(raw)
                        except Exception:
                            continue

                        event_type = chunk.get("event")
                        if event_type == "stage":
                            stage_box.info(f"**{chunk.get('filename', '')}**: {chunk.get('stage', '')}")
                        elif event_type == "file_done":
                            uploaded_results.append(chunk.get("file", {}))
                            completed += 1
                            progress_bar.progress(min(completed / total_files, 1.0))
                        elif event_type == "error":
                            detail = f"{chunk.get('filename', '')}: {chunk.get('detail', '')}"
                            upload_errors.append(detail)
                            stage_box.error(detail)
                        elif event_type == "done":
                            progress_bar.progress(1.0)

            if uploaded_results:
                existing_names = {f["original_name"] for f in st.session_state.uploaded_files_info}
                added = 0
                for fi in uploaded_results:
                    if fi.get("original_name") not in existing_names:
                        st.session_state.uploaded_files_info.append({
                            "file_id": fi["file_id"],
                            "original_name": fi["original_name"],
                            "chunk_count": fi["chunk_count"],
                            "size_mb": fi["size_mb"],
                        })
                        added += 1
                st.success(f"{added} dosya yüklendi!")
                st.rerun()
            elif upload_errors:
                st.error(f"{len(upload_errors)} dosya islenemedi. Detaylar yukarida.")
            else:
                st.warning("Yükleme tamamlandı ama işlenecek dosya döndürülmedi.")
        except requests.exceptions.Timeout:
            st.error("Yükleme uzun sürdü. Backend hâlâ indeksliyor olabilir; lütfen biraz sonra tekrar dene.")
        except requests.exceptions.ChunkedEncodingError:
            st.error(
                "Upload stream beklenmedik şekilde kesildi. "
                "Backend logunu kontrol edip tekrar deneyin; dosyaların bir kısmı işlenmiş olabilir."
            )
        except requests.exceptions.ConnectionError:
            st.error(f"Backend'e ulaşılamadı. Lütfen sunucunun çalıştığından emin olun. (Denenen URL: {BASE_URL})")

    # Yüklü dosyalar listesi + silme
    if st.session_state.uploaded_files_info:
        st.divider()
        st.subheader("Yüklü Dosyalar")
        file_options = {"Tüm Dosyalar": None}
        for f in st.session_state.uploaded_files_info:
            file_options[f["original_name"]] = f["original_name"]

        selected_label = st.selectbox("Sorgulama kapsamı:", list(file_options.keys()))
        selected_source = file_options[selected_label]

        for f in st.session_state.uploaded_files_info:
            col_name, col_del = st.columns([5, 1])
            with col_name:
                st.markdown(f"📄 **{f['original_name']}**  \n"
                            f"<small>{f['chunk_count']} chunk · {f['size_mb']} MB</small>",
                            unsafe_allow_html=True)
            with col_del:
                if st.button("🗑", key=f"del_{f['file_id']}", help="Sil"):
                    try:
                        r = requests.delete(f"{BASE_URL}/upload", json={"file_id": f["file_id"]})
                        if r.status_code == 200:
                            st.session_state.uploaded_files_info = [
                                x for x in st.session_state.uploaded_files_info
                                if x["file_id"] != f["file_id"]
                            ]
                            st.rerun()
                        else:
                            st.error(f"Silinemedi: {r.status_code}")
                    except requests.exceptions.ConnectionError:
                        st.error("Backend'e ulaşılamadı.")

        st.divider()
        if st.button("Seçili Dosyayı Özetle", use_container_width=True):
            payload = {"max_chunks": 8, "username": st.session_state.get("username") or None}
            if selected_source:
                payload["source_file"] = selected_source
            try:
                with st.spinner("Özetleniyor..."):
                    r = requests.post(f"{BASE_URL}/summarize", json=payload)
                if r.status_code == 200:
                    summary_text = r.json().get("summary", "Özet alınamadı.")
                    label = selected_label if selected_label != "Tüm Dosyalar" else "Tüm Dosyalar"
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"**Özet — {label}:**\n\n{summary_text}",
                        "sources": [],
                    })
                    st.rerun()
                else:
                    st.error(f"Özetleme hatası: {r.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("Backend'e ulaşılamadı.")
    else:
        selected_source = None


# ── Ana alan ──────────────────────────────────────────────────────────────────
header_col, clear_col = st.columns([6, 1])
with header_col:
    st.title(f"Merhaba, {st.session_state.username} 👋")
with clear_col:
    if st.button("Sohbeti Temizle", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Geçmiş mesajlar
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            source_names = list({s.get("source_file", "") for s in msg["sources"] if s.get("source_file")})
            st.caption("Kaynak: " + " · ".join(f"📄 {n}" for n in source_names))

# Yeni soru
if prompt := st.chat_input("Dokümanlarla ilgili ne öğrenmek istersin?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})

    if not st.session_state.uploaded_files_info:
        # Login akısında dosyalar bilincli olarak lazy yukleniyor; chat'te bir kez daha dene.
        st.session_state.uploaded_files_info = _load_user_files(st.session_state.get("username", ""))

    if not st.session_state.uploaded_files_info:
        reply = "Henüz doküman görünmüyor. Sol panelden dosya yükleyebilir veya 'Kayıtlı dosyaları yükle' butonunu kullanabilirsin."
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply, "sources": []})
    else:
        payload = {"question": prompt, "top_k": 5, "username": st.session_state.get("username") or None}
        if selected_source:
            payload["source_file"] = selected_source

        try:
            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_response = ""
                sources = []
                evaluation = None
                retrieval = None

                with st.spinner("🔍 Dökümanlar taranıyor ve analiz ediliyor..."):
                    with requests.post(
                        f"{BASE_URL}/chat/stream",
                        json=payload,
                        stream=True,
                        timeout=60,
                    ) as r:
                        for line in r.iter_lines():
                            if not line:
                                continue
                            line = line.decode("utf-8")
                            if not line.startswith("data:"):
                                continue
                            raw = line[len("data:"):].strip()
                            try:
                                chunk = json.loads(raw)
                            except Exception:
                                continue

                            if chunk.get("type") == "token":
                                full_response += chunk.get("content", "")
                                
                                # Master Level: Clean technical citations on-the-fly for cleaner UI
                                import re
                                display_text = re.sub(r"\[(?:Source|Kaynak|Doc|Doküman):\s*[^\]]+\]", "", full_response)
                                
                                if "Nihai Cevap:" in display_text:
                                    display_text = display_text.split("Nihai Cevap:", 1)[1]
                                elif "Düşünce Süreci:" in display_text:
                                    display_text = "*(Analiz ediliyor...)*"
                                
                                placeholder.markdown(display_text.strip() + "▌")
                            elif chunk.get("type") == "sources":
                                sources = chunk.get("content", [])
                            elif chunk.get("type") == "retrieval":
                                retrieval = chunk.get("content", {})
                            elif chunk.get("type") == "evaluation":
                                evaluation = chunk.get("content", {})
                            elif chunk.get("type") == "error":
                                full_response = f"Hata: {chunk.get('detail', 'Bilinmeyen hata')}"

                # Final formatting (Hide thoughts, show only final answer)
                import re
                final_main_text = re.sub(r"\[(?:Source|Kaynak|Doc|Doküman):\s*[^\]]+\]", "", full_response)
                if "Nihai Cevap:" in final_main_text:
                    final_main_text = final_main_text.split("Nihai Cevap:", 1)[1].split("Sources Table:", 1)[0].split("Kaynak Tablosu:", 1)[0].strip()
                elif "Final Answer:" in final_main_text:
                    final_main_text = final_main_text.split("Final Answer:", 1)[1].split("Sources Table:", 1)[0].split("Kaynak Tablosu:", 1)[0].strip()

                placeholder.markdown(final_main_text.strip())

                if sources:
                    source_names = list({s.get("source_file", "") for s in sources if s.get("source_file")})
                    st.caption("Kaynak: " + " · ".join(f"📄 {n}" for n in source_names))


            st.session_state.messages.append({
                "role": "assistant",
                "content": final_main_text,
                "sources": sources,
                "evaluation": evaluation,
                "retrieval": retrieval,
            })

        except requests.exceptions.ConnectionError:
            err = "Backend'e ulaşılamadı (port 8000)."
            with st.chat_message("assistant"):
                st.markdown(err)
            st.session_state.messages.append({"role": "assistant", "content": err, "sources": []})

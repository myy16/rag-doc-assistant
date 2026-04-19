# P2P_YZTA — Kendi Dokümanlarınla Sohbet Et

RAG (Retrieval-Augmented Generation) mimarisi üzerine kurulu, doküman yükleme, metin çıkarma, temizleme, chunking ve yapay zeka ile soru-cevap yapabilen bir backend uygulaması.

---

## Kurulum ve Çalıştırma

### 1. Yerel Kurulum (venv)

**Gereksinimler:** Python 3.9+, pip, antiword (DOC dosyaları için)

```bash
# antiword kur (macOS)
brew install antiword

# Projeyi klonla
git clone https://github.com/myy16/P2P_YZTA.git
cd P2P_YZTA/backend

# Sanal ortam oluştur ve aktifleştir
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Bağımlılıkları kur
pip install -r requirements.txt

# Sunucuyu başlat
uvicorn main:app --reload
```

Swagger UI: http://127.0.0.1:8000/docs

---

### 2. Docker ile Çalıştırma

**Gereksinim:** Docker Desktop

```bash
# Proje kök dizininden
cd P2P_YZTA
docker-compose up --build
```

Swagger UI: http://localhost:8000/docs

Durdurmak için:
```bash
docker-compose down
```

Chroma vektör verisi Docker volume içinde tutulur ve `backend/data/chroma` altında kalıcıdır.

---

## API Endpoints

### POST /api/upload

Bir veya birden fazla doküman yükler. Sonuçları tümünü birden döner.

**Desteklenen formatlar:** PDF, DOCX, DOC, TXT
**Maksimum dosya boyutu:** 20 MB

**Örnek response:**
```json
{
  "uploaded_files": [
    {
      "file_id": "uuid",
      "original_name": "rapor.pdf",
      "file_type": "pdf",
      "size_mb": 0.057,
      "extracted_text": "...",
      "chunks": [
        {
          "chunk_id": "uuid",
          "file_id": "uuid",
          "source_file": "rapor.pdf",
          "file_type": "pdf",
          "chunk_index": 0,
          "total_chunks": 7,
          "text": "...",
          "char_count": 491
        }
      ],
      "chunk_count": 7
    }
  ],
  "count": 1
}
```

---

### POST /api/upload/stream

Aynı işlemi Server-Sent Events (SSE) ile yapar. Her dosya işlenince anlık event gönderir.

**Event formatı:**
```
data: {"event": "file_done", "file": {...}}
data: {"event": "error", "filename": "...", "detail": "..."}
data: {"event": "done", "count": 1}
```

---

### POST /api/chat

Yüklenen dokümanlara göre soru-cevap üretir. Kaynakları da response içinde döner.

**Örnek request:**
```json
{
  "question": "Bu dokümanda ana konu nedir?",
  "top_k": 5
}
```

### POST /api/summarize

Seçili doküman veya tüm indeks üzerinden özet üretir.

**Örnek request:**
```json
{
  "source_file": "rapor.pdf",
  "max_chunks": 8
}
```

---

## Doküman İşleme Pipeline

Yüklenen her dosya şu adımlardan geçer:

```
Yükleme → Parse → Temizleme → Chunking → Response
```

| Adım | Modül | Açıklama |
|------|-------|----------|
| Parse | `app/core/parser.py` | PDF (pdfplumber), DOCX/DOC (python-docx, antiword), TXT (encoding detection) |
| Temizleme | `app/core/cleaner.py` | Kontrol karakterleri, Unicode NFC, header/footer pattern'ları, fazla boşluk |
| Chunking | `app/core/chunker.py` | Recursive character splitting, chunk_size=500, overlap=50 |

---

## Testleri Çalıştırma

```bash
cd backend
source venv/bin/activate
pytest tests/test_chunker.py -v
```

---

## Proje Yapısı

```
P2P_YZTA/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── upload.py        # Upload ve streaming endpoint'leri
│   │   └── core/
│   │       ├── config.py        # Uygulama ayarları
│   │       ├── parser.py        # Doküman parser'ları
│   │       ├── cleaner.py       # Metin temizleme
│   │       └── chunker.py       # Chunking stratejisi
│   ├── tests/
│   │   └── test_chunker.py      # Unit testler
│   ├── main.py                  # FastAPI uygulama giriş noktası
│   ├── requirements.txt
│   └── Dockerfile
└── docker-compose.yml
```

---

# P2P_YZTA
🛠️ GitHub Commit Format: (type) scope : description
Bu format, dokümanların nasıl işlendiğini, vektörleştiğini ve yapay zeka tarafından nasıl anlamlandırıldığını adım adım izlememizi sağlar.

Types
(feat): Yeni özellikler veya arayüz bileşenleri (PDF yükleme, chat ekranı vb.).

(fix): Hata düzeltmeleri (dosya okuma hatası, LLM bağlantı kopması vb.).

(style): Sadece görsel değişiklikler (CSS, buton renkleri, chat balonları).

(refactor): RAG mantığını veya kod yapısını iyileştirme (chunking stratejisi değişimi).

(chore): API anahtarları, kütüphane kurulumları veya ayar dosyaları.

(rag): Embedding, vektör veritabanı veya doküman işleme süreçlerine özel güncellemeler.

(docs): README, kurulum kılavuzu veya yorum satırı eklemeleri.

RAG Projesi İçin Örnekler
(feat) upload : add multi-format file support for PDF and DOCX uploads

(rag) chunking : implement recursive character splitting for better context

(rag) vector-db : integrate FAISS for efficient similarity search

(fix) parser : resolve encoding issues while reading Turkish characters in TXT files

(refactor) prompt : optimize system prompt to include source citations in responses

(data) cleaning : remove redundant white spaces and headers from parsed text

(chore) deps : add langchain and groq-sdk to requirements.txt

(style) chat : apply scrolling effect to chat window for long conversations

(docs) readme : add architecture diagram and local setup instructions

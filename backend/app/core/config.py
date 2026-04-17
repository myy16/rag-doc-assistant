ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "doc"}
MAX_FILE_SIZE_MB = 20
UPLOAD_DIR = "uploads"

# Text cleaning
MAX_CONSECUTIVE_BLANK_LINES = 2  # collapse runs longer than this

# Chunking
CHUNK_SIZE = 500        # max characters per chunk
CHUNK_OVERLAP = 50      # characters shared between consecutive chunks

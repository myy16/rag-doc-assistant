import re
import unicodedata


def clean_text(raw_text: str) -> str:
    """
    Clean raw extracted text for downstream processing (chunking, embedding).

    Steps applied in order:
    1. Remove null bytes and non-printable control characters
    2. Normalize Unicode to NFC (handles Turkish characters correctly)
    3. Strip common header/footer patterns (page numbers, repeated short lines)
    4. Collapse excessive whitespace and blank lines

    Args:
        raw_text: Raw text extracted from a document parser.

    Returns:
        Cleaned plain text string.
    """
    if not raw_text:
        return ""

    text = _remove_control_characters(raw_text)
    text = _normalize_unicode(text)
    text = _remove_header_footer_patterns(text)
    text = _collapse_whitespace(text)

    return text.strip()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _remove_control_characters(text: str) -> str:
    """Remove null bytes and non-printable control characters (keep newlines/tabs)."""
    # Keep: tab (0x09), newline (0x0A), carriage return (0x0D)
    # Remove: all other C0/C1 control chars, null bytes, form feeds, etc.
    return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\x80-\x9F]", "", text)


def _normalize_unicode(text: str) -> str:
    """Normalize Unicode to NFC form (composes decomposed chars like Turkish ş, ğ, ı)."""
    return unicodedata.normalize("NFC", text)


def _remove_header_footer_patterns(text: str) -> str:
    """
    Remove common header/footer patterns found in PDFs and Word documents:
    - Standalone page numbers:  "Page 3", "Sayfa 3", "- 3 -", lone digits on a line
    - Lines that are purely decorative (dashes, underscores, equals signs)
    - Lines shorter than 3 chars that appear alone (likely artifacts)
    """
    lines = text.splitlines()
    cleaned = []

    page_number_pattern = re.compile(
        r"^\s*("
        r"page\s+\d+"           # "Page 3"
        r"|sayfa\s+\d+"         # "Sayfa 3" (Turkish)
        r"|-\s*\d+\s*-"         # "- 3 -"
        r"|\d+\s*/\s*\d+"       # "3 / 10"
        r"|\d+"                 # lone digit(s)
        r")\s*$",
        re.IGNORECASE,
    )

    decorative_line_pattern = re.compile(r"^\s*[-_=*#~]{3,}\s*$")

    for line in lines:
        stripped = line.strip()
        if page_number_pattern.match(stripped):
            continue
        if decorative_line_pattern.match(stripped):
            continue
        cleaned.append(line)

    return "\n".join(cleaned)


def _collapse_whitespace(text: str) -> str:
    """
    - Replace multiple spaces/tabs on a single line with a single space
    - Collapse 3+ consecutive blank lines into at most 2
    - Strip trailing whitespace from each line
    """
    # Strip trailing spaces on each line
    lines = [line.rstrip() for line in text.splitlines()]

    # Collapse runs of blank lines (>2 consecutive) into 2
    result = []
    blank_count = 0
    for line in lines:
        if line == "":
            blank_count += 1
            if blank_count <= 2:
                result.append(line)
        else:
            blank_count = 0
            # Collapse multiple inline spaces/tabs to single space
            result.append(re.sub(r"[ \t]+", " ", line))

    return "\n".join(result)
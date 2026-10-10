from typing import list
import logfire

# split the text into chunks of size of 1500 characters each
def chunk_text(text:str,chunk_size:int = 1500)->List[str]:
    """
    Simple semantic chuker that splits by paragraphs ensures chunks do not exced the specifik size
    """
    with logfire.span("Text Chunking",text_lenght = len(text)):
        if not text.strip():
            return []

        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for p in paragraphs:
            if len(current_chunk) + len(p) <= chunk_size:
                current_chunk += "\n\n" + p
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = p + "\n\n"

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        valid_chunks = [ch for ch in chunks if ch.strip()]

        logfire.info("Text Chunking", chunks = len(valid_chunks))
        return valid_chunks
        
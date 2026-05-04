"""Loader for PDF files using pypdf."""

from pathlib import Path

from financial_rag.ingestion.base import BaseLoader
from financial_rag.ingestion.models import Document


class PDFLoader(BaseLoader):
    """Loads a PDF file, producing one Document per page.

    One-document-per-page is a deliberate choice: it preserves page-level
    provenance so citations can reference exact page numbers, and it keeps
    individual documents small enough for chunking to work predictably.
    """

    def load(self) -> list[Document]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError("Install pypdf: pip install pypdf") from exc

        reader = PdfReader(str(self.path))
        documents: list[Document] = []

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()

            if not text:
                # Scanned page or image-only — skip with a warning
                print(f"  [warning] Page {page_num + 1} of {self.path.name} has no extractable text (may be scanned).")
                continue

            documents.append(
                Document(
                    content=text,
                    source=str(self.path),
                    page=page_num,
                    metadata={
                        "filename": self.path.name,
                        "extension": ".pdf",
                        "total_pages": len(reader.pages),
                        "size_bytes": self.path.stat().st_size,
                    },
                )
            )

        if not documents:
            raise ValueError(
                f"No extractable text found in {self.path.name}. "
                "The PDF may be scanned. Consider an OCR pre-processing step."
            )

        return documents

import base64
import fitz  # PyMuPDF


def pdf_to_page_images(pdf_path: str, dpi: int = 150) -> list[dict]:
    """
    Render each page of a PDF to a PNG image and encode it as base64.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Resolution for rendering (default 150 DPI is a good balance
             between quality and payload size for vision models).

    Returns:
        A list of dicts, one per page:
            {
                "page_num": int,          # 1-based
                "base64_image": str       # base64-encoded PNG bytes
            }
    """
    doc = fitz.open(pdf_path)
    results = []

    # fitz uses a zoom matrix: zoom = dpi / 72 (72 is the default PDF DPI)
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    for page_index in range(len(doc)):
        page = doc[page_index]
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        png_bytes = pixmap.tobytes("png")
        encoded = base64.b64encode(png_bytes).decode("utf-8")
        results.append({
            "page_num": page_index + 1,
            "base64_image": encoded,
        })

    doc.close()
    return results

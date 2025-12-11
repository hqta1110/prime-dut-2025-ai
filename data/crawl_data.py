import os
from docx import Document
from striprtf.striprtf import rtf_to_text
import pdfplumber
import fitz
import shutil
import easyocr
reader = easyocr.Reader(['vi'])

def pdf_to_images_pymupdf(pdf_path, dpi=200):
    doc = fitz.open(pdf_path)
    images = []

    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=matrix)

        img_bytes = pix.tobytes("png")
        images.append(img_bytes)

    os.makedirs(os.path.splitext(pdf_path)[0], exist_ok=True)
    for i, data in enumerate(images):
        with open(os.path.join(os.path.splitext(pdf_path)[0], f"page_{i+1}.png"), "wb") as f:
            f.write(data)

def extract_text(path):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".docx":
        doc = Document(path)
        return "\n".join([p.text for p in doc.paragraphs])

    elif ext == ".rtf":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return rtf_to_text(f.read())

    elif ext == ".pdf":
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        if len(text) > 0:
            return text
        else:
            folder_path = os.path.splitext(path)[0]
            pdf_to_images_pymupdf(path)
            result = reader.readtext([os.path.join(folder_path, file) for file in os.listdir(folder_path)][0])

            shutil.rmtree(folder_path)
            return result
            
    else:
        raise ValueError(f"Unsupported file type: {ext}")
print(extract_text("Hien_phap_Viet_Nam/test.pdf"))

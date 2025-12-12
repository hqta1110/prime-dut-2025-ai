# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import os
import json
import re
import time
import shutil
from PyPDF2 import PdfReader
from google import genai
from google.genai import types

def get_pdf_page_count(pdf_path: str) -> int:
    try:
        reader = PdfReader(pdf_path)
        return len(reader.pages)
    except Exception as e:
        print(f"⚠️ Cannot read page count: {pdf_path} - {e}")
        return -1

def parse_json(result_text: str):
    if not result_text:
        raise ValueError("Empty result from model")

    # Loại bỏ ```json ``` nếu có
    cleaned = re.sub(r"```json|```", "", result_text).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON returned by model:\n{cleaned}") from e


def generate():
    client = genai.Client(
        # api_key=""
    )
    pdf_path = "D:\\selflearning\\vnpt-ai-infor\\analysize\\pdf_files\\[SCAN]_01-tt-vpcp.signed.pdf"
    output_path = "D:\\selflearning\\vnpt-ai-infor\\analysize\\output\\01-tt-vpcp.json"


    pdf_dir = "D:\\other\\prime-dut-2025-ai\\crawled_decree"
    output_dir = "D:\\other\\prime-dut-2025-ai\\crawled_decree\\output"
    skipped_dir = "D:\\other\\prime-dut-2025-ai\\crawled_decree\\skipped_large_pdf"


    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(skipped_dir, exist_ok=True)
    # with open(pdf_path, "rb") as f:
    #     pdf_bytes = f.read()

    model = "gemini-flash-latest"

    pdf_files = [
        f for f in os.listdir(pdf_dir)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print("❌ No PDF files found")
        return
    
    for file_name in pdf_files:
        pdf_path = os.path.join(pdf_dir, file_name)
        json_name = file_name.replace(".pdf", ".json")
        output_path = os.path.join(output_dir, json_name)

        print(f"\n📄 Checking: {file_name}")

        # ✅ CHECK 1: đã xử lý rồi thì skip
        if os.path.exists(output_path):
            print("⏭️ Already processed, skipping.")
            continue

        # ✅ CHECK 2: PDF > 40 trang thì copy & skip
        page_count = get_pdf_page_count(pdf_path)
        if page_count == -1:
            print("⏭️ Cannot determine page count, skipping.")
            continue

        if page_count >= 40:
            shutil.copy2(pdf_path, os.path.join(skipped_dir, file_name))
            print(f"📦 Skipped ({page_count} pages) → moved to skipped_large_pdf/")
            continue

        print(f"📑 Pages: {page_count} → Processing")

        try:
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(
                            mime_type="application/pdf",
                            data=pdf_bytes,
                        ),
                        types.Part.from_text(text="""
Extract information from the provided Vietnamese legal PDF document
and return the result in the EXACT JSON format defined below.

Target JSON format:
{
  "metadata": {
    "document_type": "Thông tư | Nghị định",
    "document_number": "string | null",
    "issuing_agency": "string | null",
    "issuing_authority": "string | null",
    "issue_place": "string | null",
    "issue_date": "YYYY-MM-DD | null",
    "effective_date": "YYYY-MM-DD | null"
  },
  "title": "string | null",
  "legal_basis": "string | null",
  "content": [
    "string"
  ]
}

Extraction rules:
- The PDF may be a scanned document.
- Only extract information that is explicitly visible in the document.
- Do NOT infer, guess, or normalize information.
- If a field cannot be found or is unreadable, use null.

Critical disambiguation rule:
- A new content item MUST start ONLY if the line begins with the exact word "Điều"
  followed by a space and a number (e.g. "Điều 3", "Điều 82.").
- Lines starting with numbers only (e.g. "18.", "19.") MUST NEVER be treated
  as a new "Điều", even if they appear to introduce amendments.
- Any quoted or referenced "Điều" inside the content of another "Điều"
  MUST be kept inside the current content item.



Field instructions:
- metadata.document_type:
  Use "Thông tư" or "Nghị định" exactly as stated in the document title.
- title:
  Extract the full official title of the document as ONE string.
- legal_basis:
  Extract the entire “Căn cứ …” section as a single string.
  Preserve original wording and punctuation.
- content:
  Extract the main body of the document.
  Each element in the array should represent ONE major content unit
  (usually one “Điều” or equivalent block of provisions).
  Keep original legal wording.
  A "Điều" ends ONLY when the next line starts with "Điều <number>" or when the "Nơi nhận" section begins.


Output rules:
- Return ONLY valid JSON.
- No explanation, no markdown.
- Do NOT include any extra fields.
- Preserve original Vietnamese legal wording exactly.
- All line breaks inside strings MUST be escaped as "\n".
- Do NOT include raw line breaks inside JSON strings.
- The output MUST be parseable by standard JSON parsers.
        """),
                    ],
                ),
            ]
            generate_content_config = types.GenerateContentConfig(
                # thinkingConfig: {
                #     thinkingBudget: 0,
                # },
                system_instruction=[
                    types.Part.from_text(text="""
You are a Vietnamese legal document information extraction engine.

Your task:
- Read the provided legal PDF document.
- Extract information strictly based on visible text.
- The document may be scanned and partially unreadable.

Strict constraints:
- Output MUST match the provided JSON format exactly.
- DO NOT add extra fields.
- DO NOT restructure the JSON.
- DO NOT hallucinate or infer missing information.
- If information is unclear or missing, return null.
- Preserve original legal wording exactly.
- Dates must follow ISO format: YYYY-MM-DD.
- Return ONLY valid JSON, no explanation, no markdown.

You MUST ensure the output is strictly valid JSON.
Any newline inside a string MUST be escaped as "\n".

- Escape all double quotes inside content as \"
- Do NOT include semicolons outside strings
- If unsure, replace “ ” with '
- If JSON may be invalid, return ERROR instead


                    """),
                ]
            )

            result =  client.models.generate_content(
                model=model,
                contents=contents,
                config=generate_content_config,
            ).text

            result_json = parse_json(result)

            os.makedirs(output_dir, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result_json, f, ensure_ascii=False, indent=2)

            print(f"✅ Saved: {json_name}")
            time.sleep(1) 
        except Exception as e:
            print(f"❌ Error processing {file_name}: {e}")

if __name__ == "__main__":
    generate()
    
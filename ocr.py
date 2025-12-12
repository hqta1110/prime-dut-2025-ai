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
    # client = genai.Client(
    #     api_key=os.environ.get("GEMINI_API_KEY"),
    # )
    pdf_path = "D:\\selflearning\\vnpt-ai-infor\\analysize\\pdf_files\\[SCAN]_01-tt-vpcp.signed.pdf"
    output_path = "D:\\selflearning\\vnpt-ai-infor\\analysize\\output\\01-tt-vpcp.json"


    pdf_dir = "D:\\selflearning\\vnpt-ai-infor\\analysize\\pdf_files"
    output_dir = "D:\\selflearning\\vnpt-ai-infor\\analysize\\output"
    skipped_dir = "D:\\selflearning\\vnpt-ai-infor\\analysize\\skipped_large_pdf"


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
                        Extract information from the provided PDF and fill in the following JSON schema.

        Schema:
        {
        "metadata": {
            "document_type": "Thông tư",
            "document_number": "string",
            "symbol": "string",
            "issuing_agency": "string",
            "issuing_authority": "string",
            "issue_place": "string",
            "issue_date": "YYYY-MM-DD",
            "effective_date": "YYYY-MM-DD | null",
            "language": "vi"
        },

        "title": {
            "short_title": "string | null",
            "full_title": "string"
        },

        "amended_documents": [
            {
            "document_type": "string",
            "document_number": "string",
            "issue_date": "YYYY-MM-DD | null",
            "issuing_authority": "string | null",
            "description": "string | null"
            }
        ],

        "legal_basis": [
            {
            "basis_type": "Nghị định | Luật | Quyết định | Khác",
            "document_number": "string",
            "issue_date": "YYYY-MM-DD | null",
            "issuing_authority": "string | null",
            "summary": "string"
            }
        ],

        "content": {
            "articles": [
            {
                "article_number": "Điều X",
                "title": "string | null",
                "clauses": [
                {
                    "clause_number": "Khoản X",
                    "points": [
                    {
                        "point_label": "a | b | c | d | ...",
                        "text": "string"
                    }
                    ],
                    "text": "string | null"
                }
                ]
            }
            ]
        },

        "effect_and_execution": {
            "effective_date": "YYYY-MM-DD",
            "responsible_organizations": [
            "string"
            ],
            "implementation_notes": "string | null"
        },

        "recipients": [
            "string"
        ],

        "signature": {
            "signing_authority": "string",
            "signer_name": "string",
            "signature_date": "YYYY-MM-DD | null",
            "seal_present": true
        },

        "summary": "string | null",
        "subjects": [
            "string"
        ]
        }


        Instructions:
        - Only extract information that is explicitly stated in the PDF.
        - Do not infer missing information.
        - If information is not found, use null.
        - Return ONLY the JSON object.
        - Before returning the result, verify that all extracted fields match the document context.

        Important:
        - Preserve the hierarchical structure: Article → Clause → Point.
        - Do not merge or summarize legal provisions.
        - Keep original legal wording.

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
                    You are a legal document information extraction engine.

        Your task:
        - Read and understand the provided PDF document.
        - Extract information strictly based on the document content.
        - Fill the given JSON schema accurately.
        - DO NOT add, infer, or hallucinate information.

        Rules:
        - Use ONLY information explicitly present in the PDF.
        - If a field cannot be found, return null.
        - Keep original wording where applicable.
        - Dates must follow ISO format: YYYY-MM-DD.
        - Return ONLY valid JSON, no explanation, no markdown.
        - Do not paraphrase legal titles. Preserve official wording.


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
    
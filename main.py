import requests
import PyPDF2
from io import BytesIO
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()


def extract_text_from_pdf_url(pdf_url: str) -> list:
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        pdf_file = BytesIO(response.content)
        reader = PyPDF2.PdfReader(pdf_file)
        pdf_text = []
        for i, page in enumerate(reader.pages):
            content = page.extract_text()
            if content:
                pdf_text.append(f"\n--- Page {i + 1} ---\n{content}")
            else:
                pdf_text.append(f"\n--- Page {i + 1} ---\n(No extractable text on this page)")
        return pdf_text
    except Exception as e:
        return [f"Error extracting PDF: {str(e)}"]


@app.post("/hackrx/run")
async def hackrx_run(request: Request):
    # Bearer token authentication
    auth = request.headers.get("authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    data = await request.json()
    documents = data.get("documents")
    questions = data.get("questions")
    if not documents or not questions or not isinstance(questions, list):
        raise HTTPException(status_code=400, detail="Invalid input format")

    pdf_text_pages = extract_text_from_pdf_url(documents)  # list of each page's text
    all_pdf_text = "\n".join(pdf_text_pages)

    # DEBUG: Return first 2500 characters of extracted text, plus dummy answers
    return JSONResponse({
        "pdf_text": all_pdf_text,  # Just for inspection/debug
        "answers": ["dummy answer" for _ in questions]
    })

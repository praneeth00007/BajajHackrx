import requests
import fitz  # PyMuPDF
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()


def extract_pdf_text(url):
    response = requests.get(url)
    response.raise_for_status()
    doc = fitz.open(stream=response.content, filetype="pdf")
    # Concatenate all page texts (or page-wise, if you wish)
    return "\n".join(page.get_text() for page in doc)


@app.post("/hackrx/run")
async def hackrx_run(request: Request):
    # Strict Bearer authentication
    auth = request.headers.get("authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    data = await request.json()
    doc_url = data.get("documents")
    questions = data.get("questions")
    if not doc_url or not questions or not isinstance(questions, list):
        raise HTTPException(status_code=400, detail="Invalid input")

    # Extract PDF text
    pdf_text = extract_pdf_text(doc_url)

    # Return both PDF text (first 3000 chars for sanity) and dummy answers
    return JSONResponse({
        "pdf_text": pdf_text,  # <-- show this for debugging! Remove in production.
        "answers": ["dummy answer" for _ in questions]
    })

import requests
import fitz  # PyMuPDF
import re
import hashlib
import pinecone
import openai
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

# Set your API keys (secure in production: use .env or secrets manager)
PINECONE_API_KEY = "YOUR_PINECONE_API_KEY"
PINECONE_ENV = "YOUR_PINECONE_ENV"
PINECONE_INDEX = "hackrx-index"  # must exist in your dashboard, dim=1536 for ada-002
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"

# Init external services
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index(PINECONE_INDEX)
openai.api_key = OPENAI_API_KEY

app = FastAPI()

# ----------- PDF Extraction -----------
def extract_pdf_text(url):
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    doc = fitz.open(stream=response.content, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

# ----------- Chunking -----------
def chunk_text(text, min_length=100, max_length=500):
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    buffer = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) + len(buffer) < min_length:
            buffer += " " + para
            continue
        if buffer:
            para = buffer + " " + para
            buffer = ""
        while len(para) > max_length:
            chunks.append(para[:max_length])
            para = para[max_length:]
        if para:
            chunks.append(para)
    if buffer:
        chunks.append(buffer)
    return chunks

# ----------- Embedding -----------
def get_embedding(text):
    result = openai.Embedding.create(
        input=[text], model="text-embedding-ada-002"
    )
    return result['data'][0]['embedding']

# ----------- Pinecone Upsert -----------
def upsert_chunks_to_pinecone(chunks, doc_id):
    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        vectors.append(
            (f"{doc_id}_chunk_{i}", embedding, {"text": chunk, "doc_id": doc_id})
        )
    index.upsert(vectors)

# ----------- Pinecone Retrieval -----------
def retrieve_top_chunks(question, doc_id, top_k=5):
    query_vec = get_embedding(question)
    res = index.query(
        vector=query_vec,
        top_k=top_k,
        filter={"doc_id": {"$eq": doc_id}},
        include_metadata=True,
    )
    return [m["metadata"]["text"] for m in res.get("matches", [])]

# ----------- OpenAI Answer Generation -----------
def answer_from_context(context, question):
    prompt = f"Answer the following question only using the context below.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Or "gpt-4"
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"].strip()

# ----------- FastAPI Endpoint -----------
@app.post("/hackrx/run")
async def hackrx_run(request: Request):
    # Bearer Auth
    auth = request.headers.get("authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    data = await request.json()
    doc_url = data.get("documents")
    questions = data.get("questions")
    if not doc_url or not isinstance(questions, list) or not questions:
        raise HTTPException(status_code=400, detail="Invalid input format")

    # Step 1: PDF Extraction
    try:
        pdf_text = extract_pdf_text(doc_url)
    except Exception:
        raise HTTPException(status_code=500, detail="PDF extraction failed")

    # Step 2: Chunk
    chunks = chunk_text(pdf_text)
    doc_id = hashlib.md5(doc_url.encode("utf-8")).hexdigest()

    # Step 3: Upsert chunks to Pinecone. (Optional: cache if already exists)
    upsert_chunks_to_pinecone(chunks, doc_id)

    # Step 4: For each question, find best context and answer
    answers = []
    for q in questions:
        top_chunks = retrieve_top_chunks(q, doc_id, top_k=5)
        context = "\n\n".join(top_chunks)
        if not context.strip():
            answers.append("Sorry, no relevant information found in document.")
        else:
            try:
                answer = answer_from_context(context, q)
            except Exception:
                answer = "Error generating answer."
            answers.append(answer)

    return JSONResponse({"answers": answers})


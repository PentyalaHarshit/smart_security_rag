import os
from pypdf import PdfReader

import chromadb
import ollama
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from sentence_transformers import SentenceTransformer

app = FastAPI()

EVENT_FILE = "data/events.txt"
DOCS_FOLDER = "data/docs"

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()


def load_all_documents():
    docs = []

    if os.path.exists(EVENT_FILE):
        with open(EVENT_FILE, "r", encoding="utf-8") as f:
            docs.extend([line.strip() for line in f if line.strip()])

    if os.path.exists(DOCS_FOLDER):
        for file in os.listdir(DOCS_FOLDER):
            path = os.path.join(DOCS_FOLDER, file)

            if file.endswith(".txt"):
                with open(path, "r", encoding="utf-8") as f:
                    docs.append(f.read())

            elif file.endswith(".pdf"):
                reader = PdfReader(path)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        docs.append(text)

    return docs


def build_db():
    docs = load_all_documents()

    try:
        client.delete_collection("security_rag")
    except Exception:
        pass

    collection = client.get_or_create_collection("security_rag")

    for i, doc in enumerate(docs):
        emb = embedder.encode(doc).tolist()

        collection.add(
            ids=[str(i)],
            documents=[doc],
            embeddings=[emb]
        )

    return collection


def get_llm_answer(question, context):
    try:
        prompt = f"""
You are an AI security camera assistant.

Answer only using the context below.
If the answer is not found, say: I don't know from the camera records.

Context:
{context}

Question:
{question}

Answer:
"""

        response = ollama.chat(
            model="llama3",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response["message"]["content"]

    except Exception as e:
        return f"""
        Ollama Error: {e}

        Fix:
        1. Open terminal
        2. Run: ollama run llama3
        3. Keep it running
        4. Refresh this page
        """


@app.get("/", response_class=HTMLResponse)
def home():
    docs = load_all_documents()
    recent_docs = docs[-10:]

    recent_html = "".join([f"<li>{doc}</li>" for doc in recent_docs])

    return f"""
    <html>
    <head>
        <title>AI Security Camera RAG</title>
    </head>
    <body>
        <h1>AI Security Camera + RAG + Local LLM</h1>

        <form action="/ask" method="post">
            <input 
                type="text" 
                name="question" 
                placeholder="Ask: was phone detected?" 
                style="width:500px; padding:8px;"
            >
            <button type="submit">Ask</button>
        </form>

        <h2>Recent Events / Documents</h2>
        <ul>
            {recent_html}
        </ul>
    </body>
    </html>
    """


@app.post("/ask", response_class=HTMLResponse)
def ask(question: str = Form(...)):
    try:
        docs = load_all_documents()

        if not docs:
            return """
            <html>
            <body>
                <h2>No events/documents found.</h2>
                <p>Run camera_detect.py first or add files in data/docs.</p>
                <a href="/">Back</a>
            </body>
            </html>
            """

        collection = build_db()

        query_emb = embedder.encode(question).tolist()

        result = collection.query(
            query_embeddings=[query_emb],
            n_results=min(5, len(docs))
        )

        retrieved_docs = result["documents"][0]
        context = "\n".join(retrieved_docs)

        answer = get_llm_answer(question, context)

        retrieved_html = "".join([f"<li>{doc}</li>" for doc in retrieved_docs])

        return f"""
        <html>
        <head>
            <title>RAG Answer</title>
        </head>
        <body>
            <h1>AI Security Camera + RAG + Local LLM</h1>

            <h2>Question</h2>
            <p>{question}</p>

            <h2>LLM Answer</h2>
            <pre style="white-space: pre-wrap;">{answer}</pre>

            <h2>Retrieved Context</h2>
            <ul>
                {retrieved_html}
            </ul>

            <br>
            <a href="/">Back</a>
        </body>
        </html>
        """

    except Exception as e:
        return f"""
        <html>
        <body>
            <h1>Error</h1>
            <pre>{e}</pre>
            <a href="/">Back</a>
        </body>
        </html>
        """
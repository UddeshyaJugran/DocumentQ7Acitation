
import json, os, sys, time
import numpy as np
from google import genai
from google.genai import types
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

MODEL = "gemini-3.6-flash"   # model used 
CHUNK_WORDS = 150   # ek chunk me kitne words
OVERLAP = 30        # do chunks ke beech overlap, taaki baat beech me na kate
TOP_K = 4           # kitne chunks retrieve karne hain
MAX_TRIES = 4       # server busy ho toh kitni baar koshish karni hai

embedder = SentenceTransformer("all-MiniLM-L6-v2")  # free aur local embeddings
client = genai.Client()  # GEMINI_API_KEY environment variable se leta hai


# 1. Documents load karo: har page ka (file, page number, text)
def load_pages(folder):
    pages = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if name.lower().endswith(".pdf"):
            for i, page in enumerate(PdfReader(path).pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append((name, i, text))
        elif name.lower().endswith(".txt"):
            with open(path, encoding="utf-8") as f:
                pages.append((name, 1, f.read()))
    return pages


# 2. Text ko chhote chunks me todo, har chunk ko ID do
def make_chunks(pages):
    chunks, step = [], CHUNK_WORDS - OVERLAP
    for source, page_no, text in pages:
        words = text.split()
        for start in range(0, max(len(words), 1), step):
            piece = " ".join(words[start:start + CHUNK_WORDS])
            if piece.strip():
                chunks.append({"id": len(chunks) + 1, "source": source,
                               "page": page_no, "text": piece})
    return chunks


# 3. Har chunk ka vector banao
def build_index(chunks):
    return np.array(embedder.encode([c["text"] for c in chunks],
                                    normalize_embeddings=True))


# 4. Sawaal se milte-julte top chunks nikalo (cosine similarity)
def retrieve(question, chunks, vectors, k=TOP_K):
    q = embedder.encode([question], normalize_embeddings=True)[0]
    scores = vectors @ q
    return [chunks[i] for i in np.argsort(scores)[::-1][:k]]


# 5. Gemini se answer + citations maango
SYSTEM = ("Sirf diye gaye numbered context chunks se answer do. "
          "Agar answer chunks me nahi hai to bolo ki nahi mila. "
          'Sirf valid JSON return karo: {"answer": "...", "citations": [chunk ids]}')


def call_gemini(contents):
    """503 (server busy) ya 429 (quota) aane par ruk ke dobara koshish karta hai."""
    for attempt in range(MAX_TRIES):
        try:
            return client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM,
                    response_mime_type="application/json",   # JSON hi wapas aaye
                ),
            )
        except Exception as e:
            busy = "503" in str(e) or "429" in str(e)
            if busy and attempt < MAX_TRIES - 1:
                wait = 2 ** (attempt + 1)   # 2, 4, 8 second
                print(f"Server busy, {wait} second baad dobara koshish...")
                time.sleep(wait)
            else:
                raise


def answer(question, retrieved):
    context = "\n\n".join(
        f"[{c['id']}] ({c['source']}, page {c['page']})\n{c['text']}" for c in retrieved)
    resp = call_gemini(f"Context:\n{context}\n\nQuestion: {question}")
    raw = resp.text.strip().replace("```json", "").replace("```", "")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"answer": raw, "citations": []}
    # Fake citations hatao: sirf wahi IDs rakho jo retrieve hue the
    valid = {c["id"] for c in retrieved}
    data["citations"] = [i for i in data.get("citations", []) if i in valid]
    return data


# 6. Command line 
def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "docs"
    pages = load_pages(folder)
    chunks = make_chunks(pages)
    if not chunks:
        print("Koi document nahi mila. docs/ folder me PDF ya txt daalo.")
        return
    vectors = build_index(chunks)
    print(f"{len(pages)} pages, {len(chunks)} chunks index ho gaye.\n")
    by_id = {c["id"]: c for c in chunks}
    while True:
        q = input("Question: ").strip()
        if q.lower() in ("exit", "quit", ""):
            break
        try:
            result = answer(q, retrieve(q, chunks, vectors))
        except Exception as e:
            print(f"\nError: {e}\n")   # crash ki jagah error dikhao aur agla sawaal lo
            continue
        print("\nAnswer:", result["answer"])
        for cid in result["citations"]:
            c = by_id[cid]
            print(f"  [{cid}] {c['source']} p.{c['page']}: "
                  f"{c['text'][:200].replace(chr(10), ' ')}...")
        print()

if __name__ == "__main__":
    main()
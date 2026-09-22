# Smart Document Assistant

A small RAG (Retrieval-Augmented Generation) app that lets a user upload PDF/TXT
documents and ask questions about their contents, with grounded, source-cited answers.

## 1. Problem Understanding

Users often have several documents (policies, handbooks, contracts) and want quick,
trustworthy answers without reading everything. The system needs to: extract text from
uploaded files, find the passages relevant to a question, and have an LLM answer using
*only* that retrieved content — while clearly saying when it doesn't know, instead of
guessing.

## 2. Architecture

See `architecture.png`.

```
User → Streamlit UI → Document Parser (pdfplumber/txt)
     → Chunking (overlapping windows)
     → Embeddings (sentence-transformers, local, free)
     → FAISS vector store
     → Retrieval (top-k cosine similarity)
     → LLM (Groq, Llama 3.1 8B) with a grounded prompt
     → Answer + sources + confidence, shown in the UI
```

Everything runs in a single Streamlit process; FAISS and the embedding model are
in-memory (rebuilt each session), which keeps the app simple and dependency-light for
an 8-hour build.

## 3. Technology Choices

| Component | Choice | Why |
|---|---|---|
| UI | Streamlit | Fastest way to build a usable upload/chat UI in Python |
| PDF parsing | pdfplumber | Reliable text + page-number extraction |
| Chunking | Custom overlapping-window splitter | No extra dependency, easy to explain/tune |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) | Runs locally, free, no API key needed, fast |
| Vector store | FAISS (`IndexFlatIP`) | Simple, in-memory, exact cosine search — fine at this scale |
| LLM | Groq API (Llama 3.1 8B Instant) | Free tier, very fast inference, OpenAI-compatible API |

## 4. How to Run
```bash
### 1. Create a virtual environment
py -m venv venv    #Mac/Linux: python3 -m venv venv
### 2. Activate the virtual environment
venv\Scripts\activate.ps1     #Mac/Linux: source venv/bin/activate    

pip install -r requirements.txt

# Get a free key at https://console.groq.com/keys

streamlit run app.py
```
The application should open automatically in a browser,
If it does not open automatically, open the URL displayed in the terminal.
Then: paste the key into the left sidebar field. upload one or more PDF/TXT files in the sidebar → click **Process documents** →
type a question → **Ask**.

## 5. Hallucination Handling

Two layers:
1. **Retrieval gate** — if the best-matching chunk's cosine similarity is below
   `SIMILARITY_THRESHOLD` (0.30), the app never calls the LLM at all and returns
   *"I couldn't find information about that in the uploaded documents."*
2. **Prompt grounding** — the LLM's system prompt explicitly restricts it to the
   retrieved context only, forbids outside knowledge, and gives it the exact
   fallback sentence to use when the context is insufficient.

A **confidence indicator** (based on the top similarity score) is shown with every
answer so the user can see how strong the match was, even when an answer is given.

## 6. Creative Feature(s)

- **Conversation memory** — recent Q&A turns are passed back into the prompt so
  follow-up questions ("what about for part-time staff?") work without repeating
  context.
- **Confidence indicator** — a similarity-based confidence badge (strong / partial /
  weak) next to every answer.

## 7. Known Limitations

- No persistent storage — the index is rebuilt every session (fine for a demo, not
  for production).
- Chunking is character-based, not sentence/semantic-aware, so a chunk boundary can
  occasionally split a sentence.
- Citation is at document + chunk level (with page numbers embedded in PDF chunk text),
  not exact bounding-box/page-level highlighting.
- Only PDF and TXT are supported (DOCX/CSV are optional per the spec and not implemented).
- Groq's free tier has rate limits; heavy use may hit them.

## 8. AI Tools Used

Claude was used to scaffold the RAG pipeline (chunking/embedding/retrieval logic),
the Streamlit UI, the hallucination-guard design, and this README.
All generated code was read, run, and adjusted before submission.

## 9. Time Log (approximate)

| Phase | Time |
|---|---|
| Problem understanding | 20 min |
| Core pipeline (parsing, chunking, embeddings, FAISS) | 2.5 hr |
| UI + LLM integration + hallucination handling | 2 hr |
| Creative features (memory, confidence) | 45 min |
| Testing/debugging | 1.5 hr |
| README + architecture diagram + video | 1 hr |

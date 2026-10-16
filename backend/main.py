from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from rag_pipeline import main_chain

app = FastAPI(title="VERITAS RAG API")


@app.get("/", include_in_schema=False)
def root():
    # This is an API, not a webpage -- send anyone who opens it in a
    # browser to the interactive docs instead of a bare 404.
    return RedirectResponse(url="/docs")

# Tighten allow_origins to your actual deployed frontend domain before
# sharing this link publicly -- "*" is fine for local dev only.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    model_mode: str = "cloud"  # "cloud" (Gemini, wired) or "local" (Ollama, not wired yet)


@app.post("/ask")
def ask(req: QueryRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if req.model_mode == "local":
        # Placeholder: plug your Ollama chain into rag_pipeline.py and branch
        # on model_mode here once it exists. Left explicit rather than silently
        # falling back to cloud, so the toggle never lies about what ran.
        raise HTTPException(
            status_code=501,
            detail="Local model mode isn't wired up yet. Add an Ollama-backed "
                   "chain to rag_pipeline.py and branch on model_mode here.",
        )

    try:
        result = main_chain.invoke(query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {exc}")

    return {"answer": result["answer"], "sources": result["sources"]}


@app.get("/health")
def health():
    return {"status": "ok"}

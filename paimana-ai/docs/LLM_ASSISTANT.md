# Adding a local open-source LLM to the assistant

The assistant is **retrieval-grounded by default**: it computes every figure from the scored
portfolio and returns the rows used, so it cannot hallucinate a number. That is the right
default for a government monitoring tool.

If you want free-form phrasing on top, keep the grounding and let the LLM only *rephrase*.

## 1. Run a local model

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b          # or qwen2.5:7b, mistral-nemo - all open weights
```

## 2. Patch `backend/app.py::assistant`

At the end of each intent branch, before returning, pass the computed answer plus the evidence
rows through the model:

```python
import os, requests

LLM = os.getenv("PAIMANA_LLM_ENDPOINT")   # e.g. http://localhost:11434/api/generate

def phrase(answer, evidence):
    if not LLM:
        return answer
    prompt = (
        "You are a project monitoring analyst for MoSPI. Rewrite the FACTS below as two or "
        "three clear sentences for a senior officer. Use ONLY these facts. Do not add, "
        "estimate or infer any number that is not present.\n\n"
        f"FACTS: {answer}\n\nSUPPORTING ROWS: {evidence[:5]}"
    )
    try:
        r = requests.post(LLM, json={"model": "llama3.1:8b", "prompt": prompt,
                                     "stream": False}, timeout=30)
        return r.json().get("response", answer).strip()
    except Exception:
        return answer            # always degrade to the grounded answer
```

Then `return {"answer": phrase(ans, ev), ...}`.

## 3. Optional - RAG over flash reports

Index the PDF flash reports and ministry review minutes with `sentence-transformers`
(`all-MiniLM-L6-v2`) into FAISS, retrieve the top three passages for the question, and append
them to the FACTS block. This lets the assistant answer *narrative* questions ("what did the
last review say about this corridor") that structured data cannot.

**Keep the guardrail:** numbers come from the dataframe, never from the model.

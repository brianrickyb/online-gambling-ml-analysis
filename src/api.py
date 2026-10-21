"""REST API that serves the trained model.

Usage:
    uvicorn src.api:app --reload
    # open http://127.0.0.1:8000 (web form) or http://127.0.0.1:8000/docs (Swagger)
"""
import hashlib
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.config import MODEL_INFO_PATH, PREDICTION_LOG_PATH, THRESHOLD
from src.predict import load_model

state = {}


@asynccontextmanager
async def lifespan(app):
    # Load once at startup, not on every request: loading takes ~1 second.
    state["model"] = load_model()
    state["info"] = json.loads(MODEL_INFO_PATH.read_text()) if MODEL_INFO_PATH.exists() else {}
    yield
    state.clear()


app = FastAPI(title="Judol Comment Detector", version="1.0", lifespan=lifespan)


def get_model():
    if "model" not in state:
        raise HTTPException(503, "Model not loaded")
    return state["model"]


def get_model_info():
    return state.get("info", {})


class PredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=1000, examples=[["gacor parah di 𝐒𝐆𝐈𝟖𝟖", "mobilnya keren bang"]])


class Prediction(BaseModel):
    text: str
    label: str
    score: float


class PredictResponse(BaseModel):
    model_version: str
    predictions: list[Prediction]


def log_predictions(texts, scores, labels, version):
    """Append one line per prediction for monitoring (src/monitor.py).

    Only a hash of the text is stored, not the comment itself, to avoid
    keeping users' personal data in logs.
    """
    PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with PREDICTION_LOG_PATH.open("a", encoding="utf-8") as f:
        for text, score, label in zip(texts, scores, labels):
            f.write(json.dumps({
                "ts": now,
                "model_version": version,
                "text_sha1": hashlib.sha1(text.encode("utf-8")).hexdigest(),
                "text_length": len(text),
                "score": round(float(score), 6),
                "label": label,
            }) + "\n")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/model-info")
def model_info(info=Depends(get_model_info)):
    return info


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, model=Depends(get_model), info=Depends(get_model_info)):
    threshold = info.get("threshold", THRESHOLD)
    scores = model.predict_proba(req.texts)[:, 1]
    labels = ["judol" if s >= threshold else "normal" for s in scores]
    version = info.get("version", "unknown")
    log_predictions(req.texts, scores, labels, version)
    return PredictResponse(
        model_version=version,
        predictions=[Prediction(text=t, label=l, score=round(float(s), 4)) for t, l, s in zip(req.texts, labels, scores)],
    )


@app.get("/", response_class=HTMLResponse)
def index():
    return """<!doctype html>
<html lang="id"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Judol Comment Detector</title>
<style>
  body{font-family:system-ui,sans-serif;max-width:640px;margin:40px auto;padding:0 16px;background:#fafafa;color:#222}
  textarea{width:100%;height:140px;font:inherit;padding:8px;box-sizing:border-box}
  button{margin-top:8px;padding:8px 20px;font:inherit;cursor:pointer}
  .row{padding:8px 12px;margin:6px 0;border-radius:6px;background:#fff;border-left:5px solid #2e7d32}
  .row.judol{border-color:#c62828}
  small{color:#666}
</style></head><body>
<h1>Judol Comment Detector</h1>
<p>Satu komentar per baris.</p>
<textarea id="t">mantap videonya bang
gacor parah, sehari maxwin di 𝐒𝐆𝐈𝟖𝟖
k o i s l o t bikin aku sultan</textarea>
<button onclick="go()">Cek</button>
<div id="out"></div>
<script>
async function go(){
  const texts=document.getElementById('t').value.split('\\n').map(s=>s.trim()).filter(Boolean);
  if(!texts.length) return;
  const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({texts})});
  const d=await r.json(); const out=document.getElementById('out'); out.innerHTML='';
  for(const p of d.predictions){
    const div=document.createElement('div'); div.className='row '+p.label;
    const b=document.createElement('b'); b.textContent=p.label.toUpperCase();
    const s=document.createElement('small'); s.textContent=' skor '+p.score.toFixed(3)+' · ';
    div.append(b,s,document.createTextNode(p.text)); out.append(div);
  }
}
</script></body></html>"""

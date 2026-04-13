"""
FraudSentinel — FastAPI backend for real-time fraud scoring and SSE alerts.
"""

from __future__ import annotations

import asyncio
import json
import time
import warnings
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import chromadb
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from ollama import Client as OllamaClient
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from rag_setup import DeterministicEmbeddingFunction

# Suppress sklearn feature name mismatch warning (arrays don't have feature names)
warnings.filterwarnings("ignore", message="X does not have valid feature names*")


MERCHANT_CATEGORIES = [
    "Grocery",
    "Gas Station",
    "Restaurant",
    "Online Shopping",
    "ATM",
    "Electronics",
    "Travel",
    "Healthcare",
]


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _creditcard_feature_columns() -> list[str]:
    # Column order after dropping Time and Class in train.py preprocessing (V1–V28, Amount).
    return [f"V{i}" for i in range(1, 29)] + ["Amount"]


def _fit_iso_minmax_scaler(
    iso_model: IsolationForest,
    amount_scaler: StandardScaler,
    creditcard_path: Path,
    max_rows: int = 100_000,
) -> MinMaxScaler | None:
    # Fit MinMaxScaler on -decision_function(train-like rows) so IF scores map to [0, 1] at inference (matches train.py evaluation spirit).
    if not creditcard_path.is_file():
        return None
    usecols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    df = pd.read_csv(creditcard_path, usecols=usecols, nrows=max_rows)
    amt_scaled = amount_scaler.transform(df[["Amount"]].values)
    X_ref = np.hstack(
        [df[[f"V{i}" for i in range(1, 29)]].values.astype(np.float64), amt_scaled]
    )
    raw = -iso_model.decision_function(X_ref)
    mm = MinMaxScaler()
    mm.fit(raw.reshape(-1, 1))
    return mm


def load_models_and_scaler(
    models_dir: Path,
) -> tuple[xgb.XGBClassifier, IsolationForest, StandardScaler, list[str]]:
    # Deserialize XGBoost classifier, IsolationForest, fitted StandardScaler, and persisted feature column names from models_dir.
    xgb_model: xgb.XGBClassifier = joblib.load(models_dir / "xgb_model.pkl")
    iso_model: IsolationForest = joblib.load(models_dir / "iso_model.pkl")
    scaler: StandardScaler = joblib.load(models_dir / "scaler.pkl")
    feature_columns = _creditcard_feature_columns()
    return xgb_model, iso_model, scaler, feature_columns


def load_chroma_collection(
    persist_directory: Path,
    collection_name: str,
) -> tuple[chromadb.PersistentClient, chromadb.Collection]:
    # Open the persistent Chroma client and return the named collection for similarity search at inference time.
    client = chromadb.PersistentClient(path=str(persist_directory))
    collection = client.get_collection(
        name=collection_name,
        embedding_function=DeterministicEmbeddingFunction(),
    )
    return client, collection


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    transaction_id: str
    amount: float
    v1: float = Field(validation_alias=AliasChoices("v1", "V1"))
    v2: float = Field(validation_alias=AliasChoices("v2", "V2"))
    v3: float = Field(validation_alias=AliasChoices("v3", "V3"))
    v4: float = Field(validation_alias=AliasChoices("v4", "V4"))
    v5: float = Field(validation_alias=AliasChoices("v5", "V5"))
    v6: float = Field(validation_alias=AliasChoices("v6", "V6"))
    v7: float = Field(validation_alias=AliasChoices("v7", "V7"))
    v8: float = Field(validation_alias=AliasChoices("v8", "V8"))
    v9: float = Field(validation_alias=AliasChoices("v9", "V9"))
    v10: float = Field(validation_alias=AliasChoices("v10", "V10"))
    v11: float = Field(validation_alias=AliasChoices("v11", "V11"))
    v12: float = Field(validation_alias=AliasChoices("v12", "V12"))
    v13: float = Field(validation_alias=AliasChoices("v13", "V13"))
    v14: float = Field(validation_alias=AliasChoices("v14", "V14"))
    v15: float = Field(validation_alias=AliasChoices("v15", "V15"))
    v16: float = Field(validation_alias=AliasChoices("v16", "V16"))
    v17: float = Field(validation_alias=AliasChoices("v17", "V17"))
    v18: float = Field(validation_alias=AliasChoices("v18", "V18"))
    v19: float = Field(validation_alias=AliasChoices("v19", "V19"))
    v20: float = Field(validation_alias=AliasChoices("v20", "V20"))
    v21: float = Field(validation_alias=AliasChoices("v21", "V21"))
    v22: float = Field(validation_alias=AliasChoices("v22", "V22"))
    v23: float = Field(validation_alias=AliasChoices("v23", "V23"))
    v24: float = Field(validation_alias=AliasChoices("v24", "V24"))
    v25: float = Field(validation_alias=AliasChoices("v25", "V25"))
    v26: float = Field(validation_alias=AliasChoices("v26", "V26"))
    v27: float = Field(validation_alias=AliasChoices("v27", "V27"))
    v28: float = Field(validation_alias=AliasChoices("v28", "V28"))
    hour: int = Field(ge=0, le=23)
    merchant_category: str | None = None


class ShapFeatureItem(BaseModel):
    feature: str
    value: float
    impact: float


class ScoreResponse(BaseModel):
    transaction_id: str
    risk_score: float
    verdict: Literal["FRAUD", "REVIEW", "APPROVE"]
    verdict_logic: Literal["BLOCK", "OTP", "APPROVE"]
    shap_features: list[ShapFeatureItem]
    processing_ms: int


class ExplainRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    transaction_id: str
    risk_score: float
    shap_features: list[ShapFeatureItem]
    amount: float
    hour: int = Field(ge=0, le=23)


class ExplainResponse(BaseModel):
    transaction_id: str
    explanation: str
    similar_cases: list[str]
    model: str


def _shap_values_positive_class_row(shap_values: Any) -> np.ndarray:
    # Normalize SHAP return type to shape (n_features,) for the positive (fraud) class.
    if isinstance(shap_values, list):
        arr = np.asarray(shap_values[1])
    else:
        arr = np.asarray(shap_values)
    if arr.ndim == 3 and arr.shape[-1] == 2:
        arr = arr[:, :, 1]
    if arr.ndim == 2:
        return arr[0]
    return arr.ravel()


def score_transaction(
    body: ScoreRequest,
    xgb_model: xgb.XGBClassifier,
    iso_model: IsolationForest,
    amount_scaler: StandardScaler,
    feature_columns: list[str],
    iso_norm_scaler: MinMaxScaler | None,
    shap_explainer: Any,
    hybrid_threshold: float,
) -> dict[str, Any]:
    # Scale Amount, build X row, hybrid score, verdicts, SHAP top-5, risk_score 0–100, processing_ms.
    t0 = time.perf_counter()
    v_order = [f"v{i}" for i in range(1, 29)]
    raw_amount = np.array([[body.amount]], dtype=np.float64)
    scaled_amount = float(amount_scaler.transform(raw_amount)[0, 0])
    row = np.array(
        [[getattr(body, vk) for vk in v_order] + [scaled_amount]],
        dtype=np.float64,
    )

    xgb_prob = float(xgb_model.predict_proba(row)[0, 1])
    iso_raw = float(-iso_model.decision_function(row)[0])
    if iso_norm_scaler is not None:
        iso_score = float(iso_norm_scaler.transform([[iso_raw]])[0, 0])
        iso_score = max(0.0, min(1.0, iso_score))
    else:
        iso_score = float(1.0 / (1.0 + np.exp(-iso_raw)))

    hybrid = 0.7 * xgb_prob + 0.3 * iso_score
    risk_score = float(max(0.0, min(100.0, hybrid * 100.0)))
    thr100 = hybrid_threshold * 100.0
    if risk_score >= thr100:
        verdict: Literal["FRAUD", "REVIEW", "APPROVE"] = "FRAUD"
        verdict_logic: Literal["BLOCK", "OTP", "APPROVE"] = "BLOCK"
    elif risk_score >= thr100 * 0.7:
        verdict = "REVIEW"
        verdict_logic = "OTP"
    else:
        verdict = "APPROVE"
        verdict_logic = "APPROVE"

    # SHAP computation with timeout and error handling
    shap_features = []
    try:
        sv = shap_explainer.shap_values(row)
        shap_vec = _shap_values_positive_class_row(sv)
        names = feature_columns
        top_idx = np.argsort(np.abs(shap_vec))[::-1][:5]
        shap_features = [
            {
                "feature": names[i],
                "value": float(shap_vec[i]),
                "impact": float(abs(shap_vec[i])),
            }
            for i in top_idx
        ]
    except Exception as e:
        # If SHAP fails or times out, return empty list instead of hanging
        print(f"[SCORE] Warning: SHAP computation failed ({str(e)[:50]}), skipping features")
        shap_features = []

    processing_ms = int((time.perf_counter() - t0) * 1000)
    return {
        "transaction_id": body.transaction_id,
        "risk_score": risk_score,
        "verdict": verdict,
        "verdict_logic": verdict_logic,
        "shap_features": shap_features,
        "processing_ms": processing_ms,
    }


def retrieve_rag_context(
    collection: chromadb.Collection,
    query_embedding: list[float],
    top_k: int,
) -> list[str]:
    # Query Chroma with the transaction embedding (or text embedding) and return top_k document snippets for UI or LLM context.
    pass


def _chroma_query_similar_cases(
    collection: chromadb.Collection,
    query_text: str,
    k: int = 3,
) -> list[str]:
    # Text query against fraud_cases; return up to k document strings with metadata encoded for UI.
    result = collection.query(query_texts=[query_text], n_results=k)
    docs_batch = result.get("documents") or []
    metas_batch = result.get("metadatas") or []
    docs = list(docs_batch[0]) if docs_batch and docs_batch[0] else []
    metas = list(metas_batch[0]) if metas_batch and metas_batch[0] else []
    out: list[str] = []
    for i in range(k):
        doc = (docs[i] if i < len(docs) else "") or ""
        meta = metas[i] if i < len(metas) else {}
        case_id = str(meta.get("id") or f"case_{i+1:03d}").upper()
        case_type = str(meta.get("type") or "unknown").upper()
        out.append(f"{case_id} | {case_type} | {doc.strip()}")
    return out


def _top_shap_search_string(shap_features: list[ShapFeatureItem], n: int = 3) -> str:
    ranked = sorted(shap_features, key=lambda f: f.impact, reverse=True)[:n]
    return " ".join(f"{f.feature} impact {f.impact:.6f}" for f in ranked)


def _build_explain_ollama_prompt(
    amount: float,
    hour: int,
    risk_score: float,
    top_shap: list[ShapFeatureItem],
    case1: str,
    case2: str,
    case3: str,
) -> str:
    factors = ", ".join(
        f"{f.feature} (impact {f.impact:.6f})" for f in top_shap
    )
    return (
        "You are a fraud analyst assistant. Based only on the retrieved \n"
        "fraud cases below, explain in 2-3 plain English sentences why \n"
        "this transaction is suspicious. Do not add information not present \n"
        "in the cases or transaction data.\n"
        "\n"
        "Transaction details:\n"
        f"- Amount: {amount}\n"
        f"- Hour of day: {hour}\n"
        f"- Top risk factors: {factors}\n"
        f"- Risk score: {risk_score}/100\n"
        "\n"
        "Similar historical fraud cases:\n"
        f"{case1}\n"
        f"{case2}\n"
        f"{case3}\n"
        "\n"
        "Explanation:"
    )


def _ollama_explain_chat(client: OllamaClient, prompt: str) -> str:
    resp = client.chat(
        model="mistral:7b-instruct-q4_0",
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )
    content = resp.message.content
    if not content:
        return ""
    return str(content).strip()


async def explain_transaction(
    body: ExplainRequest,
    collection: chromadb.Collection,
    ollama_client: OllamaClient,
) -> ExplainResponse:
    search = _top_shap_search_string(body.shap_features, n=3)
    similar = _chroma_query_similar_cases(collection, search, k=3)
    case1, case2, case3 = similar[0], similar[1], similar[2]
    top3_shap = sorted(body.shap_features, key=lambda f: f.impact, reverse=True)[:3]
    prompt = _build_explain_ollama_prompt(
        body.amount,
        body.hour,
        body.risk_score,
        top3_shap,
        case1,
        case2,
        case3,
    )
    try:
        explanation = await asyncio.wait_for(
            asyncio.to_thread(_ollama_explain_chat, ollama_client, prompt),
            timeout=10.0,
        )
        return ExplainResponse(
            transaction_id=body.transaction_id,
            explanation=explanation,
            similar_cases=similar[:3],
            model="mistral:7b-instruct-q4_0",
        )
    except Exception:
        pass
    factors = ", ".join(
        f"{f.feature} (impact {f.impact:.4f})"
        for f in sorted(body.shap_features, key=lambda x: -x.impact)[:5]
    )
    return ExplainResponse(
        transaction_id=body.transaction_id,
        explanation=(
            "Flagged by hybrid model. Top factors: "
            f"{factors}. Manual review recommended."
        ),
        similar_cases=[],
        model="fallback",
    )


def _generate_and_score_synthetic_transaction(state: Any, rng: np.random.Generator) -> dict[str, Any]:
    # Every 5th synthesized event is intentionally a seeded fraud sample. Other events are drawn from legitimate rows.
    df: pd.DataFrame = state.transactions_df
    fraud_df = df[df["Class"] == 1]
    legit_df = df[df["Class"] == 0]
    state.seeded_transaction_count += 1
    is_seeded = state.seeded_transaction_count % 5 == 0
    if is_seeded and len(fraud_df) > 0:
        pool = fraud_df
    else:
        pool = legit_df if len(legit_df) > 0 else df
    idx = int(rng.integers(0, len(pool)))
    row = pool.iloc[idx]
    txn_id = f"TXN-{int(rng.integers(0, 1_000_000)):06d}"
    merchant_category = MERCHANT_CATEGORIES[int(rng.integers(0, len(MERCHANT_CATEGORIES)))]
    hour = int((int(row["Time"]) // 3600) % 24)
    data: dict[str, Any] = {
        "transaction_id": txn_id,
        "amount": float(row["Amount"]),
        "hour": hour,
        "merchant_category": merchant_category,
    }
    for i in range(1, 29):
        data[f"v{i}"] = float(row[f"V{i}"]) + float(rng.uniform(-0.1, 0.1))
    req = ScoreRequest.model_validate(data)
    out = score_transaction(
        req,
        state.xgb_model,
        state.iso_model,
        state.scaler,
        state.feature_columns,
        state.iso_norm_scaler,
        state.shap_explainer,
        state.threshold,
    )
    payload = dict(out)
    payload["amount"] = float(req.amount)
    payload["merchant_category"] = merchant_category
    payload["hour"] = hour
    payload["is_seeded"] = is_seeded
    return payload


async def synthetic_transaction_sse_stream(request: Request) -> AsyncIterator[str]:
    # Emit one scored synthetic transaction every 2s as SSE until the client disconnects.
    state = request.app.state
    rng = np.random.default_rng()
    try:
        while True:
            try:
                payload = await asyncio.to_thread(
                    _generate_and_score_synthetic_transaction,
                    state,
                    rng,
                )
                yield f"data: {json.dumps(payload, default=str)}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'error': str(exc)}, default=str)}\n\n"
            await asyncio.sleep(2)
    except GeneratorExit:
        raise
    except asyncio.CancelledError:
        raise
    except StopIteration:
        return


def register_routes(app: FastAPI) -> None:
    # Attach GET / (dashboard HTML), POST /score, GET /events (SSE), and health routes to the application instance.
    root = _project_root()

    @app.get("/")
    async def serve_index() -> FileResponse:
        return FileResponse(
            root / "static" / "index.html",
            media_type="text/html",
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "model": "loaded", "rag": "ready"}

    @app.post("/score", response_model=ScoreResponse)
    async def score_endpoint(body: ScoreRequest, request: Request) -> ScoreResponse:
        state = request.app.state
        out = score_transaction(
            body,
            state.xgb_model,
            state.iso_model,
            state.scaler,
            state.feature_columns,
            state.iso_norm_scaler,
            state.shap_explainer,
            state.threshold,
        )
        return ScoreResponse.model_validate(out)

    @app.post("/explain", response_model=ExplainResponse)
    async def explain_endpoint(body: ExplainRequest, request: Request) -> ExplainResponse:
        state = request.app.state
        return await explain_transaction(
            body,
            state.chroma_collection,
            state.ollama_client,
        )

    @app.get("/stream")
    async def stream_synthetic_scores(request: Request) -> StreamingResponse:
        return StreamingResponse(
            synthetic_transaction_sse_stream(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/analyst")
    async def serve_analyst() -> FileResponse:
        return FileResponse(
            root / "static" / "analyst.html",
            media_type="text/html",
        )


@asynccontextmanager
async def _lifespan(app: FastAPI):
    root = _project_root()
    models_dir = root / "models"
    chroma_dir = root / "chroma_db"

    xgb_model, iso_model, scaler, feature_columns = load_models_and_scaler(models_dir)
    app.state.xgb_model = xgb_model
    app.state.iso_model = iso_model
    app.state.scaler = scaler
    app.state.feature_columns = feature_columns

    app.state.shap_explainer = joblib.load(models_dir / "shap_explainer.pkl")
    app.state.threshold = float((models_dir / "threshold.txt").read_text(encoding="utf-8").strip())

    app.state.iso_norm_scaler = _fit_iso_minmax_scaler(
        iso_model,
        scaler,
        root / "data" / "creditcard.csv",
    )

    chroma_client, fraud_collection = load_chroma_collection(
        chroma_dir,
        collection_name="fraud_cases",
    )
    app.state.chroma_client = chroma_client
    app.state.chroma_collection = fraud_collection

    app.state.ollama_client = OllamaClient(
        host="http://localhost:11434",
        timeout=10.0,
    )
    app.state.seeded_transaction_count = 0

    print("[LIFESPAN] Loading creditcard dataset for synthetic transaction streaming (10,000 rows max)")
    creditcard_path = root / "data" / "creditcard.csv"
    app.state.transactions_df = pd.read_csv(creditcard_path, nrows=10_000)
    print(f"[LIFESPAN] Loaded {len(app.state.transactions_df)} transactions for streaming")

    print("FraudSentinel backend ready")
    yield


def create_app() -> FastAPI:
    # Construct FastAPI app, mount static files, register routes, and attach lifespan hooks to load models and Chroma once.
    app = FastAPI(title="FraudSentinel", lifespan=_lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    static_dir = _project_root() / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    register_routes(app)
    return app


# Create app instance for uvicorn to use when imported (e.g., uvicorn main:app)
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:create_app", factory=True, host="0.0.0.0", port=8000, reload=True)

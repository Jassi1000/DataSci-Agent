# backend/agents/chat_agent.py
import json
import os

import pandas as pd


class ChatAgent:
    def __init__(self):
        self.chunks = []
        self.index = None
        self.embed_model = None
        self._use_faiss = False

    def build_index(self, df: pd.DataFrame, pipeline_outputs: dict) -> None:
        from utils.serializer import make_serializable

        self.chunks = [str(row.to_dict()) for _, row in df.head(300).iterrows()]
        summary = json.dumps(make_serializable(pipeline_outputs), indent=2)
        for i in range(0, min(len(summary), 8000), 500):
            self.chunks.append(summary[i : i + 500])
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(self.chunks).astype("float32")
            idx = faiss.IndexFlatL2(embeddings.shape[1])
            idx.add(embeddings)
            self.index = idx
            self.embed_model = model
            self._use_faiss = True
        except ImportError:
            self._use_faiss = False
        except Exception:
            self._use_faiss = False

    def retrieve(self, query: str, k: int = 5) -> list:
        if self._use_faiss and self.index is not None:
            q_emb = self.embed_model.encode([query]).astype("float32")
            _, idxs = self.index.search(q_emb, k)
            return [self.chunks[i] for i in idxs[0] if i < len(self.chunks)]
        query_words = set(query.lower().split())
        scored = [(sum(w in c.lower() for w in query_words), c) for c in self.chunks]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for score, c in scored[:k] if score > 0] or self.chunks[:k]

    def answer(self, question: str, history: list, pipeline_outputs: dict) -> str:
        context = "\n---\n".join(self.retrieve(question))
        try:
            from groq import Groq

            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise RuntimeError("GROQ_API_KEY is not configured.")
            client = Groq(api_key=api_key)
            safe_history = [
                {"role": item.get("role"), "content": str(item.get("content", ""))}
                for item in (history or [])
                if item.get("role") in {"user", "assistant"} and item.get("content")
            ]
            messages_payload = [
                {
                    "role": "system",
                    "content": (
                        "You are a precise data analyst. Answer only from the provided dataset "
                        "and pipeline context. Be concise and factual. If the context does not "
                        "contain the answer, say so clearly."
                    ),
                },
                *safe_history,
                {
                    "role": "user",
                    "content": f"Dataset & Analysis Context:\n{context}\n\nQuestion: {question}",
                },
            ]
            resp = client.chat.completions.create(
                model=os.getenv("LLM_MODEL_NAME", "llama-3.3-70b-versatile"),
                max_tokens=1000,
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
                messages=messages_payload,
            )
            return resp.choices[0].message.content
        except Exception:
            return self._fallback_answer(question, pipeline_outputs)

    def _fallback_answer(self, question, pipeline_outputs):
        ml = pipeline_outputs.get("ml_results", {})
        explain = pipeline_outputs.get("explainability_results", {})
        report = pipeline_outputs.get("report_results", {})
        lower = question.lower()
        if "best model" in lower or "model" in lower:
            return ml.get("why_best") or f"The best model is {ml.get('best_model_name', 'not available')}."
        if "feature" in lower or "important" in lower:
            features = explain.get("top_features") or []
            return "Top features: " + ", ".join(features[:5]) if features else "Feature importance is not available for this run."
        if "summar" in lower or "finding" in lower:
            findings = report.get("key_findings") or []
            return " ".join(findings[:5]) if findings else report.get("executive_summary", "No report summary is available.")
        return "I could not reach the Groq API, so I used local retrieval. The closest context is: " + "\n".join(self.retrieve(question, 3))

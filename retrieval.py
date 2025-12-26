import os

import torch
from sentence_transformers import SentenceTransformer, util

from constants import EMBEDDINGS_FILE
from utils import load_faqs


def load_embeddings() -> torch.Tensor:
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    if os.path.exists(EMBEDDINGS_FILE):
        return torch.load(EMBEDDINGS_FILE)

    faqs = load_faqs()

    questions = [item["question"] for item in faqs]

    embeddings = model.encode(
        questions,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    torch.save(embeddings, EMBEDDINGS_FILE)
    return embeddings


def retrieve_top_k_faqs(
    query: str, faqs: list[dict], faq_embeddings: torch.Tensor, k: int = 3
) -> dict[str, dict | list[dict]]:
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    query_embedding = model.encode(
        query, convert_to_tensor=True, normalize_embeddings=True
    )

    scores = util.cos_sim(query_embedding, faq_embeddings)[0]
    top_k = torch.topk(scores, k=k).indices.tolist()

    results = [
        {
            "question": faqs[idx]["question"],
            "answer": faqs[idx]["answer"],
            "score": float(scores[idx]),
        }
        for idx in top_k
    ]

    return {"best_answer": results[0], "top_3_faqs": results}

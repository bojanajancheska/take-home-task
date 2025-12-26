import argparse
import json

from retrieval import load_embeddings, retrieve_top_k_faqs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    args = parser.parse_args()

    query = args.query

    with open("dataset.json", "r", encoding="utf-8") as f:
        faqs = json.load(f)

    embeddings = load_embeddings()

    result = retrieve_top_k_faqs(query, faqs, embeddings, k=3)
    best = result["best_answer"]

    print(f"\nQ: {best['question']}")
    print(f"A: {best['answer']}")
    print(f"Confidence: {best['score']:.3f}")

    print("\n Top 3 most relevant FAQs:")
    for faq in result["top_3_faqs"]:
        print(f"\nQ: {faq['question']}")
        print(f"A: {faq['answer']}")
        print(f"Score: {faq['score']:.3f}")


if __name__ == "__main__":
    main()

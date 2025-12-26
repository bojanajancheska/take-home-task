import json
import os

from constants import DATASET_FILE, TOPICS
from utils import generate_qa


def main():
    if os.path.exists(DATASET_FILE):
        with open(DATASET_FILE, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    else:
        dataset = []

    while len(dataset) < 10:
        idx = len(dataset) + 1
        try:
            existing_q = [item["question"] for item in dataset]
            topic = TOPICS[len(dataset) % len(TOPICS)]
            qa = generate_qa(existing_q, topic)

            if qa["question"] in existing_q:
                print("Exact duplicate question found, retrying...")
                continue

            dataset.append(qa)

            with open(DATASET_FILE, "w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)

            print(f"Generated {idx}/10")

        except Exception:
            print(f"Error occurred at index {idx}, retrying...")


if __name__ == "__main__":
    main()

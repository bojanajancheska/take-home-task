import json
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer

from constants import DATASET_FILE


def load_faqs() -> list[dict[str, str]]:
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_prompt(existing_q: str, topic: str) -> str:
    return f"""
        Generate ONE customer support question–answer pair.

        Topic: {topic}

        Do NOT repeat or paraphrase any of these questions:

        {existing_q}

        Return ONLY valid JSON:
        {{
        "question": "string",
        "answer": "string"
        }}
    """


def load_model_and_tokenizer() -> tuple[Any, Any]:
    model = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model)

    model = AutoModelForCausalLM.from_pretrained(  # type: ignore
        model,
        dtype="auto",
        device_map="auto",
    )

    return model, tokenizer


def generate_qa(existing_q: str, topic: str) -> dict[str, str]:
    model, tokenizer = load_model_and_tokenizer()

    prompt = get_prompt(existing_q, topic)
    messages = [{"role": "user", "content": prompt}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.9,
        top_p=0.9,
        repetition_penalty=1.2,
        do_sample=True,
    )

    output = tokenizer.decode(
        generated_ids[0][len(inputs.input_ids[0]) :],
        skip_special_tokens=True,
    ).strip()

    try:
        return json.loads(output)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON")

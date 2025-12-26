# FAQ Retrieval Assistant

## Approach

### FAQ Dataset Generation

A fixed list of customer support topics is defined in `constants.py`.

For each topic:

- A single **question–answer (QA) pair** is generated using a causal language model (**Qwen/Qwen3-1.7B**).
- The model is prompted to:
  - Generate **exactly one QA pair**
  - **Avoid repeating or paraphrasing** previously generated questions
  - **Return valid JSON only**

Generated QA entries are appended to `dataset.json` until the dataset reaches **10 FAQs**.

### FAQ Retrieval

Once the dataset is created:

- All FAQ questions are embedded using a **SentenceTransformer** model.
- A user query is embedded into the same vector space.
- **Cosine similarity** is used to retrieve the top-K most relevant FAQs.

The system returns:

- The **best matching answer**
- The **top 3 most relevant FAQs**, along with similarity scores

## Tools Used

### Language Models & NLP

- **Transformers** – loading and running the Qwen causal language model
- **Sentence-Transformers** – generating semantic embeddings
- **PyTorch** – tensor operations and similarity computation

### Models

- **Qwen/Qwen3-1.7B** – question–answer generation
- **paraphrase-multilingual-MiniLM-L12-v2** – sentence embeddings

### Utilities

- **argparse** – command-line interface
- **JSON** – dataset persistence
- **black**, **isort**, **flake8**, **mypy** – formatting, linting, and type checking

## How to Run the Project

### 1. Install Dependencies

```bash
uv sync
```

### 2. Generate FAQ dataset

The generated dataset is already included in the project, by running this command you can regenerate it.

```bash
uv run generating_dataset.py
```

This will:

- Generate up to 10 QA pairs
- Store them in dataset.json

### 3. User query

```bash
uv run main.py --query "I cannot log in to my account"
```

### Example output

```
Q: How can I reset my account password?
A: You can reset your account password by visiting the login page and clicking on 'Forgot Password.' Follow the instructions to verify your email address and set a new password.
Confidence: 0.763

Top 3 most relevant FAQs:

Q: How can I reset my account password?
A: You can reset your account password by visiting the login page and clicking on 'Forgot Password.' Follow the instructions to verify your email address and set a new password.
Score: 0.763

Q: How can I enable two-factor authentication on my account?
A: You can enable two-factor authentication by going to your account settings, selecting the security tab, and following the instructions to set up a verification method such as an authenticator app or SMS code.
Score: 0.576

Q: How can I change my account's email address?
A: To change your account's email address, log into your account dashboard, navigate to the profile settings, and follow the instructions to edit your contact information.
Score: 0.551
```

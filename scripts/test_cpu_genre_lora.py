from openai import OpenAI
from transformers import AutoTokenizer

from src.data.loader import load_story_dataset
from src.classification.prompts import build_classification_messages


STORY_ID = "960869"
BASE_MODEL = "Qwen/Qwen3-4B"
MAX_LENGTH = 2048

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="not-needed",
)

# Same tokenizer used by the HF/Kaggle evaluation
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

df = load_story_dataset()

row = df[df["id"].astype(str) == STORY_ID].iloc[0]
genre_labels = sorted(df["genre"].unique().tolist())

messages = build_classification_messages(
    story=row["story"],
    genre_labels=genre_labels,
)

# Build exactly the same Qwen chat-formatted prompt
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)

# Same truncation policy as the HF classifier
input_ids = tokenizer(
    prompt,
    truncation=True,
    max_length=MAX_LENGTH,
    add_special_tokens=False,
)["input_ids"]

truncated_prompt = tokenizer.decode(
    input_ids,
    skip_special_tokens=False,
)

print(f"Input tokens: {len(input_ids)}")
print(
    f"Truncated:    {len(tokenizer.encode(prompt, add_special_tokens=False)) > MAX_LENGTH}"
)

# Important:
# We already applied the Qwen chat template ourselves, so send the final
# formatted prompt through /v1/completions rather than /v1/chat/completions.
response = client.completions.create(
    model="qwen",
    prompt=truncated_prompt,
    temperature=0,
    max_tokens=32,
)

raw_output = response.choices[0].text.strip()

# Defensive cleanup in case llama.cpp exposes a trailing special token as text.
prediction = raw_output.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()

print("=" * 60)
print(f"ID:        {row['id']}")
print(f"Title:     {row['title']}")
print(f"Expected:  {row['genre']}")
print(f"Predicted: {prediction!r}")
print(f"Valid:     {prediction in genre_labels}")
print(f"Raw:       {raw_output!r}")
print("=" * 60)

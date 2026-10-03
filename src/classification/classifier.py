# src/classification/classifier.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import joblib

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from .prompts import build_classification_messages


@dataclass
class ClassificationResult:
    predicted_genre: str
    raw_output: str
    is_valid: bool


class QwenGenreClassifier:
    """
    Genre classifier based on Qwen3-4B with a PEFT LoRA adapter.

    The base model is loaded separately from the adapter so the same
    underlying Qwen model can later be reused by other system components.
    """

    def __init__(
        self,
        base_model_name: str,
        adapter_path: str,
        genre_labels: list[str],
        device_map: str = "auto",
    ) -> None:
        self.genre_labels = genre_labels
        self.genre_set = set(genre_labels)

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=quantization_config,
            device_map=device_map,
        )

        self.model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
        )

        self.model.eval()

    @torch.inference_mode()
    def classify(
        self,
        story: str,
        max_new_tokens: int = 32,
    ) -> ClassificationResult:

        messages = build_classification_messages(
            story=story,
            genre_labels=self.genre_labels,
        )

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        generated_tokens = outputs[0, inputs["input_ids"].shape[1] :]

        raw_output = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        predicted_genre = raw_output.strip()

        return ClassificationResult(
            predicted_genre=predicted_genre,
            raw_output=raw_output,
            is_valid=predicted_genre in self.genre_set,
        )


@dataclass
class CPUClassificationResult:
    predicted_genre: str


class CPUGenreClassifier:
    """
    Production CPU genre classifier.

    Uses the TF-IDF + LinearSVC model selected during Task 3
    for low-latency CPU deployment.
    """

    def __init__(
        self,
        model_dir: str | Path = "models/tfidf_model",
    ) -> None:
        model_dir = Path(model_dir)

        vectorizer_path = model_dir / "tfidf_vectorizer.joblib"
        classifier_path = model_dir / "linear_svc.joblib"
        encoder_path = model_dir / "label_encoder.joblib"

        required_files = [
            vectorizer_path,
            classifier_path,
            encoder_path,
        ]

        missing = [str(path) for path in required_files if not path.exists()]

        if missing:
            raise FileNotFoundError(
                "Missing CPU classifier artifacts: " + ", ".join(missing)
            )

        self.vectorizer = joblib.load(vectorizer_path)
        self.classifier = joblib.load(classifier_path)
        self.label_encoder = joblib.load(encoder_path)

    def classify(
        self,
        story: str,
    ) -> CPUClassificationResult:
        if not story or not story.strip():
            raise ValueError("Story text cannot be empty.")

        features = self.vectorizer.transform([story])

        encoded_prediction = self.classifier.predict(features)

        predicted_genre = self.label_encoder.inverse_transform(encoded_prediction)[0]

        return CPUClassificationResult(
            predicted_genre=str(predicted_genre),
        )

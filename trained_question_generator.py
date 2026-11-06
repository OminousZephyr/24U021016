"""Automatic Question Generation using a pretrained T5 question-generation model.

This is inference with an already-trained model, not model training/fine-tuning.
First run downloads the model from Hugging Face and caches it locally.
"""

from __future__ import annotations

import re
from typing import Iterable

import spacy
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "valhalla/t5-small-qg-hl"
SPACY_MODEL = "en_core_web_sm"


class QuestionGenerator:
    """Generate questions by highlighting candidate answers and using a trained T5 model."""

    def __init__(self, model_name: str = MODEL_NAME, max_input_tokens: int = 512):
        # Keep the tokenizer exclusion workaround required by this user's setup.
        # spaCy still supplies POS tags, dependencies, noun chunks, and entities.
        self.nlp = spacy.load(SPACY_MODEL, exclude=["tokenizer"])
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()
        self.max_input_tokens = max_input_tokens
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @staticmethod
    def _clean_question(text: str) -> str:
        text = " ".join(text.strip().split())
        text = text.strip(" \t\r\n\"'`-•")
        if not text:
            return ""
        text = text[0].upper() + text[1:]
        if text[-1] not in "?!.":
            text += "?"
        # Model sometimes ends a question with a period.
        if text.endswith("."):
            text = text[:-1] + "?"
        return text

    @staticmethod
    def _candidate_answers(sentence) -> list[str]:
        """Find answer spans using named entities and noun chunks, not every token."""
        candidates: list[tuple[int, str]] = []

        # Named entities are often useful answers (people, places, dates, organisations).
        for ent in sentence.ents:
            value = ent.text.strip(" ,.;:")
            if value and len(value.split()) <= 8:
                candidates.append((0, value))

        # Noun chunks provide additional answer candidates when the sentence has no entity.
        try:
            for chunk in sentence.noun_chunks:
                value = chunk.text.strip(" ,.;:")
                if value and len(value.split()) <= 7:
                    # Prefer meaningful noun phrases over very short generic words.
                    candidates.append((1 if len(value.split()) > 1 else 2, value))
        except (ValueError, NotImplementedError):
            pass

        # Stable, case-insensitive deduplication; keep entity candidates first.
        seen: set[str] = set()
        result: list[str] = []
        for _, value in sorted(candidates, key=lambda item: item[0]):
            key = value.casefold()
            if key in seen or key in {"it", "they", "he", "she", "we", "i", "you", "this", "that"}:
                continue
            if not re.search(r"[A-Za-z0-9]", value):
                continue
            seen.add(key)
            result.append(value)
        return result

    @staticmethod
    def _remove_answer_once(sentence: str, answer: str) -> str:
        """Remove one exact occurrence; the model format puts answer before its context."""
        match = re.search(re.escape(answer), sentence, flags=re.IGNORECASE)
        if match:
            return (sentence[:match.start()] + sentence[match.end():]).strip(" ,;:-")
        return sentence

    def _generate_one(self, sentence: str, answer: str, num_beams: int = 4) -> str:
        context = self._remove_answer_once(sentence, answer)
        prompt = f"<hl> {answer} <hl> {context} </s>"
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=64,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2,
            )
        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return self._clean_question(decoded)

    def generate_questions(
        self,
        text: str,
        max_questions: int = 8,
        max_per_sentence: int = 3,
    ) -> list[str]:
        """Return up to max_questions questions from a paragraph."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text.strip() or max_questions <= 0 or max_per_sentence <= 0:
            return []

        doc = self.nlp(text)
        questions: list[str] = []
        seen_questions: set[str] = set()

        for sentence in doc.sents:
            sent_text = sentence.text.strip()
            if not sent_text:
                continue

            candidates = self._candidate_answers(sentence)
            sentence_count = 0
            for answer in candidates:
                question = self._generate_one(sent_text, answer)
                if not question or "?" not in question:
                    continue
                # Avoid outputs that simply repeat the original sentence.
                if re.sub(r"\W+", " ", question).casefold() == re.sub(r"\W+", " ", sent_text).casefold():
                    continue
                key = question.casefold()
                if key in seen_questions:
                    continue
                seen_questions.add(key)
                questions.append(question)
                sentence_count += 1
                if sentence_count >= max_per_sentence or len(questions) >= max_questions:
                    break
            if len(questions) >= max_questions:
                break

        return questions


def main() -> None:
    print("Automatic Question Generation (trained T5 model)")
    print("The first run downloads the model; later runs use the local cache.\n")
    generator = QuestionGenerator()
    sample = (
        "Marie Curie studied radioactivity in Paris. "
        "Albert Einstein developed the theory of relativity in 1915. "
        "Photosynthesis is the process by which plants convert sunlight into energy."
    )
    print("TEXT:\n", sample, "\n", sep="")
    for index, question in enumerate(generator.generate_questions(sample), start=1):
        print(f"{index}. {question}")


if __name__ == "__main__":
    main()

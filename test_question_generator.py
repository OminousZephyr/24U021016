"""Tests for question_generator.py.

Run from the project directory with:
    python -m unittest -v test_question_generator.py

Tests that need the trained spaCy parser are skipped if en_core_web_sm is not
installed. Install it with `python -m spacy download en_core_web_sm` to run them.
"""

import unittest

from question_generator import (
    _lower_initial_article,
    _normalise,
    _nlp,
    generate_questions,
)


ALICE = (
    "Alice was beginning to get very tired of sitting by her sister on the bank, "
    "and of having nothing to do. Once or twice she had peeped into the book "
    "her sister was reading, but it had no pictures or conversations in it. "
    '"And what is the use of a book," thought Alice, "without pictures or conversations?"'
)

try:
    _nlp()
    MODEL_AVAILABLE = True
except RuntimeError:
    MODEL_AVAILABLE = False


class UtilityTests(unittest.TestCase):
    def test_normalise_removes_punctuation_and_case(self):
        self.assertEqual(_normalise("What is a Book?"), "what is a book")

    def test_lower_initial_article(self):
        self.assertEqual(_lower_initial_article("The scientist"), "the scientist")
        self.assertEqual(_lower_initial_article("Marie Curie"), "Marie Curie")

    def test_empty_text_returns_empty_list(self):
        self.assertEqual(generate_questions("   "), [])

    def test_non_string_raises_type_error(self):
        with self.assertRaises(TypeError):
            generate_questions(None)  # type: ignore[arg-type]


@unittest.skipUnless(MODEL_AVAILABLE, "requires the en_core_web_sm spaCy model")
class ModelIntegrationTests(unittest.TestCase):
    def test_science_sentence_generates_action_questions(self):
        questions = generate_questions("The scientist discovered a new planet.")
        self.assertIn("What did the scientist discover?", questions)
        self.assertIn("Who discovered a new planet?", questions)

    def test_location_question_is_about_the_action(self):
        questions = generate_questions("The Amazon River flows through South America.")
        self.assertIn("Where does the Amazon River flow?", questions)

    def test_date_question_keeps_the_main_object(self):
        questions = generate_questions(
            "Albert Einstein developed the theory of relativity in 1915."
        )
        self.assertIn(
            "When did Albert Einstein develop the theory of relativity?",
            questions,
        )

    def test_definition_question_requires_a_definition_pattern(self):
        questions = generate_questions(
            "Photosynthesis is the process by which plants convert sunlight into energy."
        )
        self.assertIn("What is Photosynthesis?", questions)

    def test_alice_paragraph_does_not_turn_each_noun_into_a_definition(self):
        questions = generate_questions(ALICE)
        self.assertTrue(questions)
        for question in questions:
            self.assertNotRegex(
                question.casefold(),
                r"^what is (?:alice|sister|book|picture|conversation)\?$",
            )
        self.assertLessEqual(len(questions), 10)
        normalized = [_normalise(question) for question in questions]
        self.assertEqual(len(normalized), len(set(normalized)))

    def test_max_questions_is_respected(self):
        questions = generate_questions(
            "The scientist discovered a new planet. Marie Curie studied radioactivity. "
            "Albert Einstein developed the theory of relativity in 1915.",
            max_questions=2,
        )
        self.assertLessEqual(len(questions), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)

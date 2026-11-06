"""A small, explainable rule-based automatic question generator.

Requirements:
    pip install spacy
    python -m spacy download en_core_web_sm

In environments where the saved spaCy tokenizer model has a loading issue,
we load the pipeline with exclude=["tokenizer"]. The default spaCy tokenizer
is then used alongside the trained tagger, parser and entity recognizer.

This is a coursework-sized heuristic, not a general-purpose question model.
It aims to produce a few text-grounded questions and skips uncertain patterns
instead of making a question out of every noun.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

import spacy
from spacy.tokens import Doc, Span, Token


MODEL_NAME = "en_core_web_sm"

# Common human-role nouns that spaCy may not label as PERSON.
HUMAN_WORDS = {
    "author", "doctor", "engineer", "explorer", "inventor", "king",
    "queen", "researcher", "scientist", "student", "teacher", "writer",
    "professor", "president", "artist", "chemist", "physicist",
}

LOCATION_LABELS = {"GPE", "LOC", "FAC"}
TIME_LABELS = {"DATE", "TIME"}
LOCATION_PREPOSITIONS = {
    "at", "in", "on", "near", "under", "over", "through", "across",
    "around", "inside", "outside", "beneath", "beside", "behind",
}


def _load_nlp():
    """Load spaCy once, retaining the tokenizer workaround used in this setup."""
    try:
        return spacy.load(MODEL_NAME, exclude=["tokenizer"])
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model {MODEL_NAME!r} is missing. In the active environment, run:\n"
            f"  python -m spacy download {MODEL_NAME}\n"
            "Then run this program again."
        ) from exc


@lru_cache(maxsize=1)
def _nlp():
    return _load_nlp()


def _clean_space(text: str) -> str:
    return " ".join(text.split()).strip()


def _normalise(text: str) -> str:
    """Lowercase and remove punctuation for duplicate/source-question checks."""
    return re.sub(r"[^\w\s]", "", text.casefold()).strip()


def _sentence_noun_chunk(token: Token, sent: Span) -> str:
    """Return the noun phrase containing token, or the token itself as fallback."""
    try:
        matches = [
            chunk for chunk in token.doc.noun_chunks
            if chunk.start <= token.i < chunk.end
            and chunk.start >= sent.start and chunk.end <= sent.end
        ]
        if matches:
            # Usually there is one match; shortest is safer if parses overlap.
            chunk = min(matches, key=lambda item: len(item))
            return _clean_space(chunk.text)
    except (ValueError, NotImplementedError):
        # Some malformed or partial parses may not provide noun chunks.
        pass
    return token.text


def _lower_initial_article(phrase: str) -> str:
    """Use natural lowercase articles after question auxiliaries."""
    words = phrase.split()
    if words and words[0].casefold() in {"the", "a", "an"}:
        words[0] = words[0].lower()
    return " ".join(words)


def _direct_subject(verb: Token) -> Token | None:
    for child in verb.children:
        if child.dep_.lower() in {"nsubj", "nsubjpass", "csubj", "nsubj:pass"}:
            return child
    return None


def _get_subject(verb: Token, sent: Span) -> Token | None:
    """Find a subject, including a subject inherited by a coordinated/embedded verb."""
    subject = _direct_subject(verb)
    if subject is not None:
        return subject

    # A verb such as "sitting" in "Alice was tired of sitting ..." often shares
    # its subject with a nearby governing verb. Walk up a short dependency path.
    current = verb.head
    steps = 0
    while current != verb and current.i >= sent.start and current.i < sent.end and steps < 5:
        subject = _direct_subject(current)
        if subject is not None:
            return subject
        if current.head == current:
            break
        current = current.head
        steps += 1
    return None


def _auxiliaries(verb: Token, sent: Span) -> list[Token]:
    """Get direct auxiliaries, or inherit an auxiliary from a governing verb."""
    def direct_aux(node: Token) -> list[Token]:
        return sorted(
            [child for child in node.children if child.dep_.lower() in {"aux", "auxpass", "aux:pass"}],
            key=lambda item: item.i,
        )

    aux = direct_aux(verb)
    if aux:
        return aux

    current = verb.head
    steps = 0
    while current != verb and current.i >= sent.start and current.i < sent.end and steps < 5:
        aux = direct_aux(current)
        if aux:
            return aux
        if current.head == current:
            break
        current = current.head
        steps += 1
    return []


def _verb_question_parts(verb: Token, subject_text: str, sent: Span) -> tuple[str, str]:
    """Return (auxiliary phrase, verb form) for a basic question template."""
    auxiliaries = _auxiliaries(verb, sent)
    if auxiliaries:
        aux_phrase = " ".join(token.text.lower() for token in auxiliaries)
        # With an auxiliary, retain the surface verb form: was reading, had studied.
        return aux_phrase, verb.text.lower()

    tag = verb.tag_
    if tag in {"VBD", "VBN"}:
        return "did", verb.lemma_.lower()
    if tag == "VBZ":
        return "does", verb.lemma_.lower()
    return "do", verb.lemma_.lower()


def _make_clause(verb: Token, subject_text: str, sent: Span, tail: str = "") -> str:
    """Build a clause after What/When/Where, keeping basic tense where possible."""
    auxiliaries = _auxiliaries(verb, sent)
    if auxiliaries:
        aux_phrase = " ".join(token.text.lower() for token in auxiliaries)
        verb_form = verb.text.lower()
        clause = f"{aux_phrase} {subject_text} {verb_form}"
    elif verb.tag_ in {"VBD", "VBN"}:
        clause = f"did {subject_text} {verb.lemma_.lower()}"
    elif verb.tag_ == "VBZ":
        clause = f"does {subject_text} {verb.lemma_.lower()}"
    else:
        clause = f"do {subject_text} {verb.lemma_.lower()}"

    if tail:
        clause += f" {tail}"
    return _clean_space(clause)


def _is_person(subject: Token, sent: Span) -> bool:
    phrase = _sentence_noun_chunk(subject, sent)
    if any(ent.label_ == "PERSON" and ent.start <= subject.i < ent.end for ent in subject.doc.ents):
        return True
    phrase_words = {word.strip(".,;:!?()[]{}\"'").casefold() for word in phrase.split()}
    if phrase_words & HUMAN_WORDS:
        return True
    return False


def _object_token(verb: Token) -> Token | None:
    for child in verb.children:
        if child.dep_.lower() in {"dobj", "obj"}:
            return child
    return None


def _prep_objects(verb: Token) -> Iterable[tuple[Token, Token]]:
    """Yield (preposition, object-of-preposition) pairs attached to the verb."""
    for child in verb.children:
        if child.dep_.lower() == "prep":
            for grandchild in child.children:
                if grandchild.dep_.lower() in {"pobj", "obj"}:
                    yield child, grandchild


def _entity_for_token(doc: Doc, token: Token) -> Span | None:
    for ent in doc.ents:
        if ent.start <= token.i < ent.end:
            return ent
    return None


def _candidate_questions_for_verb(verb: Token, sent: Span, doc: Doc) -> list[str]:
    candidates: list[str] = []
    if verb.pos_ != "VERB" or verb.lemma_.lower() == "be":
        return candidates

    subject = _get_subject(verb, sent)
    if subject is None or subject.dep_.lower() == "expl":
        return candidates

    subject_text = _lower_initial_article(_sentence_noun_chunk(subject, sent))
    if not subject_text:
        return candidates

    obj = _object_token(verb)
    object_text = _sentence_noun_chunk(obj, sent) if obj is not None else ""

    # Passive voice with a named agent: "Who painted the Mona Lisa?"
    is_passive_subject = any(child.dep_.lower() in {"nsubjpass", "nsubj:pass"} for child in verb.children)
    if is_passive_subject:
        agent = None
        for prep, pobj in _prep_objects(verb):
            if prep.text.lower() == "by":
                agent = pobj
                break
        if agent is not None:
            patient_text = subject_text
            candidates.append(f"Who {verb.text.lower()} {patient_text}?")

    # Direct-object questions: these usually preserve the main fact in the sentence.
    if obj is not None:
        clause = _make_clause(verb, subject_text, sent)
        candidates.append(f"What {clause}?")

        # Ask who performed the action when the subject is a person.
        if _is_person(subject, sent):
            auxiliaries = _auxiliaries(verb, sent)
            if auxiliaries:
                lead = " ".join(token.text.lower() for token in auxiliaries)
                who_clause = f"Who {lead} {verb.text.lower()} {object_text}?"
            else:
                # Surface form handles both "studied" and "discovers".
                who_clause = f"Who {verb.text.lower()} {object_text}?"
            candidates.append(who_clause)

    # Ask about a date/time attached to the event; retain the direct object when possible.
    sentence_entities = [ent for ent in doc.ents if sent.start <= ent.start and ent.end <= sent.end]
    if any(ent.label_ in TIME_LABELS for ent in sentence_entities):
        clause = _make_clause(verb, subject_text, sent, object_text)
        candidates.append(f"When {clause}?")

    # Ask where if a location entity is attached in the sentence.
    has_location_entity = any(ent.label_ in LOCATION_LABELS for ent in sentence_entities)
    if has_location_entity:
        clause = _make_clause(verb, subject_text, sent)
        candidates.append(f"Where {clause}?")

    # Location prepositions can provide a useful Where question even without NER.
    # We only use a small, explicit list so "peeped into a book" is not mistaken
    # for a physical-location question.
    for prep, prep_obj in _prep_objects(verb):
        if prep.text.lower() in LOCATION_PREPOSITIONS:
            clause = _make_clause(verb, subject_text, sent)
            candidates.append(f"Where {clause}?")
            break
        if prep.text.lower() in {"into", "about", "with", "for", "to", "from"} and obj is None:
            prep_object = _sentence_noun_chunk(prep_obj, sent)
            clause = _make_clause(verb, subject_text, sent)
            # Do not add an object noun twice if parser treats it as a direct object too.
            candidates.append(f"What {clause} {prep.text.lower()}?")
            break

    # Complement clauses permit a compact question such as "What did Alice think?"
    if any(child.dep_.lower() == "ccomp" for child in verb.children):
        clause = _make_clause(verb, subject_text, sent)
        candidates.append(f"What {clause}?")

    return candidates


def _definition_questions(sent: Span) -> list[str]:
    """Make a definition question only for an actual noun complement ("X is Y")."""
    candidates: list[str] = []
    for token in sent:
        # spaCy often represents "Photosynthesis is a process" with "process"
        # as the sentence root and "is" as its child (dep_ == "cop"). In other
        # parses, the copula itself heads the clause. Support both structures.
        if token.lemma_.lower() != "be":
            continue

        if token.dep_.lower() == "cop" and token.head != token:
            complement = token.head
            subject = _direct_subject(complement)
        else:
            subject = _direct_subject(token)
            complement = next(
                (child for child in token.children if child.dep_.lower() in {"attr", "acomp"}),
                None,
            )

        if subject is None or complement is None:
            continue
        if complement.pos_ not in {"NOUN", "PROPN"}:
            continue

        subject_text = _sentence_noun_chunk(subject, sent)
        if not subject_text or subject_text.casefold() in {"it", "he", "she", "they", "we", "i", "you"}:
            continue
        # Keep proper names capitalized; common nouns/articles may be lowercase.
        if subject.pos_ == "PROPN" or subject.ent_type_ == "PERSON":
            subject_text = _sentence_noun_chunk(subject, sent)
        else:
            subject_text = _lower_initial_article(subject_text)
        candidates.append(f"What is {subject_text}?")
    return candidates


def generate_questions(text: str, max_questions: int = 10) -> list[str]:
    """Generate a small set of text-grounded questions from a paragraph.

    Args:
        text: A paragraph or a few sentences.
        max_questions: Maximum number of questions to return (default 10).

    Returns:
        A de-duplicated list of questions. Some grammar and interpretation errors
        are possible because this is a small, template-based generator.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip():
        return []
    if max_questions < 1:
        return []

    doc = _nlp()(text)
    candidates: list[str] = []

    for sent in doc.sents:
        # First use verb/action structure and named entities.
        for token in sent:
            candidates.extend(_candidate_questions_for_verb(token, sent, doc))
        # Definition question only when a genuine copular definition was parsed.
        candidates.extend(_definition_questions(sent))

    # Normalize, remove duplicates, skip questions already present in the source,
    # and suppress malformed one- or two-word outputs.
    source_normalized = _normalise(text)
    results: list[str] = []
    seen: set[str] = set()
    for question in candidates:
        question = _clean_space(question)
        if not question.endswith("?") or len(question.split()) < 4:
            continue
        key = _normalise(question)
        if not key or key in seen:
            continue
        # Avoid asking a question that the paragraph already asks verbatim.
        if key in source_normalized:
            continue
        # Common unhelpful outputs from shallow parses.
        if re.search(r"\b(?:is|are|was|were)\s+(?:a|an|the)?\s*(?:alice|sister|book|picture|conversation)\?", question, re.I):
            continue
        seen.add(key)
        results.append(question)
        if len(results) >= max_questions:
            break

    return results


if __name__ == "__main__":
    examples = {
        "Alice excerpt": (
            "Alice was beginning to get very tired of sitting by her sister on the bank, "
            "and of having nothing to do. Once or twice she had peeped into the book "
            "her sister was reading, but it had no pictures or conversations in it. "
            '"And what is the use of a book," thought Alice, "without pictures or conversations?"'
        ),
        "Science": "The scientist discovered a new planet.",
        "Person and topic": "Marie Curie studied radioactivity.",
        "Location": "The Amazon River flows through South America.",
        "Date": "Albert Einstein developed the theory of relativity in 1915.",
        "Definition": "Photosynthesis is the process by which plants convert sunlight into energy.",
    }

    for name, paragraph in examples.items():
        print(f"\n{name}: {paragraph}")
        questions = generate_questions(paragraph)
        if questions:
            for index, question in enumerate(questions, start=1):
                print(f"  {index}. {question}")
        else:
            print("  (No confident questions found.)")

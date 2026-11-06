# Automatic Question Generation Using T5

An NLP-based application that automatically generates questions from a given paragraph using a pretrained and fine-tuned T5 question-generation model.

## 1. Project Overview

Automatic Question Generation (AQG) is a Natural Language Processing (NLP) task that involves generating meaningful questions from textual content.

This project implements an automatic question-generation system that accepts a paragraph as input, identifies potential answer candidates, and generates questions based on those candidates.

The system combines **spaCy** for linguistic preprocessing and candidate-answer extraction with a **T5 (Text-to-Text Transfer Transformer)** model for question generation.

A Streamlit-based web interface allows users to interact with the application without requiring command-line input.

**Important:** This project uses an existing pretrained and fine-tuned model. It performs inference and does not train or fine-tune the model from scratch.

## 2. Objectives

- Generate questions automatically from user-provided text.
- Identify potential answers using Named Entity Recognition (NER) and noun-phrase extraction.
- Use a pretrained T5 model to generate natural-language questions.
- Apply text cleaning and duplicate removal to improve output readability.
- Provide a simple and accessible web interface.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| spaCy | Sentence processing, named entity recognition, and noun-chunk extraction |
| Hugging Face Transformers | Loading the pretrained model and tokenizer |
| T5-small QG-HL | Question generation |
| PyTorch | Neural network inference |
| SentencePiece | Subword tokenization used by the model |
| Streamlit | Web-based user interface |
| Regular Expressions (`re`) | Text cleaning and pattern matching |

## 4. Model Information

**Model:** `valhalla/t5-small-qg-hl`

**Architecture:** T5 (Text-to-Text Transfer Transformer)

**Model type:** Encoder-Decoder Transformer

The T5 model follows a text-to-text approach, where an input text sequence is transformed into an output text sequence.

In this project, the model receives a sentence containing a highlighted answer span and generates a question targeting that answer.

Example:

**Input:**

`<hl> Warsaw <hl> Marie Curie was born in Warsaw in 1867.`

**Possible output:**

`Where was Marie Curie born?`

The special `<hl>` markers identify the candidate answer that the model should focus on.

## 5. System Architecture and Workflow

The application follows a hybrid NLP pipeline combining linguistic preprocessing and neural text generation.

1. **User Input:** The user enters a paragraph through the Streamlit interface.

2. **Sentence Segmentation:** spaCy processes the paragraph and divides it into individual sentences.

3. **Candidate Answer Extraction:** spaCy identifies named entities, such as people, locations, and dates, along with noun chunks that can serve as potential answers.

4. **Candidate Filtering:** The system applies length restrictions, removes duplicate candidates, and filters certain generic pronouns.

5. **Input Preparation:** Each candidate answer is separated from its sentence context and highlighted using the special `<hl>` markers.

6. **Tokenization:** Hugging Face's AutoTokenizer converts the prepared text into numerical token IDs.

7. **Question Generation:** The T5 encoder-decoder model generates questions using PyTorch inference.

8. **Post-processing:** Generated questions undergo whitespace normalization, punctuation correction, basic filtering, and duplicate removal.

9. **Output:** The generated questions are displayed through Streamlit and can be downloaded.

## 6. Project Structure

```text
nlp/
│
├── app.py
│   └── Streamlit web application
│
├── trained_question_generator.py
│   └── Core question-generation pipeline
│
├── README.md
│   └── Project documentation
│
├── .venv313/
│   └── Python virtual environment
│
└── requirements.txt
    └── Project dependencies (if maintained)
```

The virtual environment and downloaded model cache do not need to be committed to GitHub.

## 7. Installation and Setup

### Prerequisites

- Python 3.13
- pip or uv
- Internet connection for the initial model download

### Step 1: Clone the repository

Alternatively, navigate to the directory containing the project files.

### Step 2: Create a virtual environment

```bash
python3.13 -m venv .venv
```

Activate it on Linux:

```bash
source .venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install --upgrade pip
pip install torch transformers spacy sentencepiece streamlit
```

### Step 4: Download the spaCy English model

```bash
python -m spacy download en_core_web_sm
```

The application uses spaCy's English model for sentence processing, entity recognition, and noun-chunk extraction.

### Step 5: Run the application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

`http://localhost:8501`

**Note:** The first execution downloads the pretrained T5 model from Hugging Face and stores it in the local cache. Subsequent executions can reuse the downloaded model.

If using the CPU-only PyTorch setup, the model runs on the CPU. A compatible CUDA-enabled PyTorch installation can be used when GPU inference is available.

## 8. How to Use

1. Start the Streamlit application.
2. Open the application in your browser.
3. Enter or paste a paragraph into the text input area.
4. Select the desired number of questions.
5. Click the Generate Questions button.
6. View the generated questions.
7. Download the generated questions if the download option is available.

## 9. Example

**Input Paragraph:**

Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity. She was born in Warsaw, Poland, in 1867. In 1903, she shared the Nobel Prize in Physics with Pierre Curie and Henri Becquerel.

**Possible Generated Questions:**

1. Who conducted pioneering research on radioactivity?
2. Where was Marie Curie born?
3. In what year was Marie Curie born?
4. Who shared the Nobel Prize in Physics with Marie Curie?
5. In which year did Marie Curie receive the Nobel Prize in Physics?

*Note: These are illustrative reference questions. Actual outputs may vary depending on model generation and candidate-answer selection.*

## 10. Advantages

- Reduces the manual effort required to create questions.
- Uses a pretrained model, eliminating the need for training from scratch.
- Generates multiple questions from a single paragraph.
- Combines linguistic preprocessing with neural text generation.
- Provides an accessible interface through Streamlit.

## 11. Limitations

- Generated questions may occasionally be grammatically incorrect, repetitive, or irrelevant.
- Question quality depends on the candidate answers extracted by spaCy.
- Processing sentences individually limits the use of information spread across multiple sentences.
- Model inference can be computationally expensive, particularly on a CPU.
- The current implementation does not include a formal question-quality evaluation metric or an independent factual verification mechanism.

## 12. Future Improvements

- Improve candidate-answer selection using more advanced NLP techniques.
- Introduce semantic similarity checks to identify redundant questions.
- Add question-quality evaluation using metrics such as BLEU, ROUGE, or BERTScore, with suitable reference questions.
- Support different question types, such as factual, conceptual, and inferential questions.
- Explore larger question-generation models for improved output quality.
- Improve contextual understanding across multiple sentences.

## 13. Conclusion

This project demonstrates the application of Natural Language Processing and Transformer-based models to Automatic Question Generation.

By combining spaCy's linguistic analysis capabilities with a pretrained T5 question-generation model, the system converts user-provided text into questions through an automated pipeline.

The project provides a practical demonstration of how pretrained language models can be integrated into an interactive NLP application.

## 14. References

- [T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)
- [Hugging Face Model: valhalla/t5-small-qg-hl](https://huggingface.co/valhalla/t5-small-qg-hl)
- [spaCy Documentation](https://spacy.io/usage)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Streamlit Documentation](https://docs.streamlit.io/)

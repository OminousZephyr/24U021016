
import streamlit as st
from trained_question_generator import QuestionGenerator

# --------------------------------------------------
# Project information: update these fields
# --------------------------------------------------
STUDENT_NAME = "Pranay Singh"
STUDENT_ID = "24U021016"
BATCH = "CSE-AI"
INSTITUTE = "Indian Institute of Information Technology Bhopal"
DEPARTMENT = "Computer Science and Engineering"
ACADEMIC_YEAR = "2026–27"

st.set_page_config(
    page_title="Automatic Question Generation | NLP Project",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------
# Interface styling
# --------------------------------------------------
st.markdown("""
<style>
    :root {
        --navy: #183153;
        --blue: #285b91;
        --ink: #202b3c;
        --muted: #687588;
        --line: #dce2e9;
        --surface: #ffffff;
        --page: #f5f7fa;
    }

    .stApp {
        background: var(--page);
        color: var(--ink);
        font-family: Arial, Helvetica, sans-serif;
    }

    .block-container {
        max-width: 1120px;
        padding: 2rem 2.2rem 3rem;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    h1, h2, h3 {
        color: var(--navy) !important;
        letter-spacing: -0.025em;
    }

    h1 {
        font-size: 2rem !important;
        font-weight: 650 !important;
    }

    h2 {
        font-size: 1.35rem !important;
        font-weight: 650 !important;
    }

    h3 {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
    }

    p, li, label {
        color: var(--ink);
        line-height: 1.65;
    }

    .institution {
        font-size: 0.78rem;
        font-weight: 600;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.07em;
        line-height: 1.6;
    }

    .student-details {
        text-align: right;
        font-size: 0.83rem;
        color: var(--muted);
        line-height: 1.8;
    }

    .student-details strong {
        color: var(--navy);
        font-size: 0.94rem;
    }

    .project-heading {
        text-align: center;
        padding: 2.2rem 1rem 1.8rem;
        margin-top: 1.3rem;
        border-top: 1px solid var(--line);
        border-bottom: 1px solid var(--line);
        background: var(--surface);
    }

    .project-kicker {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--blue);
        margin-bottom: 0.8rem;
    }

    .project-title {
        font-size: 2.15rem;
        font-weight: 700;
        letter-spacing: -0.045em;
        line-height: 1.25;
        color: var(--navy);
    }

    .project-subtitle {
        max-width: 720px;
        margin: 0.9rem auto 0;
        font-size: 1rem;
        line-height: 1.75;
        color: var(--muted);
    }

    .section-intro {
        color: var(--muted);
        font-size: 0.92rem;
        margin-top: -0.4rem;
        margin-bottom: 1.2rem;
    }

    .metadata-label {
        font-size: 0.75rem;
        font-weight: 700;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }

    .metadata-value {
        font-size: 0.94rem;
        color: var(--navy);
        font-weight: 600;
        margin-top: 0.25rem;
    }

    .stTextArea textarea {
        background: #ffffff !important;
        color: #202b3c !important;
        border: 1px solid #cfd7e2 !important;
        border-radius: 4px !important;
        font-size: 0.94rem !important;
        line-height: 1.7 !important;
    }

    .stTextArea textarea:focus {
        border-color: var(--blue) !important;
        box-shadow: 0 0 0 1px var(--blue) !important;
    }

    .stButton > button {
        background: var(--navy);
        color: #ffffff;
        border: 1px solid var(--navy);
        border-radius: 4px;
        min-height: 42px;
        font-weight: 600;
    }

    .stButton > button:hover {
        background: #24496f;
        border-color: #24496f;
        color: #ffffff;
    }

    .result-row {
        display: flex;
        align-items: flex-start;
        gap: 14px;
        padding: 14px 16px;
        background: #ffffff;
        border: 1px solid var(--line);
        border-bottom: none;
    }

    .result-row:first-child {
        border-radius: 4px 4px 0 0;
    }

    .result-row:last-child {
        border-bottom: 1px solid var(--line);
        border-radius: 0 0 4px 4px;
    }

    .result-number {
        min-width: 28px;
        color: var(--blue);
        font-family: monospace;
        font-size: 0.88rem;
        font-weight: 700;
    }

    .result-text {
        color: var(--ink);
        font-size: 0.94rem;
        line-height: 1.65;
    }

    .info-panel {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 4px;
        padding: 18px 20px;
        margin-bottom: 12px;
    }

    .info-panel p:last-child {
        margin-bottom: 0;
    }

    [data-testid="stTabs"] button {
        font-size: 0.9rem;
        font-weight: 600;
    }

    [data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--navy);
    }

    [data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 4px;
    }

    hr {
        border-color: var(--line) !important;
    }

    .project-footer {
        text-align: center;
        color: #788496;
        font-size: 0.78rem;
        border-top: 1px solid var(--line);
        margin-top: 2.5rem;
        padding-top: 1.2rem;
        line-height: 1.8;
    }

    @media (max-width: 700px) {
        .block-container {
            padding: 1.1rem 1rem 2rem;
        }
        .project-title {
            font-size: 1.65rem;
        }
        .student-details {
            text-align: left;
            margin-top: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Academic identity
# --------------------------------------------------
identity_left, identity_right = st.columns([1.5, 1])

with identity_left:
    st.markdown(
        f'<div class="institution">{INSTITUTE}<br>{DEPARTMENT}</div>',
        unsafe_allow_html=True,
    )

with identity_right:
    st.markdown(
        f'<div class="student-details">'
        f'<strong>{STUDENT_NAME}</strong><br>'
        f'{STUDENT_ID}<br>'
        f'Batch: {BATCH}<br>'
        f'Academic Year: {ACADEMIC_YEAR}'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("""
<div class="project-heading">
    <div class="project-kicker">Natural Language Processing · Project</div>
    <div class="project-title">Automatic Question Generation</div>
    <div class="project-subtitle">
        A text-to-text generation system that produces questions from
        an input passage using a pretrained Transformer model and
        linguistic candidate-answer extraction.
    </div>
</div>
""", unsafe_allow_html=True)

st.write("")

# Project metadata
meta1, meta2, meta3 = st.columns(3)

with meta1:
    st.markdown(
        '<div class="metadata-label">Task</div>'
        '<div class="metadata-value">Question Generation</div>',
        unsafe_allow_html=True,
    )

with meta2:
    st.markdown(
        '<div class="metadata-label">Model</div>'
        '<div class="metadata-value">T5-small · Fine-tuned</div>',
        unsafe_allow_html=True,
    )

with meta3:
    st.markdown(
        '<div class="metadata-label">Framework</div>'
        '<div class="metadata-value">Python · Streamlit</div>',
        unsafe_allow_html=True,
    )

st.write("")

# --------------------------------------------------
# Main application and documentation
# --------------------------------------------------
tab_app, tab_method, tab_technical = st.tabs([
    "Question Generator",
    "Methodology",
    "Technical Details",
])

with tab_app:
    st.subheader("Generate Questions")
    st.markdown(
        '<div class="section-intro">'
        'Provide an English passage. The system identifies candidate answer '
        'spans and generates questions based on the surrounding context.'
        '</div>',
        unsafe_allow_html=True,
    )

    passage = st.text_area(
        "Input passage",
        height=230,
        placeholder=(
            "Paste the passage here...\n\n"
            "Example: Marie Curie studied radioactivity in Paris."
        ),
        label_visibility="visible",
    )

    settings_left, settings_right = st.columns([1, 1])

    with settings_left:
        question_limit = st.select_slider(
            "Maximum number of questions",
            options=[1, 2, 3, 4, 5, 6, 7, 8],
            value=5,
        )

    with settings_right:
        st.markdown(
            '<div class="metadata-label" style="margin-top: 2.2rem;">'
            'Generation mode</div>'
            '<div class="metadata-value">Answer-aware generation</div>',
            unsafe_allow_html=True,
        )

    run_generation = st.button(
        "Generate Questions",
        type="primary",
        use_container_width=True,
    )

    if run_generation:
        if not passage.strip():
            st.warning("Enter a passage before generating questions.")
        else:
            try:
                with st.spinner("Loading model and generating questions..."):

                    @st.cache_resource
                    def load_generator():
                        return QuestionGenerator()

                    generator = load_generator()
                    questions = generator.generate_questions(
                        passage,
                        max_questions=question_limit,
                    )

                st.write("")
                st.subheader("Generated Output")

                if questions:
                    st.caption(
                        f"{len(questions)} question(s) generated. "
                        "Review the output for relevance and accuracy."
                    )

                    output_html = ""
                    for index, question in enumerate(questions, start=1):
                        safe_question = (
                            question.replace("&", "&amp;")
                            .replace("<", "&lt;")
                            .replace(">", "&gt;")
                        )
                        output_html += (
                            '<div class="result-row">'
                            f'<span class="result-number">{index:02d}</span>'
                            f'<span class="result-text">{safe_question}</span>'
                            '</div>'
                        )

                    st.markdown(output_html, unsafe_allow_html=True)

                    st.download_button(
                        "Download questions (.txt)",
                        data="\n".join(
                            f"{i}. {q}" for i, q in enumerate(questions, 1)
                        ),
                        file_name="generated_questions.txt",
                        mime="text/plain",
                    )
                else:
                    st.info(
                        "No questions were generated. Try a longer or "
                        "more descriptive passage."
                    )

            except Exception as error:
                st.error(f"Question generation failed: {error}")

    st.markdown("")
    st.caption(
        "Generated questions are model outputs and may require manual review."
    )

with tab_method:
    st.subheader("Project Overview")

    st.markdown("""
    **Objective**

    To develop a Python application that automatically generates
    natural-language questions from a user-provided passage.

    **Method**

    The system uses a pretrained, task-fine-tuned T5 model. spaCy
    processes the input text and supplies sentence boundaries, named
    entities, and noun chunks. These linguistic features are used to
    identify candidate answer spans.

    **Generation process**

    1. Accept an English passage from the user.
    2. Segment the passage into sentences.
    3. Extract candidate answers using named entities and noun chunks.
    4. Mark each candidate answer using the model's highlight-token format.
    5. Pass the formatted text to T5 for question generation.
    6. Clean the generated text and remove duplicate questions.
    7. Return a limited list of questions to the interface.

    **Model approach**

    The project uses `valhalla/t5-small-qg-hl`, a pretrained T5 model
    fine-tuned for question generation with highlighted answer spans.
    The model is used for inference; this project does not train or
    fine-tune its parameters.
    """)

    st.subheader("System Pipeline")
    st.markdown("""
    **Input passage → Sentence analysis → Candidate answer extraction
    → Answer highlighting → T5 generation → Output cleaning → Questions**
    """)

with tab_technical:
    st.subheader("Implementation Details")

    st.markdown("""
    | Component | Role in the project |
    |---|---|
    | Python | Main programming language |
    | Transformers | Loads the T5 tokenizer and sequence-to-sequence model |
    | PyTorch | Executes neural-network inference |
    | spaCy | Sentence segmentation, named entities, and noun chunks |
    | SentencePiece | Supports the model's subword tokenisation |
    | Streamlit | User interface and interactive controls |
    | `re` | Text cleaning and string matching |
    """)

    st.subheader("Core Concepts")

    with st.expander("Transformer and T5"):
        st.write(
            "T5 stands for Text-to-Text Transfer Transformer. It treats "
            "the task as converting an input text sequence into an output "
            "text sequence. Its learned parameters guide question generation."
        )

    with st.expander("Answer-aware question generation"):
        st.write(
            "Candidate answers are marked with special highlight tokens "
            "before inference. The model uses the marked span and its context "
            "to generate a question about that candidate answer."
        )

    with st.expander("Inference versus training"):
        st.write(
            "Inference means using an existing trained model to produce "
            "outputs. This application performs inference with a pretrained, "
            "fine-tuned model; it does not train the model from scratch."
        )

    st.subheader("Limitations")

    st.markdown("""
    - Generated questions may be irrelevant, repetitive, or grammatically awkward.
    - Candidate answer extraction depends on the linguistic analysis of the input.
    - The model may generate questions that are not fully supported by the passage.
    - The current application does not calculate a formal question-quality score.
    - Output quality should be assessed through human review or a labelled evaluation dataset.
    """)

st.markdown("""
<div class="project-footer">
    Automatic Question Generation · Natural Language Processing<br>
    Academic project demonstration
</div>
""", unsafe_allow_html=True)

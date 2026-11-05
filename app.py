import logging
from concurrent.futures import ThreadPoolExecutor

import streamlit as st

from modules.pdf_reader import extract_text, PDFExtractionError
from modules.simplifier import simplify_content_stream
from modules.takeaways import generate_takeaways
from modules.quiz import generate_quiz

logger = logging.getLogger("edusimplify.app")


st.set_page_config(
    page_title="EduSimplify",
    page_icon="📚",
    layout="wide"
)

# Streamlit re-runs this whole script on EVERY interaction (a radio click, the
# Submit button...). A button's value is only True for the single run right
# after it is clicked, so results must live in session_state, not inside
# `if st.button(...)`, or they disappear on the next click.
if "result" not in st.session_state:
    st.session_state.result = None
if "result_id" not in st.session_state:
    st.session_state.result_id = 0
if "submitted" not in st.session_state:
    st.session_state.submitted = False


def generate_extras(text, language):
    """Takeaways and quiz are independent, so run them at the same time."""
    names = ("takeaways", "quiz")
    errors = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = (
            pool.submit(generate_takeaways, text, language),
            pool.submit(generate_quiz, text, language),
        )
        results = []
        for name, future in zip(names, futures):
            try:
                results.append(future.result())
            except Exception as exc:
                logger.exception("Generating %s failed", name)
                errors[name] = exc
                results.append([])
    if errors:
        for name, exc in errors.items():
            st.warning(f"⚠️ Could not generate {name}: {exc}")
    return results


def render_results(result):
    with st.expander("View Extracted Text"):
        st.write(result["content"][:3000])

    st.header("Simplified Explanation")
    st.write(result["simplified"])
    st.success("Content simplified successfully!")

    st.divider()

    # ── Key Takeaways ──────────────────────────────────────────────────────
    st.header("🗝️ Key Takeaways")
    if result["takeaways"]:
        for point in result["takeaways"]:
            st.markdown(f"- {point}")
    else:
        st.info("Could not extract takeaways for this content.")

    st.divider()

    # ── Quiz ───────────────────────────────────────────────────────────────
    st.header("📝 Test Your Understanding")
    questions = result["questions"]
    if not questions:
        st.info("Could not generate quiz questions for this content.")
        return

    rid = st.session_state.result_id      # keys change per result, so old answers never carry over
    user_answers = {}
    for i, q in enumerate(questions):
        st.markdown(f"**Q{i + 1}. {q['question']}**")
        opts = q["options"]
        user_answers[i] = st.radio(
            label=f"q{i}",
            options=list(opts.keys()),
            format_func=lambda k, o=opts: f"{k}. {o[k]}",
            index=None,
            label_visibility="collapsed",
            key=f"q_{rid}_{i}",
        )
        st.write("")

    if st.button("Submit Answers"):
        st.session_state.submitted = True

    if st.session_state.submitted:
        score = 0
        for i, q in enumerate(questions):
            correct = q["answer"]
            chosen = user_answers.get(i)
            if chosen is None:
                st.warning(f"Q{i + 1}: Not answered.")
            elif chosen == correct:
                st.success(
                    f"Q{i + 1}: ✅ Correct! ({correct}. {q['options'][correct]})"
                )
                score += 1
            else:
                st.error(
                    f"Q{i + 1}: ❌ Wrong. "
                    f"Your answer: {chosen}. {q['options'][chosen]} | "
                    f"Correct: {correct}. {q['options'][correct]}"
                )
        st.info(f"Score: {score} / {len(questions)}")


st.title("📚 EduSimplify")
st.subheader("AI-Powered Course Content Simplification Agent")

st.write("Understand complex academic content in simple language.")

st.divider()

st.header("Upload Your Study Material")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

text_input = st.text_area(
    "Or paste your academic content here",
    height=150
)

col1, col2 = st.columns(2)

with col1:
    level = st.selectbox(
        "Select Difficulty Level",
        ["Beginner", "Intermediate", "Advanced"]
    )

with col2:
    language = st.selectbox(
        "Select Language",
        ["English", "Hindi"]
    )


if st.button("Simplify Content", type="primary"):

    st.session_state.result = None        # drop the previous result
    content = ""

    try:
        if uploaded_file:
            content = extract_text(uploaded_file)

        elif text_input.strip():
            content = text_input.strip()

        if not content:
            st.warning(
                "No readable text found. Please upload a text-based PDF or enter some text."
            )

        else:
            # Live view while generating; replaced by the saved result below
            live = st.empty()
            with live.container():
                with st.expander("View Extracted Text"):
                    st.write(content[:3000])

                st.header("Simplified Explanation")
                streamed = st.write_stream(
                    simplify_content_stream(content, level, language)
                )
                simplified = (
                    streamed if isinstance(streamed, str)
                    else "".join(str(piece) for piece in streamed)
                )

                with st.spinner("Preparing key takeaways and quiz..."):
                    takeaways, questions = generate_extras(simplified or content, language)

            live.empty()

            st.session_state.result_id += 1
            st.session_state.submitted = False
            st.session_state.result = {
                "content": content,
                "simplified": simplified,
                "takeaways": takeaways,
                "questions": questions,
            }

    except PDFExtractionError as e:
        # Expected problems: scanned PDF, password-protected, empty file, etc.
        st.warning(str(e))

    except Exception as e:
        st.error("Something went wrong!")
        st.exception(e)


# Shown on every run, so answering the quiz doesn't wipe the page
if st.session_state.result:
    render_results(st.session_state.result)
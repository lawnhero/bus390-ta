"""Shared UI components for the BUS 390 Python Virtual TA.

Markdown helpers, route provenance metadata, status/badge/feedback renderers,
starter prompts, and quick-action chips. All course-facing strings live here.
Ported from the BUS 390 SQL toolkit's UI overhaul.
"""

import streamlit as st

TA_AVATAR = ":material/school:"
TA_NAME = "Peyton"

INSTRUCTOR_NAME = "Dr. Wenjun Gu"
INSTRUCTOR_EMAIL = "wenjun.gu@emory.edu"
COURSE_LABEL = "BUS 390 · Python Toolkit"

# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------

def escape_md_dollars(text: str) -> str:
    """Escape $ so Streamlit markdown does not treat it as LaTeX."""
    if not text:
        return text
    placeholder = "\ue000"  # private-use char that never occurs in model output
    protected = text.replace("\\$", placeholder)
    return protected.replace("$", "\\$").replace(placeholder, "\\$")


def md(text, container=None):
    (container or st).markdown(escape_md_dollars(text))


def write_stream_md(stream, container=None):
    """Stream LLM tokens into one placeholder, escaping $ as we go."""
    placeholder = (container or st).empty()
    chunks = []
    for chunk in stream:
        text = chunk if isinstance(chunk, str) else getattr(chunk, "content", None) or str(chunk)
        chunks.append(text)
        placeholder.markdown(escape_md_dollars("".join(chunks)))
    return "".join(chunks)


# ---------------------------------------------------------------------------
# Route provenance: three tenses of the same fact, from one table
# ---------------------------------------------------------------------------
# Color semantics: blue = grounded in course materials, violet = model's
# Python knowledge, green = generated practice, gray = no lookup,
# orange = fallback. Keys are the tool names the router emits.

ROUTE_META = {
    "course_information": {
        "working": "Looking through the course materials",
        "done": "Checked the course materials",
        "badge": "Course materials",
        "icon": ":material/menu_book:",
        "color": "blue",
        "help": "Answered from the syllabus and course content for BUS 390.",
    },
    "explain_concept": {
        "working": "Writing a plain-English explanation",
        "done": "Explained the concept",
        "badge": "Python explanation",
        "icon": ":material/lightbulb:",
        "color": "violet",
        "help": "A general Python explanation from the tutor's knowledge, with a business example.",
    },
    "generate_exercise": {
        "working": "Building a practice question for you",
        "done": "Built a practice question",
        "badge": "Practice question",
        "icon": ":material/quiz:",
        "color": "green",
        "help": "A practice question generated for you. Try it before asking for the answer.",
    },
    "debug_code": {
        "working": "Reading your code and the error",
        "done": "Reviewed your code",
        "badge": "Debugging help",
        "icon": ":material/build:",
        "color": "violet",
        "help": "Hints toward the fix, based on the code and error you shared — "
                "you make the change yourself.",
    },
    "general_chat": {
        "working": "Thinking about your message",
        "done": "Answered directly",
        "badge": "General chat",
        "icon": ":material/chat:",
        "color": "gray",
        "help": "A conversational reply — no course lookup was needed.",
    },
    "fallback": {
        "working": "Trying a general answer",
        "done": "Answered without a course lookup",
        "badge": "Not from course materials",
        "icon": ":material/warning:",
        "color": "orange",
        "help": "Something went wrong with the usual route, so this is a general answer. "
                "Double-check it against the course materials.",
    },
}
DEFAULT_ROUTE = "general_chat"


def route_meta(label):
    return ROUTE_META.get(label or "", ROUTE_META[DEFAULT_ROUTE])


def working_label(route_label):
    return route_meta(route_label)["working"]


def completion_label(route_label, *, seconds=0.0):
    parts = [route_meta(route_label)["done"]]
    if seconds:
        parts.append(f"{seconds:.1f}s")
    return " · ".join(parts)


# ---------------------------------------------------------------------------
# Per-message renderers (replayed from message_meta on every rerun)
# ---------------------------------------------------------------------------

def render_progress(record):
    if not record:
        return
    st.status(
        record.get("label") or "Answer complete",
        state=record.get("state") or "complete",
        type="compact",
        expanded=False,
    )


def render_provenance(route_label):
    meta = route_meta(route_label)
    st.badge(meta["badge"], icon=meta["icon"], color=meta["color"], help=meta["help"])


def render_answer_footer(route_label, *, trailing=None):
    """One horizontal line under the answer: badge · feedback thumbs."""
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        render_provenance(route_label)
        if trailing is not None:
            trailing()


# ---------------------------------------------------------------------------
# Feedback thumbs — callback, no extra rerun
# ---------------------------------------------------------------------------

def _record_feedback(store_feedback, interaction_id, key):
    value = st.session_state.get(key)
    if value is None:
        return
    try:
        store_feedback(interaction_id=interaction_id,
                       helpful="Helpful" if value == 1 else "Not helpful")
    except Exception:
        pass  # never let feedback logging break a student's session
    st.session_state.setdefault("feedback_submitted_ids", []).append(interaction_id)
    st.toast("Thanks — that helps improve the tutor.", icon=":material/favorite:")


def feedback_widget(interaction_id, store_feedback):
    """Return a callable that draws the thumbs, for use as a footer `trailing`."""
    if not interaction_id:
        return None

    def draw():
        submitted = st.session_state.get("feedback_submitted_ids", [])
        if interaction_id in submitted:
            st.caption("Rating recorded.")
            return
        key = f"feedback_{interaction_id}"
        st.feedback("thumbs", key=key, on_change=_record_feedback,
                    args=(store_feedback, interaction_id, key))

    return draw


# ---------------------------------------------------------------------------
# Starter prompts, quick actions, curriculum
# ---------------------------------------------------------------------------
# Every starter must be answerable by the current chains — a first click that
# flops is worse than none. Pitched at M1–M3, where a new student starts, and
# explain-shaped (concept questions are answered directly under the hint
# policy, so starters never demonstrate a withheld answer on first click).
# "Rule of Harmony" is course vocabulary — defined in the curriculum block
# below, so the TA can connect the name to its substance.

STARTER_PROMPTS = [
    "What is Python and why does it matter in business?",
    "What is the Rule of Harmony?",
    "Why does input() always give me text, not a number?",
    "Give me an easy practice question on the print function",
]

# The course modules, from the Canvas module/quiz map (2026-08-31): each entry
# carries what the module TEACHES (instruction scope) and what its quiz
# actually ASKS (quiz scope) — quizzes test syntax and vocabulary, while the
# applications are assessed only through projects. One source of truth that
# drives the topic pills, the sidebar outline, and the curriculum block in
# the LLM prompts, so they can never disagree. M0 (welcome survey) is
# logistics-only and deliberately absent.
MODULES = [
    {"code": "M1", "title": "Introduction — Hello World",
     "covers": "why Python, Colab, print(), and the Rule of Harmony",
     "teach": "what a high-level language is, why Python for business and Python vs Excel, "
              "Google Colab basics, print('Hello World'), and the Rule of Harmony — every "
              "opening parenthesis or quote must be closed and matched, or Python raises a "
              "SyntaxError",
     "quiz": "print() usage, valid syntax, parentheses inside strings, unclosed quotes, "
             "print(42) vs print(\"42\"), what 'high-level' means"},
    {"code": "M2", "title": "Python as calculator",
     "covers": "arithmetic, comments, and variables",
     "teach": "arithmetic operators and PEMDAS order, # comments, variables and assignment, "
              "reassignment, informative variable names",
     "quiz": "operator symbols, *, comments (their symbol and purpose, ignored at runtime), "
             "assignment, reassignment like x = x + 2, informative names like "
             "revenue_quarter, why print(2 + ) is a SyntaxError"},
    {"code": "M3", "title": "Data types",
     "covers": "input(), str/int/float, and casting",
     "teach": "input() and why it always returns a string, str/int/float, type(), casting "
              "between types, fixing the TypeError from mixing strings and numbers",
     "quiz": "what input() does and that it returns a str, int()/float()/type(), matching "
             "values to their types, why \"Hello\" + 5 is a TypeError, casting input "
             "before arithmetic"},
    {"code": "M4", "title": "Functions",
     "covers": "def, parameters, return, and scope",
     "teach": "defining functions with def, parameters vs arguments, RETURN VALUES — the "
              "load-bearing concept: no quiz item asks about return, but Module 5 and both "
              "projects depend on it, so always teach return at full weight — plus "
              "indentation, consistent naming, and local vs global scope",
     "quiz": "the def keyword and header, indentation, parameters in the header, calling "
             "with an argument, 'parameter' vocabulary, local vs global scope, and what "
             "happens when a required argument is missing — teach that this raises a "
             "TypeError (the final quiz keys it correctly; if the module quiz marked a "
             "student differently, acknowledge the key is being fixed)"},
    {"code": "M5", "title": "Create your own module",
     "covers": "writing .py modules and importing them in Colab",
     "teach": "what a module is vs a function, writing simple_calculator.py, and the "
              "four-step Colab session-storage workflow — session files vanish when Colab "
              "disconnects, so the .py must be re-created or re-uploaded each session; this "
              "is where most students actually get stuck — then import and aliases. Due the "
              "same night as M4 and built directly on M4's return values",
     "quiz": "what a module is, the .py extension, import, aliases like "
             "import simple_calculator as sc, dotted calls like sc.divide() and "
             "mo.square(5)"},
    {"code": "M6", "title": "Use other's modules",
     "covers": "installing libraries and making QR codes",
     "teach": "what libraries are, %pip install followed by import, qrcode.make() for a "
              "URL, the advanced QRCode(version, box_size, border) form with "
              "fill_color/back_color, and embedding a logo with PIL",
     "quiz": "the library concept, pip install, the three import styles (whole library, "
             "alias, from qrcode import make), making a QR code for a URL, what version=1 "
             "means for size, fill_color/back_color, business uses"},
    {"code": "M7", "title": "Python on your computer",
     "covers": "local install, VS Code, and Jupyter — no quiz, project only",
     "teach": "installing Python (check 'Add to PATH'), VS Code with the Python and "
              "Jupyter extensions, ipykernel, recreating simple_calculator locally; "
              "troubleshooting: PATH problems, missing extensions, the ipykernel prompt, "
              "and a 0 KB file meaning it never saved",
     "quiz": None},
    {"code": "Final", "title": "Course quiz & final project",
     "covers": "everything from M1–M6",
     "teach": "the final quiz re-asks the module quizzes — the best preparation is redoing "
              "the M1–M6 quizzes (M5 appears verbatim); nothing from M7 is on it. For the "
              "final project: submit the .ipynb file, use %pip inside notebook cells vs "
              "pip in a terminal, and a 0 KB file means it never saved",
     "quiz": "a shuffle of the M1–M6 quiz items, no new topics"},
]

# Topic pills (12): one per module plus cross-cutting pills for the friction
# points the quiz map surfaced — errors, the running examples that chain
# M2→M7, Colab session storage, %pip vs pip, and final prep.
TOPICS = [f"{m['code']} · {m['title']}" for m in MODULES[:-1]] + [
    "Error doctor",
    "The running examples",
    "Colab & session storage",
    "%pip vs pip",
    "Final prep",
]

# Injected into the explain/exercise system prompts so the TA teaches to the
# module ladder instead of wandering beyond course scope. Carries both layers:
# chains explain at TEACH scope and generate practice at QUIZ scope.
CURRICULUM_PROMPT = (
    "The course is a seven-module beginner Python toolkit for business "
    "students with no prior coding experience, run in Google Colab, ending in "
    "a final course quiz and final project. Module quizzes test syntax and "
    "vocabulary; the applications are assessed only through projects — so "
    "EXPLAIN at the 'teaches' scope, and write PRACTICE questions at the "
    "'quiz asks' scope unless the student asks for project-style practice.\n"
    "Three running examples chain through the course: bill_per_person (M2) → "
    "dog_years (M3–M4) → simple_calculator (M5, recreated locally in M7). Use "
    "them when showing how concepts connect; practice questions should vary "
    "the scenario instead.\n"
    "The modules, in learning order:\n"
    + "\n".join(
        f"- {m['code']} {m['title']} — teaches: {m['teach']}."
        + (f" Quiz asks: {m['quiz']}." if m.get("quiz") else " No quiz; assessed by project only.")
        for m in MODULES
    )
)

# Intent chips: "query" sends value directly; "intent" starts a clarifying turn.
QUICK_ACTIONS = [
    {"label": "Explain a concept", "icon": ":material/menu_book:", "kind": "intent",
     "value": "explain", "needs": "topic",
     "clarify": "Happy to explain! Pick a topic below, or type the concept you're curious about."},
    {"label": "Practice question", "icon": ":material/quiz:", "kind": "intent",
     "value": "practice", "needs": "topic",
     "clarify": "Let's practice. Pick a topic below, or type the one you want to drill."},
    {"label": "Fix my error", "icon": ":material/build:", "kind": "intent",
     "value": "debug", "needs": "attempt",
     "clarify": "Paste your Python code and the error message into the chat, and I'll "
                "point you toward the fix — you'll make the change yourself."},
]

FOLLOW_UPS = [
    {"label": "Explain it more simply", "icon": ":material/lightbulb:", "kind": "query",
     "value": "Can you explain that again more simply, for a complete beginner?"},
    {"label": "Give me a practice question", "icon": ":material/quiz:", "kind": "query",
     "value": "Give me a practice question on what we just discussed."},
    {"label": "Show a business example", "icon": ":material/storefront:", "kind": "query",
     "value": "Can you show a short business example of what we just discussed?"},
]

INTENT_QUERY_TEMPLATES = {
    "explain": "Explain {topic} for a complete beginner, with a short business example.",
    "practice": "Give me a beginner practice question on {topic}.",
}

# The cross-cutting pills don't read naturally through the generic templates,
# so they carry their own composed queries per intent. Module pills and typed
# free-text topics fall through to the templates.
TOPIC_QUERY_OVERRIDES = {
    "Error doctor": {
        "explain": "Teach me how to read and fix the common Python errors in this course: "
                   "SyntaxError, TypeError, IndentationError, and NameError.",
        "practice": "Give me a practice question where I have to spot and fix a common "
                    "Python error (SyntaxError, TypeError, IndentationError, or NameError).",
    },
    "The running examples": {
        "explain": "Walk me through how the course's running examples connect from module "
                   "to module: bill_per_person, dog_years, and simple_calculator.",
        "practice": "Give me a practice question that extends one of the course's running "
                    "examples (bill_per_person, dog_years, or simple_calculator).",
    },
    "Colab & session storage": {
        "explain": "Explain how Colab session storage works, why my .py files disappear "
                   "between sessions, and the workflow for using my own module in Colab.",
        "practice": "Give me a practice question on the Colab session-storage workflow "
                    "for using my own module.",
    },
    "%pip vs pip": {
        "explain": "Explain the difference between %pip in a notebook cell and pip in a "
                   "terminal, and when to use each.",
        "practice": "Give me a practice question on installing a library with %pip and "
                    "importing it.",
    },
    "Final prep": {
        "explain": "How should I prepare for the final course assessment, and what does "
                   "it cover?",
        "practice": "Give me a mixed practice question like the final quiz, drawing on "
                    "topics from M1–M6.",
    },
}


def intent_query(intent, topic):
    """The query a topic pill (or typed topic) sends for the given intent."""
    override = TOPIC_QUERY_OVERRIDES.get(topic, {}).get(intent)
    return override or INTENT_QUERY_TEMPLATES[intent].format(topic=topic)


def render_action_row(actions, *, key_prefix):
    """Horizontal row of chip buttons; returns the clicked action or None."""
    clicked = None
    with st.container(horizontal=True, horizontal_alignment="distribute"):
        for idx, action in enumerate(actions):
            if st.button(action["label"], icon=action.get("icon"),
                         key=f"{key_prefix}_{idx}", width="stretch"):
                clicked = action
    return clicked


# ---------------------------------------------------------------------------
# Help dialog
# ---------------------------------------------------------------------------

@st.dialog("How this tutor works", width="large")
def show_help_dialog():
    st.subheader("Hints, not answers")
    st.markdown(
        "- Concept questions (\"what does a for loop do?\") get direct explanations "
        "with examples\n"
        "- Your own code, debugging, and anything from a quiz get **hints** — "
        "I won't hand you the answer, even if you ask twice\n"
        "- Course logistics (deadlines, contacts, policies) get straight answers"
    )
    st.subheader("Where answers come from")
    st.markdown(
        "- :blue-badge[Course materials] — answered from the BUS 390 syllabus and course content\n"
        "- :violet-badge[Python explanation] · :violet-badge[Debugging help] — the tutor's general "
        "Python knowledge, not course-specific\n"
        "- :green-badge[Practice question] — generated for you to try\n"
        "- :gray-badge[General chat] — conversation, no lookup needed\n"
        "- :orange-badge[Not from course materials] — a fallback answer; double-check it"
    )
    st.subheader("Getting better results")
    st.markdown(
        "- Tell me which module you're on (M1–M7) so I can pitch answers at the right level\n"
        "- Paste your code, not a description of it\n"
        "- Paste the exact error message when something breaks\n"
        "- Say what you expected the code to do, and what it did instead"
    )
    st.subheader("Good to know")
    st.markdown(
        "- This tutor can make mistakes — verify anything that affects your grade\n"
        "- Don't include personal information in your questions\n"
        "- Questions are logged (anonymously) to improve the course\n"
        f"- Stuck? Email {INSTRUCTOR_NAME}: [{INSTRUCTOR_EMAIL}](mailto:{INSTRUCTOR_EMAIL})"
    )

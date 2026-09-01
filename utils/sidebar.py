import streamlit as st
from datetime import datetime
from langchain_core.messages import HumanMessage

from utils.ui import (
    MODULES,
    COURSE_LABEL,
    INSTRUCTOR_NAME,
    INSTRUCTOR_EMAIL,
    show_help_dialog,
)

# Every conversation-scoped key, in one place — ad-hoc clearing leaves stale
# state behind. Keys describing the student (none yet) would deliberately survive.
CONVERSATION_KEYS = (
    "chat_history",
    "message_meta",
    "pending_intent",
    "feedback_submitted_ids",
    "starter_prompt_pills",
    "topic_pills",
)


def clear_chat_history():
    for key in CONVERSATION_KEYS:
        st.session_state.pop(key, None)


def save_chat_history():
    """Serialize the conversation to plain text for download."""
    lines = [f"{COURSE_LABEL} — Virtual TA chat, saved {datetime.now():%Y-%m-%d %H:%M}", ""]
    for message in st.session_state.get("chat_history", []):
        speaker = "You" if isinstance(message, HumanMessage) else "Virtual TA"
        lines.append(f"{speaker}:\n{message.content}\n")
    return "\n".join(lines)


def sidebar():
    with st.sidebar:
        # 1. Primary action
        if st.button("New chat", icon=":material/add_comment:", type="primary", width="stretch"):
            clear_chat_history()

        # 2. Two quiet actions on one row
        with st.container(horizontal=True):
            if st.session_state.get("chat_history"):
                st.download_button(
                    "Save chat",
                    data=save_chat_history(),
                    file_name=f"python_ta_chat_{datetime.now():%Y%m%d_%H%M}.txt",
                    mime="text/plain",
                    icon=":material/download:",
                    width="stretch",
                )
            else:
                st.button("Save chat", icon=":material/download:", disabled=True,
                          width="stretch", help="No conversation to save yet")
            if st.button("How this works", icon=":material/help:", width="stretch"):
                show_help_dialog()

        # 3. What the tutor can help with — driven by the same MODULES list as
        #    the topic pills, so the two can never disagree.
        st.space("small")
        with st.expander("The course modules", icon=":material/checklist:"):
            st.markdown("\n".join(
                f"- **{m['code']} · {m['title']}** — {m['covers']}" for m in MODULES
            ))
            st.caption("Plus course questions: syllabus, assignments, and policies.")

        # 4. Footer
        st.space("medium")
        st.caption(COURSE_LABEL)
        st.caption("This tutor is a work in progress and can make mistakes.")
        st.caption(f"Made by {INSTRUCTOR_NAME} · [{INSTRUCTOR_EMAIL}](mailto:{INSTRUCTOR_EMAIL})")

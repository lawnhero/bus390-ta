import time
import traceback
import uuid

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

import utils.chains_lcel as chains
import utils.llm_models as llms
from utils.sidebar import sidebar
from utils.utils import load_db, query_db_connection, process_and_store_query
from utils.tools import create_tool_chain
from utils.ui import (
    TA_AVATAR,
    TA_NAME,
    DEFAULT_ROUTE,
    STARTER_PROMPTS,
    TOPICS,
    QUICK_ACTIONS,
    FOLLOW_UPS,
    intent_query,
    md,
    write_stream_md,
    working_label,
    completion_label,
    render_progress,
    render_answer_footer,
    render_action_row,
    feedback_widget,
)

MEMORY_WINDOW = 6  # messages of history passed to the chains

GREETING = (
    f"Hi! I'm {TA_NAME}, your BUS 390 Python Virtual TA. I can explain anything "
    "from the seven modules (M1–M7), build practice questions, and help you fix "
    "errors — no question is too basic. One rule: on quiz questions and your own "
    "code I give hints instead of answers, because that's how it sticks. Course "
    "logistics questions get straight answers."
)


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

def initialize_resources():
    retriever = load_db().as_retriever()
    mongo_db = query_db_connection()
    collection = mongo_db["Python_toolkit"]
    return retriever, collection


@st.cache_resource
def initialize_chains(_retriever):
    """Build all chains once per session; underscore skips hashing the retriever.

    Keys are the router's tool names, so a tool call maps straight to a chain
    and to its ROUTE_META entry.
    """
    try:
        chains_dict = {
            "course_information": chains.rag_chain(llms.claude_haiku, _retriever),
            "generate_exercise": chains.exercise_chain(llms.claude_sonnet),
            "general_chat": chains.chat_chain(llms.openai_gpt4o_mini),
            "explain_concept": chains.explain_chain(llms.openai_gpt4o),
            "debug_code": chains.debug_chain(llms.claude_haiku),
        }
        tool_chain = create_tool_chain(llms.openai_gpt4o_mini, chains_dict)
        return tool_chain, chains_dict
    except Exception as e:
        st.error(f"Failed to initialize chains: {str(e)}")
        raise e


# ---------------------------------------------------------------------------
# Per-message metadata (what the transcript needs to replay a finished turn)
# ---------------------------------------------------------------------------

def _set_meta(index, **fields):
    st.session_state.message_meta.setdefault(index, {}).update(fields)


def _get_meta(index):
    return st.session_state.message_meta.get(index, {})


def _cancel_pending():
    st.session_state.pending_intent = None
    st.session_state.pop("topic_pills", None)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="BUS 390 Python Virtual TA",
        page_icon=":material/school:",
        layout="wide",
    )
    st.title("BUS 390 Python Virtual TA")

    retriever, collection = initialize_resources()
    agent, chain_dict = initialize_chains(retriever)

    def log_query(query):
        try:
            process_and_store_query(collection, kind="query", query=query)
        except Exception:
            pass  # logging must never break a student's session

    def store_feedback(**fields):
        process_and_store_query(collection, kind="feedback", **fields)

    # Sidebar first: its "New chat" button pops conversation keys, so the
    # defaults below must be applied after it runs.
    sidebar()

    # Conversation-scoped state
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("message_meta", {})
    st.session_state.setdefault("pending_intent", None)
    st.session_state.setdefault("feedback_submitted_ids", [])

    pending = st.session_state.pending_intent
    conversation_started = bool(st.session_state.chat_history)

    # ------------------------------------------------------------------
    # Transcript
    # ------------------------------------------------------------------

    def _render_ai_message(idx, content, is_last):
        with st.chat_message("AI", avatar=TA_AVATAR):
            meta = _get_meta(idx)
            render_progress(meta.get("progress"))
            md(content)
            if meta.get("route"):
                trailing = (feedback_widget(meta.get("interaction_id"), store_feedback)
                            if is_last else None)
                render_answer_footer(meta["route"], trailing=trailing)

    last_index = len(st.session_state.chat_history) - 1
    for idx, message in enumerate(st.session_state.chat_history):
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                md(message.content)
        elif isinstance(message, AIMessage):
            _render_ai_message(idx, message.content, is_last=(idx == last_index))

    # ------------------------------------------------------------------
    # Empty state: greeting + starter prompts
    # ------------------------------------------------------------------
    starter_choice = None
    if not conversation_started and not pending:
        with st.chat_message("AI", avatar=TA_AVATAR):
            md(GREETING)
        st.caption("Try asking")
        starter_choice = st.pills(
            "Example questions",
            options=STARTER_PROMPTS,
            selection_mode="single",
            key="starter_prompt_pills",
            label_visibility="collapsed",
        )

    # ------------------------------------------------------------------
    # Clarifying turn: topic pills + explicit way out
    # ------------------------------------------------------------------
    topic_choice = None
    if pending:
        if pending.get("needs") == "topic":
            st.caption("Suggested topics")
            topic_choice = st.pills(
                "Topics",
                options=TOPICS,
                selection_mode="single",
                key="topic_pills",
                label_visibility="collapsed",
            )
        with st.container(horizontal=True, horizontal_alignment="left"):
            if st.button("Never mind", icon=":material/close:", key="cancel_pending",
                         type="tertiary"):
                _cancel_pending()
                st.rerun()

    # ------------------------------------------------------------------
    # Pinned composer: chips + chat input
    # ------------------------------------------------------------------
    if pending and pending.get("needs") == "attempt":
        chat_placeholder = "Paste your Python code and the error message here..."
    elif pending:
        chat_placeholder = "Pick a topic above, or type one here..."
    else:
        chat_placeholder = "Ask a Python or course question — no question is too basic..."

    selected_action = None
    with st.bottom:
        if not pending:
            st.caption("Quick actions" if not conversation_started else "What next?")
            actions = QUICK_ACTIONS if not conversation_started else FOLLOW_UPS
            selected_action = render_action_row(actions, key_prefix="action_chip")

        typed_query = st.chat_input(chat_placeholder, key="user_query", submit_mode="stop")
        st.caption("Do not include personal information. This tutor can make mistakes.")

    # ------------------------------------------------------------------
    # Turn execution
    # ------------------------------------------------------------------

    def run_turn(user_query):
        with st.chat_message("Human"):
            md(user_query)
        log_query(user_query)

        context_msgs = st.session_state.chat_history[-MEMORY_WINDOW:]
        conversation_context = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in context_msgs
        )

        with st.chat_message("AI", avatar=TA_AVATAR):
            status = st.status("Reading your question...", type="compact")
            started = time.perf_counter()
            route = DEFAULT_ROUTE
            ai_text = None
            progress = None

            try:
                tool_call = agent.invoke(
                    f"""
                    Your only task is to decide which tool to call based on the user query delimited with <query> tags and chat history, and generate the appropriate arguments for the tool call.

                    Previous conversation: {conversation_context} \n
                    Query: <query>{user_query}</query>\n
                    Generate the tool call with appropriate arguments. Do not generate direct response. Enrich the query for the tool call when appropriate, but don't fundamentally change it. Limit query to no more than 25 tokens.
                    """
                )
                if tool_call.tool_calls and tool_call.tool_calls[0]["name"] in chain_dict:
                    route = tool_call.tool_calls[0]["name"]
                    args = tool_call.tool_calls[0]["args"]
                else:
                    route, args = DEFAULT_ROUTE, {"query": user_query}
                status.update(label=f"{working_label(route)}...")
                stream = chain_dict[route].stream(
                    input={"chat_history": context_msgs, **args})
                ai_text = write_stream_md(stream)
            except Exception:
                # Degrade, don't abort: answer from the chat chain, marked as
                # a fallback so it never looks grounded.
                traceback.print_exc()
                route = "fallback"
                status.update(label=f"{working_label(route)}...")
                try:
                    stream = chain_dict[DEFAULT_ROUTE].stream(
                        input={"chat_history": context_msgs, "query": user_query})
                    ai_text = write_stream_md(stream)
                except Exception:
                    traceback.print_exc()
                    ai_text = ("Sorry — I hit a snag answering that. Please try "
                               "again, or rephrase your question.")
                    md(ai_text)
                    progress = {"label": "Something went wrong", "state": "error"}
                    status.update(label=progress["label"], state="error")

            if progress is None:
                label = completion_label(route, seconds=time.perf_counter() - started)
                progress = {"label": label, "state": "complete"}
                status.update(label=label, state="complete")

        st.session_state.chat_history.append(HumanMessage(user_query))
        st.session_state.chat_history.append(AIMessage(ai_text))
        _set_meta(
            len(st.session_state.chat_history) - 1,
            route=route,
            interaction_id=uuid.uuid4().hex,
            progress=progress,
        )
        st.rerun()

    def start_intent(action):
        """Append a synthetic clarify exchange and enter the pending state."""
        st.session_state.chat_history.append(HumanMessage(action["label"]))
        st.session_state.chat_history.append(AIMessage(action["clarify"]))
        st.session_state.pending_intent = {
            "intent": action["value"],
            "needs": action["needs"],
        }
        st.rerun()

    # ------------------------------------------------------------------
    # Resolve this rerun's input (at most one fires per rerun)
    # ------------------------------------------------------------------
    if topic_choice:
        intent = pending["intent"]
        _cancel_pending()
        run_turn(intent_query(intent, topic_choice))

    elif selected_action is not None:
        if selected_action["kind"] == "intent":
            start_intent(selected_action)
        else:
            run_turn(selected_action["value"])

    elif starter_choice:
        run_turn(starter_choice)

    elif typed_query and typed_query.strip():
        text = typed_query.strip()
        if pending and pending.get("needs") == "topic":
            # A short, non-question reply is the topic; anything else is a new
            # question that escapes the clarify state.
            intent = pending["intent"]
            _cancel_pending()
            if len(text.split()) <= 6 and not text.endswith("?"):
                run_turn(intent_query(intent, text))
            else:
                run_turn(text)
        elif pending and pending.get("needs") == "attempt":
            _cancel_pending()
            run_turn(f"Please help me debug this Python code:\n{text}")
        else:
            run_turn(text)


if __name__ == "__main__":
    main()

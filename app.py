import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# from langchain_core.globals import set_verbose
import utils.chains_lcel as chains
from utils.sidebar import sidebar
import utils.llm_models as llms
from utils.utils import load_db, query_db_connection, process_and_store_query
from utils.tools import create_tool_chain
from datetime import datetime

# import langchain
# langchain.debug = False

# Initialize resources
def initialize_resources():
    # Load database and setup retriever
    retriever = load_db().as_retriever()

    # Connect to MongoDB
    mongo_db = query_db_connection()
    collection = mongo_db['Python_toolkit']
    return retriever, collection

def initialize_chains(retriever):
    """Initialize all LLM models and chains with caching."""
    try:
        
        # Setup LLM models
        gpt4o_mini = llms.openai_gpt4o_mini
        gpt4o_mini_json = llms.openai_4o_mini_json
        claude_sonnet = llms.claude_sonnet
        claude_haiku = llms.claude_haiku
        gpt4o = llms.openai_gpt4o
        
        # Create base chains
        chains_dict = {
            'rag': chains.rag_chain(claude_haiku, retriever),
            'exercise': chains.exercise_chain(claude_sonnet),
            'chat': chains.chat_chain(gpt4o_mini),
            'explain': chains.code_chain(gpt4o),
            'debug': chains.code_chain(claude_haiku),
        }
        
        # Create tool chain
        tool_chain = create_tool_chain(gpt4o_mini, chains_dict)
        
        # Return chains dictionary with tool chain
        return tool_chain, chains_dict
    
    except Exception as e:
        st.error(f"Failed to initialize chains: {str(e)}")
        raise e

def call_function(name, args: dict, chat_history, chains_dict):
    """Invoke the appropriate tool based on the name and arguments."""

    if name == "course_information":
        return chains_dict['rag'].stream(input={'chat_history': chat_history,**args})     
    if name == "explain_concept":
        return chains_dict['explain'].stream(input={'chat_history': chat_history, **args})
    if name == "generate_exercise":
        return chains_dict['exercise'].stream(input={'chat_history': chat_history, **args})
    if name == "debug_code":
        return chains_dict['debug'].stream(input=args)
    if name == "general_chat":
        return chains_dict['chat'].stream(input={**args, "chat_history": chat_history})
    else:
        return "Invalid tool name"
        
# Build an app with streamlit
def main():
    # Set the page_title
    st.set_page_config(
            page_title="🦜 GBS BUS 390 Virtual TA - Beta", page_icon="🔍", layout="wide")
    
    st.header("BUS 390 Virtual TA - Beta 🔍")
    # st.write("Currently support queries on syllabus and coding request.")
    sidebar()

    # Initialize resources
    retriever, collection = initialize_resources()
    # Initialize chains with caching
    agent, chain_dict = initialize_chains(retriever)
    
    # Initialize the query text box
    initial_text = "Hi. I'm your virtual TA Peyton. How can I help you today?"

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.chat_history.append(AIMessage(initial_text))

    # display previous conversation history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("AI", avatar="🦜"):
                st.markdown(message.content)
    
    
    
    # get user query
    if user_query := st.chat_input(initial_text):
        
        # display user query
        with st.chat_message("Human"):
            st.markdown(user_query)
        
        # save to MongoDB database
        process_and_store_query(collection, query=user_query)

        # Use tool chain to handle the query with chat history
        conversation_context = "\n".join([
            f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
            for msg in st.session_state.chat_history[-3:]  # Last 3 exchanges for context
        ])
        
        # decide which tool to call
        tool_call = agent.invoke(
            f"""
            Your only task is to decide which tool to call based on the user query delimited with <query> tags and chat history, and generate the appropriate arguements for the tool call. 
            
            Previous conversation: {conversation_context} \n
            Query: <query>{user_query}</query>\n
            Generate the tool call with appropriate arguments. Do not generate direct response. Enrich the query for the tool call when appropriate, but don't fundamentally change it. Limit query to no more than 25 tokens.
            """)
        
        print(tool_call.tool_calls)

        # call the tool
        response = call_function(
            name=tool_call.tool_calls[0]["name"],
            args=tool_call.tool_calls[0]['args'],
            chat_history=st.session_state.chat_history[-3:],
            chains_dict=chain_dict
        )
        # display AI response
        with st.chat_message("AI", avatar="🦜"):
            ai_response = st.write_stream(response)

        # append AI response to chat history
        st.session_state.chat_history.append(HumanMessage(user_query))
        st.session_state.chat_history.append(AIMessage(ai_response))

    # truncate chat history to last 5 messages
    max_num_messages = 5
    if len(st.session_state.chat_history) > max_num_messages:
        st.session_state.chat_history = st.session_state.chat_history[-max_num_messages:]
if __name__ == '__main__':
    main()

import streamlit as st
import os

def sidebar():
    with st.sidebar:
        
        st.markdown(
            "## How to use\n"
            "1. Add tick `` to enclose Python codes. Ex: `list` in Python\n"  # noqa: E501
            "2. Provide context, such as syllabus, assignments, lecture etc.\n"
            "3. Try different way to ask the questions. \n"
        )
        st.markdown("---")
        st.markdown("# About")
        st.markdown(
            '''📖 Virtual TA allows you to ask questions about course logistics, 
            as well as pass lecture contents and assignments instructions.'''
        )
        st.markdown(
            "This tool is a work in progress. "
        )
        st.markdown("Made by Dr.Wenjun Gu (wenjun.gu@emory.edu)")
       

        # api_key_input = st.text_input(
        #     "OpenAI API Key",
        #     type="password",
        #     placeholder="Paste your OpenAI API key here (sk-...)",
        #     help="You can get your API key from https://platform.openai.com/account/api-keys.",  # noqa: E501
        #     value=os.environ.get("OPENAI_API_KEY", None)
        #     or st.session_state.get("OPENAI_API_KEY", ""),
        # )

        # user_name = st.text_input(
        #     "Enter your name ",
        #     # type="password",
        #     placeholder="Your prefered name here...",
        #     help="This information will be deleted after each session",  # noqa: E501
        #     value='ISOM 352 Coder',
        # )

        # st.session_state["user_name"] = user_name

#         st.markdown("---")


#         st.markdown(
#         """
# # FAQ
# ## Best practice for interacting with virtual TA?
# Your query should provide as much information as possible to help identify the correct knowledge.  
# """
#         )

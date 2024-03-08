import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.globals import set_verbose

import chains_lcel as chains
from sidebar import sidebar

from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_rag_chain(vectorstore):
    llm = ChatOpenAI()
    
    retriever = vectorstore.as_retriever()
    
    template = """
    You are an AI paralegal assistant. Your task is to answer following query with relevant context. 
    Query: {query}
    Context: {context}
    You will use legal terms.
    """

    prompt = ChatPromptTemplate.from_template(template)
    setup_retrieval = RunnableParallel(
        {"context": retriever,
            "query": RunnablePassthrough(),
            }
    )

    chain = setup_retrieval | prompt | llm | StrOutputParser()

    return chain


def main():
    # load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")

    st.header("Chat with multiple PDFs :books:")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    # define the sidebar
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_rag_chain(vectorstore)

    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # display previous conversation history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.markdown(message.content)

    if user_query := st.chat_input("Ask a question about your documents:"):
        st.session_state.chat_history.append(HumanMessage(user_query))

        # display user query
        with st.chat_message("Human"):
            st.markdown(user_query)

        # st.markdown(f"🦜Virtual TA: I'm going to retrieve data ")
        # get response from AI
        with st.chat_message("AI"):
            rag_chain = st.session_state.conversation
            ai_response = st.write_stream(rag_chain.stream(user_query))

        # append AI response to chat history
        st.session_state.chat_history.append(AIMessage(ai_response))

    

if __name__ == '__main__':
    main()
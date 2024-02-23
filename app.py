import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.globals import set_verbose
import chains_lcel as chains
from sidebar import sidebar

# Enable verbose logging
set_verbose(True)

# Set the page_title
st.set_page_config(
        page_title="🦜 GBS BUS 390 Virtual TA - Beta", page_icon="🔍")

# cache the vectorized embedding database 
@st.cache_resource
# load the vectorized database
def load_db(db_path, embedding_model='text-embedding-ada-002'):
    embeddings = OpenAIEmbeddings(model=embedding_model, chunk_size=1)
    db_loaded = FAISS.load_local(db_path, embeddings)
    
    return db_loaded

# 1. Load the Vectorised database
kb_db_path = 'data/emb_db'
db = load_db(kb_db_path)

# 2. Function for similarity search
retriever = db.as_retriever()

# 3. Setup LLM and chains
llm = ChatOpenAI(temperature=0.1, 
                #  model="gpt-4-0125-preview",
                 model="gpt-3.5-turbo-1106",
                 verbose=False,
                 max_tokens=300,
                 )

# 3a. Setup query router
router_chain = chains.router_chain(llm)

def router_choice(query, chain):
    choice = chain.invoke(input={'query': query})
    return int(choice)

# 3b. Setup LLMChain & prompts for RAG answer generation
rag_chain = chains.rag_chain(llm, retriever)

# 3c. Setup direct openai_chain
openai_chain = chains.openai_chain(llm)

# 4. generate response based on router choice
def generate_response(query, choice):
    if choice == 1: # LLM decides to use OpenAI directly
        decision = "use OpenAI directly"
        st.markdown(f"🦜Virtual TA: I'm going to {decision} 🐍")
        # with st.spinner(f"Generating answers..."): 
        response = openai_chain.stream(input=query)
    else: 
        decision = "get more information" # LLM router to RAG
        st.markdown(f"🦜Virtual TA: I need to {decision} 🔍")
        # with st.spinner(f"Generating answers..."): 
        response = rag_chain.stream(input=query)
    
    return response
        
# 5. Build an app with streamlit
def main():

    st.header("BUS 390 Virtual TA - Beta 🔍")
    # st.write("Currently support queries on syllabus and coding request.")
    sidebar()
    query = st.text_area(label="Query on syllabus and Python coding support.",
                           placeholder="How can I help you today?", 
                           max_chars=200)
        
    if query:
        # first decide course of action
        choice = router_choice(query, router_chain)

        # Get response
        response = generate_response(query, choice)

        # Write response
        st.write(response)
        
        # st.expander("sources").info(source)

        if 'last_query' in st.session_state: 
            st.markdown("---") 
            st.markdown(f'''Previous query: {st.session_state.last_query}''')
            # st.write(f'''Response: {st.session_state.last_result}''')
        
        st.session_state['last_query'] = query
        st.session_state['last_result'] = response

if __name__ == '__main__':
    main()

# In this file, all chains are defined with LC Expression Language 
# Doing so alone streaming of the outupt
# Created 2/21/2024
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from operator import itemgetter

output_parser = StrOutputParser()


# define the router chain
def router_chain(llm):
    query_router_template = """
    You are an AI query router for a coding course in business school. 
    The following is a user query: {query}. Based on the content of this query, determine its category according to the guidelines provided:

    - If the query is about the chat history, classify it as 0.
    - If the query requires specific knowledge, such as syllabus, assignments, lectures, classify it as 2.
    - For other queries including coding in Python, including syntax, libraries, and programming concepts, classify it as 1.

    Output the classification number without any additional text or explanation.
    """

    router_prompt = ChatPromptTemplate.from_template(query_router_template)
    setup = RunnableParallel(
        {"query": RunnablePassthrough()}
    )
    router_chain = setup | router_prompt | llm | output_parser

    return router_chain

# define the openai chain
def openai_chain(llm):
    query_template = """
    I am a virtual teaching assistant for an introductory Python class at Goizueta Business School. 
    My task is to answer student query to the best capacity. 
    My response should be concise, helpful and to the point. I will use business examples and analogies when appropriate.

    Answer the user query: {query} 
    Consider teh chat history: {chat_history}
    """

    openai_prompt = ChatPromptTemplate.from_template(query_template)

    chain = openai_prompt | llm | output_parser

    return chain

# 3b. Setup LLMChain & prompts for RAG answer generation
def rag_chain(llm, retriever):
    template = """
    You are a virtual TA for question-answering tasks. 
    Your task is to answer following query with relevant context. 
    Query: {query}
    Context: {context}
    You will use analogies, and refer to the virtual TA in first-person persona.
    """
    # 
    # Please generate an appropriate response. Format the output when possible. 

    prompt = ChatPromptTemplate.from_template(template)
    setup_retrieval = RunnableParallel(
        {"context": retriever,
         "query": RunnablePassthrough(),
         }
    #     {
    #     "context": itemgetter("query") | retriever,
    #     "query": itemgetter("query"),
    #     "chat_history": itemgetter("chat_history"),
    # }
    )

    chain = setup_retrieval | prompt | llm | output_parser

    return chain

# define chat history chain
# 3d. Setup LLMChain & prompts for RAG answer generation
def chat_history_chain(llm):
    template = """
    You're my AI assistant that answer queries based on chat hisotry. 
    Your response should be direct, concise and helpful.
    Answer the user query: {query} 
    Here is the chat history: {chat_history}
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | llm | output_parser

    return chain
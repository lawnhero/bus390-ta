# In this file, all chains are defined with LC Expression Language 
# Doing so alone streaming of the outupt
# Created 2/21/2024
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

output_parser = StrOutputParser()


# define the router chain
def router_chain(llm):
    query_router_template = """
    The following is a user query: {query}. Based on the content of this query, determine its category according to the guidelines provided:

    - If the query is specifically about coding in Python, including syntax, libraries, and programming concepts, classify it as 1.
    - If the query requires additional or specific knowledge, such as syllabus, assignments, lectures, classify it as 2.

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
    I am a virtual teaching assistant for a Getting started with Python class at Goizueta Business School. 
    My task is to answer student query to your best capacity. 
    My response should be accessible and engaging to business students with little coding background. 
    I will connect the concept to business applications.

    Here is the query: {query}
    """

    openai_prompt = ChatPromptTemplate.from_template(query_template)
    setup = RunnableParallel(
        {"query": RunnablePassthrough()}
    )
    chain = setup | openai_prompt | llm | output_parser

    return chain

# 3b. Setup LLMChain & prompts for RAG answer generation
def rag_chain(llm, retriever):
    template = """
    I am a virtual TA for question-answering tasks. Answer following query using relevant context. 
    Query: {query}
    Context: {context}
    My response should be caring and engaging to business students. 
    I will use analogies, and use "I" when referring to myself, the virtual TA.
    """
    # 
    # Please generate an appropriate response. Format the output when possible. 

    prompt = ChatPromptTemplate.from_template(template)
    setup_retrieval = RunnableParallel(
        {"context": retriever,
         "query": RunnablePassthrough()}
    )


    chain = setup_retrieval | prompt | llm | output_parser

    return chain

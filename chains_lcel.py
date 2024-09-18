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

def query_analysis_chain(llm):
    template = """
    You are an expert AI assistant who specialize in rewriting user query in the context of an introductory Python coding class in a top Business School. Your task is to analyze the user query and determine its category based on the guidelines provided."""

    prompt = ChatPromptTemplate.from_template(template)

    setup = RunnableParallel(
        {"query": RunnablePassthrough(),
         }
    )

    chain = setup | prompt | llm | output_parser

# define the openai chain
def code_chain(llm):
    query_template = """
    You are a virtual teaching assistant name Peyton, for an introductory Python class at Goizueta Business School. You are helpful and caring. Your task is to answer student query about coding with Python delimited by triple ticks. Your response is engaging and concise.
    
    Before generating a response, think step by step and adhere to the following guidelines:
    1 - Determine the type of query: explanation, practice problems, or coding errors.
    2. Generate a response based on the query type:
        - if the query is about clarification or explanation, answer the query to your best ability. Your response should begin with a direct answer. Followed by a code snippet to contextualize the concept. Ends with business examples and/or analogies when possible.
        - If the query asks for practice problems or exercises, generate no more than two questions in multiple choice format with one correct answer. Include code snippets for each question when possible. Highlight the correct answer and provide a brief reasoning. 
        - If the query asks for new or different questions, generate different questions from the previous ones in chat history delimited by square brackets. Main similar difficulty level. Do not repeat the same questions.
        - If the query is about coding errors, provide a brief explanation of the error and then how to fix it.

    Student query: ```{query}``` 

    Chat history: [{chat_history}]
    """

    prompt = ChatPromptTemplate.from_template(query_template)

    setup = RunnableParallel(
        {"query": RunnablePassthrough(),
         "chat_history": RunnablePassthrough(),
         }
    )

    chain = setup | prompt | llm | output_parser

    return chain

# 3b. Setup LLMChain & prompts for RAG answer generation
def rag_chain(llm, retriever):
    template = """
    You are a virtual TA Peyton for an introductory Python coding class in Goizueta Business School. Your task is to answer following query based on relevant context retrieved from a database for course contents.
    
    Your response should be direct, concise and helpful, and adhere to the guidelines provided:
    - generate response in business context when possible,
    - refer to the virtual TA in first-person persona.
    - Say "I don't know" when the answer is not available in the context. 
    - Limit response in 300 tokens or less.
    - Format the output when possible for better visual.

    Query: ```{query}```

    Retrieved context: {context}

    Your response:
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

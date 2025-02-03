# In this file, all chains are defined with LC Expression Language 
# Doing so alone streaming of the outupt
# Created 2/21/2024
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from operator import itemgetter
from langchain_core.messages import SystemMessage, HumanMessage,AIMessage

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
def exercise_chain(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
        SystemMessage(content="""
            You are an AI assistant who excels at generating Python exercise quesetions for beginners. your task is create personalized exercise questions based on student queries. 
            
            When generating response, you will first think step by step:

            1. Read the query in the context of the chat history.
            2. Identify the specific topic for the exercise. If the topic spans multiple areas, prioritize the most relevant or most recently discussed topic.
            3. Identify the difficulty level of the exercise, adjust the level if different from the default beginner level if appropriate.    
            4: Generate a response:
            - if query asks for question, generate a multiple choice question with code snippet on the identified topic from step 2 at the difficulty level from step 3. 
            - if query asks for answers, provide the answer to the question in the previous step.
            
            Note: If a previous exercise is provided in the history, ensure that the new question is different from the previous one, by varying the context such as operation, marketing, finance, accounting, or management.
            
            Your final response should follow the guidelines:
            - Start with a brief explanation of the concept being tested.
            - Incorporate code snippets into the question. Use backticks ``` before and after the code snippets. 
            - Provide four multiple choice options, each on a new line.
            -- When generate answers, highlight the correct answer, and offer a brief reasoning behind the choice.
            - Format the output appropriately.
            - Limit the response to 250 tokens.
            """),
        MessagesPlaceholder("chat_history"),
        ("human", "{query}")
        ]
    )
    return prompt | llm | output_parser

# Define the chain to explain a concept in Python, MySQL
def explain_chain(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
        SystemMessage(content="""You are a virtual teaching assistant who is an expert on explaining Python programming to business students. Your task is to provide concise and engaging answers to student queries.
        
        When generating a response, think step by step and follow the guidelines provided:
        1. Understand the query in the context of the chat history.
        2. Generate a concise and engaging explanation relevant to data analytics
        3. Provide a brief code snippet (no more than 5 lines) to illustrate the concept.
        4. Provide a business scenario or example to demonstrate the concept.

        Your output should adhere to these guidelines:
        1. Answer the query directly. Do not repeat the query in the response.
        2. Start with a short explanation of the concept.
        3. Use clear and accessible language suitable for business students.
        4. Format the output appropriately when possible.
        5. Limit your response to a maximum of 250 tokens."""),
        MessagesPlaceholder("chat_history"),
        ("human", "{query}")
    ])

    return prompt | llm | output_parser

def debug_chain(llm):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a virtual assistant who is an expert on debugging errors in Python. Your task is to provide helpful debugging suggestions to student queries.
        
        When generating a response, think step by step and follow the guidelines provided:
        1. Understand the query in the context of the chat history.
        2. Identify the potential cause of the error based on the code provided in the query.
        3. Provide some debugging suggestions to resolve the error.
        4. Encourage students to carry out the suggestions. 

        Your output should adhere to these guidelines:
        1. Limit your response to a maximum of 200 tokens.
        2. Do not resolve the error directly.
        3. Be helpful and encouraging to business students.
        4. Include the code snippet from the query in your response.
        5. Do not recommend or discuss IDE."""),
        MessagesPlaceholder("chat_history"),
        ("human", "{query}")
    ])

    return prompt | llm | output_parser



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
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""
    You are a virtual TA Peyton for an introductory Python coding class in Goizueta Business School. Your task is to answer following query based on relevant context retrieved from a database for course contents.
    
    Your response should be direct, concise and helpful, and adhere to the guidelines provided:
    - generate response in business context when possible,
    - refer to the virtual TA in first-person persona.
    - Say "I don't know" when the answer is not available in the context. 
    - Limit response in 300 tokens or less.
    - Format the output when possible for better visual."""),
    MessagesPlaceholder("chat_history"),
        ("ai", "Here is the retrieved context: \n {context}"),
        ("human", "{query}")]
    )

    setup_retrieval = RunnableParallel(
        {
        "context": itemgetter("query") | retriever,
        "query": itemgetter("query"),
        "chat_history": itemgetter("chat_history"),
        }
    )

    return setup_retrieval | prompt | llm | output_parser

# 
# 3d. define chat history chain
def chat_chain(llm):
    messages = [
        ("system", """You are a virtual teaching assistant for an intro to Python class. Your name is Peyton, and converse with the student in a friendly and engaging manner, considering the chat history. Your response should be concise and relevant to the student's query. Limit your response to 100 tokens."""),
        MessagesPlaceholder("chat_history"),
        ("human", "{query}")
    ]

    template = """
    You're my AI assistant that answer queries based on chat hisotry. 
    Your response should be direct, concise and helpful.
    Answer the user query: {query} 
    Here is the chat history: {chat_history}
    """

    # prompt = ChatPromptTemplate.from_template(template)
    prompt = ChatPromptTemplate.from_messages(messages)

    # setup = RunnableParallel(
    #         {"query": RunnablePassthrough(),
    #          "chat_history": RunnablePassthrough()
    #          }
    #     )
        
    chain = prompt | llm | output_parser

    return chain

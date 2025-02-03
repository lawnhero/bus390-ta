"""
Tools and Agent implementation for the Virtual TA system.
"""

from langchain.tools import StructuredTool
from typing import Optional, List
from langchain.schema.language_model import BaseLanguageModel
from pydantic import BaseModel, Field
import streamlit as st

# Tool argument schemas
class RagArgs(BaseModel):
    query: str = Field(..., description="The user query to get course information for")

class ExplainArgs(BaseModel):
    query: str = Field(..., description="The query to explain")
    # previous_query: Optional[str] = Field(default="", description="Previous query for context")

class ExerciseArgs(BaseModel):
    query: str = Field(..., description="The query to generate exercise or answer for")
    # previous_query: str = Field(default="", description="Previous query for context")
    # skill_level: str = Field(default="beginner", description="Student's skill level")
    # previous_exercise: str = Field(default="", description="Previous exercise for context")

class AnalyticsArgs(BaseModel):
    query: str = Field(..., description="The query about data analytics")
    # chat_history: str = Field(default="", description="Previous chat history for context")

class DebugArgs(BaseModel):
    query: str = Field(..., description="The code or error to debug in the query")
    # chat_history: str = Field(default="", description="Previous chat history for context")

class ChatArgs(BaseModel):
    query: str = Field(..., description="The message to respond to")
    

# Tool definitions
def rag_tool(chain):
    """Tool for retrieving course-related information."""
    def _run(args: RagArgs) -> str:
        return chain.stream(args.query)
    
    return StructuredTool(
        name="course_information",
        description="Use this tool for course-specific information, ex: instructor, syllabus, policies or assignments.",
        func=_run,
        args_schema=RagArgs
    )

def explain_tool(chain):
    """Tool for explaining technical concepts."""
    def _run(args: ExplainArgs) -> str:
        return chain.invoke({"query": args.query, "chat_history": args.previous_query})
    
    return StructuredTool(
        name="explain_concept",
        description="Use this tool when the query asks for explanation of Python or other concepts.",
        func=_run,
        args_schema=ExplainArgs
    )

def exercise_tool(chain):
    """Tool for generating exercise questions."""
    def _run(args: ExerciseArgs) -> str:
        return chain.invoke({
            "current_query": args.query,
            "previous_query": args.previous_query,
            "skill_level": args.skill_level,
            "previous_exercise": args.previous_exercise
        })
    
    return StructuredTool(
        name="generate_exercise",
        description="Use this tool when the query asks for Python practice exercises or check answers.",
        func=_run,
        args_schema=ExerciseArgs
    )

def analytics_tool(chain):
    """Tool for data analytics explanations."""
    def _run(args: AnalyticsArgs) -> str:
        return chain.invoke({
            "query": args.query,
            "chat_history": args.chat_history
        })
    
    return StructuredTool(
        name="explain_analytics",
        description="Use this tool when the query is about data analysis with pandas, matplotlib, seaborn, or statistics.",
        func=_run,
        args_schema=AnalyticsArgs
    )

def debug_tool(chain):
    """Tool for debugging assistance."""
    def _run(args: DebugArgs) -> str:
        return chain.invoke({
            "query": args.query,
            "chat_history": args.chat_history
        })
    
    return StructuredTool(
        name="debug_code",
        description="Use this tool when the query is about code errors or debugging help.",
        func=_run,
        args_schema=DebugArgs
    )

def chat_tool(chain):
    """Tool for general conversation."""
    def _run(args: ChatArgs) -> str:
        
        print("--"*10, "inside the chat tool")
        print(args.query)
        return chain.invoke({
            "query": args,
            "chat_history": "Hi. I'm your virtual TA Peyton. How can I help you today?"
        })
    
    return StructuredTool(
        name="general_chat",
        description="Use this tool for general conversation or when no other tools are appropriate.",
        func=_run,
        args_schema=ChatArgs
    )

def create_tool_chain(llm: BaseLanguageModel, chain_dict: dict):
    """Create a tool-enabled LLM chain."""
    tools = [
        rag_tool(chain_dict['rag']),
        explain_tool(chain_dict['explain']),
        exercise_tool(chain_dict['exercise']),
        debug_tool(chain_dict['debug']),
        chat_tool(chain_dict['chat'])
    ]
    
    # Bind tools to LLM with system message
    return llm.bind_tools(tools, tool_choice="any")


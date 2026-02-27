from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import streamlit as st
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime
import os
from dotenv import load_dotenv
import certifi

# Load environment variables from .env file
load_dotenv()

# knowledge base path - ChromaDB uses persist_directory instead of db_path
kb_db_path = 'data/chroma_db'
MONGODB_PASSWORD = os.getenv("MONGODB_PASSWORD")

if not MONGODB_PASSWORD:
    raise ValueError("MONGODB_PASSWORD environment variable is not set. Please check your .env file.")



@st.cache_resource
# load the vectorized database
def load_db(db_path=kb_db_path, embedding_model='text-embedding-ada-002'):
    """
    Load the ChromaDB vector database.
    
    ChromaDB stores data in SQLite and does not use pickle serialization,
    making it safer than FAISS for production use.
    """
    embeddings = OpenAIEmbeddings(model=embedding_model, chunk_size=1)
    db_loaded = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )
    print("Database loaded")
    return db_loaded


uri = f"mongodb+srv://streamlit_app:{MONGODB_PASSWORD}@virtual-ta.q344d.mongodb.net/myFirstDatabase?retryWrites=true&w=majority"

# MongoDB Atlas connection
@st.cache_resource
def query_db_connection():
    """
    Return a MongoDB connection to the user_queries_db database.
    
    Uses proper SSL/TLS certificate validation with certifi for security.
    """
    client = MongoClient(
        uri, 
        server_api=ServerApi('1'),
        tlsCAFile=certifi.where()  # Use certifi's CA bundle for proper SSL verification
    )
    print("Connected to MongoDB")
    return client['user_queries_db']


# function to store the query in the database
def process_and_store_query(collection, **kwargs):
    """Insert a query into the MongoDB collection."""
    # Create a document to insert
    document = {
        "timestamp": datetime.now()
    }
    # add any additional fields to the document
    document.update(kwargs)
    
    # Insert the document into the collection
    result = collection.insert_one(document)
    
    return result.inserted_id
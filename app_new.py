import os
import streamlit as st
from google.cloud import storage
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sqlite3

# Load environment variables from a .env file
load_dotenv()

# Now you can access the API key securely
api_key = os.getenv("OPENAI_API_KEY")

# Ensure the API key is set
if not api_key:
    st.error("The OpenAI API key is not set. Please add it to your .env file.")
    st.stop()

# Define your paths
CHROMA_GCS_PATH = 'esg_rag/chroma_10k_reports_sample/chroma.sqlite3'  # Path in GCS
CHROMA_LOCAL_PATH = '/tmp/chroma_10k_reports_sample/chroma.sqlite3'  # Local path

# Google Cloud Storage bucket name
BUCKET_NAME = "esg_rag"

# Download file from GCS bucket
def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    
    # Download the file to a local destination
    blob.download_to_filename(destination_file_name)
    st.write(f"Downloaded {source_blob_name} to {destination_file_name}.")

# Download the Chroma SQLite database from GCS to local storage
os.makedirs(os.path.dirname(CHROMA_LOCAL_PATH), exist_ok=True)
download_blob(BUCKET_NAME, CHROMA_GCS_PATH, CHROMA_LOCAL_PATH)

# Set the local path where Chroma should read files from
CHROMA_PATH = os.path.dirname(CHROMA_LOCAL_PATH)

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Streamlit application
def main():
    st.title("Climate Knowledge Base")

    # Input for the query
    query_text = st.text_input("Enter your query", placeholder="What climate change risks are mentioned in the report?")

    # Button to trigger the query
    if st.button("Submit Query") and query_text:
        with st.spinner("Processing your query..."):
            response, sources = get_response(query_text)

            if response:
                st.subheader("Response")
                st.write(response)
                
                st.subheader("Sources")
                st.write(sources)
            else:
                st.error("Unable to find relevant results.")
    else:
        st.info("Please enter a query and press 'Submit'.")

# Function to get response from Chroma database
def get_response(query_text):
    # Initialize OpenAI Embeddings and Model with API Key
    embedding_function = OpenAIEmbeddings(openai_api_key=api_key)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the database for the query
    results = db.similarity_search_with_relevance_scores(query_text, k=10)
    if len(results) == 0 or results[0][1] < 0.7:
        return None, None

    # Combine the retrieved contexts
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Use OpenAI to generate an answer based on the context
    model = ChatOpenAI(openai_api_key=api_key)
    response_text = model.predict(prompt)

    # Retrieve document sources
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    
    return response_text, sources

if __name__ == "__main__":
    main()

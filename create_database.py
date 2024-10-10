# create_database.py
from langchain.document_loaders import DirectoryLoader  # Corrected import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma  # Corrected import
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
import openai
import os
import shutil
from dotenv import load_dotenv

# Load environment variables (assumes .env file with API keys)
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')


#CHROMA_PATH = 'tbd'

#DATA_PATH = 'tbd'


def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    # List all the text files in the directory
    all_files = [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) if f.endswith('.txt')]
    
    # Filter files that include '10-K' and exclude '10-Q'
    filtered_files = [f for f in all_files if '10-K' in f and '10-Q' not in f]
    # Limit to the first 100 files
    filtered_files = filtered_files[:100]
    
    documents = []
    
    # Load each filtered file using the TextLoader
    for file_path in filtered_files:
        print(f"Loading file: {file_path}")
        loader = TextLoader(file_path)  # Use TextLoader for each file
        documents += loader.load()  # Append loaded documents to the list
    
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Adjust as needed
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks):
    # Clear out the previous Chroma database, if it exists.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new Chroma database from the chunks.
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    main()

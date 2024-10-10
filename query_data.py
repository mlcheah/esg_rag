import argparse
import os
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables from a .env file (if exists)
load_dotenv()

# Now you can access the API key securely
api_key = os.getenv("OPENAI_API_KEY")

# Ensure the API key is set
if not api_key:
    raise ValueError("The OpenAI API key is not set. Please add it to your .env file.")

CHROMA_PATH = 'tbd'

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    # Create CLI to accept query as argument
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text (e.g., 'What climate change risks are mentioned in the report?').")
    args = parser.parse_args()
    query_text = args.query_text

    # Initialize OpenAI Embeddings and Model with API Key
    embedding_function = OpenAIEmbeddings(openai_api_key=api_key)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the database for the query
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print("Unable to find relevant results.")
        return

    # Combine the retrieved contexts
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(f"Prompt sent to OpenAI: {prompt}")

    # Use OpenAI to generate an answer based on the context
    model = ChatOpenAI(openai_api_key=api_key)
    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

if __name__ == "__main__":
    main()

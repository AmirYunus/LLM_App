#!/usr/bin/env python3

import os
import time
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
from constants import CHROMA_SETTINGS

# Load environment variables from a .env file
load_dotenv()

def main() -> None:
    """
    The main function that orchestrates the execution of the application.
    Retrieves configuration from environment variables, initializes components,
    and runs an interactive loop for user queries.

    Returns:
        None
    """
    # Configuration parameters loaded from environment variables
    embeddings_model_name: str = os.getenv("EMBEDDINGS_MODEL_NAME")
    persist_directory: str = os.getenv('PERSIST_DIRECTORY')
    model_type: str = os.getenv('MODEL_TYPE')
    model_path: str = os.getenv('MODEL_PATH')
    model_n_ctx: str = os.getenv('MODEL_N_CTX')
    model_n_batch: int = int(os.getenv('MODEL_N_BATCH', 8))
    target_source_chunks: int = int(os.getenv('TARGET_SOURCE_CHUNKS', 4))

    # Initialize HuggingFace embeddings model
    embeddings: HuggingFaceEmbeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    # Initialize Chroma vector store
    db: Chroma = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    
    # Create a retriever using Chroma with specified search parameters
    retriever: RetrievalQA = db.as_retriever(search_kwargs={"k": target_source_chunks})

    # Initialize an empty list for callbacks (none provided in the original code)
    callbacks: List[StreamingStdOutCallbackHandler] = []

    # Prepare the Language Model (LLM) based on the specified type
    if model_type == "LlamaCpp":
        llm: LlamaCpp = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
    elif model_type == "GPT4All":
        llm: GPT4All = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
    else:
        raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")

    # Create a RetrievalQA instance with the specified LLM, chain type, and retriever
    qa: RetrievalQA = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

    # Interactive questions and answers loop
    while True:
        query: str = input("\nEnter a query: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        # Get the answer from the chain
        res: Dict[str, Any] = qa(query)
        docs: List[Any] = res['source_documents']

        # Print the relevant sources used for the answer
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)
            break

if __name__ == "__main__":
    main()
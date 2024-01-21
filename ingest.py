#!/usr/bin/env python3

import os
import glob
from typing import List, Tuple
from dotenv import load_dotenv
from multiprocessing import Pool
from tqdm import tqdm
from langchain.document_loaders import (
    CSVLoader, UnstructuredWordDocumentLoader, EverNoteLoader, UnstructuredEmailLoader,
    UnstructuredEPubLoader, UnstructuredHTMLLoader, UnstructuredMarkdownLoader,
    UnstructuredODTLoader, PyMuPDFLoader, UnstructuredPowerPointLoader, TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from constants import CHROMA_SETTINGS

# Load environment variables from a .env file
load_dotenv()

# Retrieve environment variables
persist_directory: str = os.environ.get('PERSIST_DIRECTORY')
source_directory: str = os.environ.get('SOURCE_DIRECTORY', 'documents')
embeddings_model_name: str = os.environ.get('EMBEDDINGS_MODEL_NAME')
chunk_size: int = 500
chunk_overlap: int = 50

# Mapping file extensions to document loaders and their arguments
LOADER_MAPPING: dict = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (UnstructuredEmailLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}

class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc: List[Document] = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"] = "text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc

def load_single_document(file_path: str) -> List[Document]:
    """
    Load a single document based on its file extension using the appropriate loader.

    Args:
        file_path (str): Path to the document file.

    Returns:
        List[Document]: List of documents loaded from the file.
    """
    ext: str = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader: loader_class = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")

def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Load all documents from the source directory, excluding specified files.

    Args:
        source_dir (str): Path to the source directory.
        ignored_files (List[str]): List of files to be ignored.

    Returns:
        List[Document]: List of loaded documents.
    """
    # Collect all files based on supported extensions
    all_files: List[str] = []
    for ext in LOADER_MAPPING:
        all_files.extend(glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True))
    # Filter files to exclude those in the ignored_files list
    filtered_files: List[str] = [file_path for file_path in all_files if file_path not in ignored_files]

    # Use multiprocessing to load documents in parallel
    with Pool(processes=os.cpu_count()) as pool:
        results: List[Document] = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                results.extend(docs)
                pbar.update()

    return results

def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Load and split documents into chunks.

    Args:
        ignored_files (List[str]): List of files to be ignored.

    Returns:
        List[Document]: List of documents after splitting into chunks.
    """
    print(f"Loading documents from {source_directory}")
    documents: List[Document] = load_documents(source_directory, ignored_files)
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    # Use RecursiveCharacterTextSplitter to split documents into chunks
    text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts: List[Document] = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    return texts

def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Check if vectorstore exists.

    Args:
        persist_directory (str): Path to the vectorstore directory.

    Returns:
        bool: True if vectorstore exists, False otherwise.
    """
    if os.path.exists(os.path.join(persist_directory, 'index')):
        if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(os.path.join(persist_directory, 'chroma-embeddings.parquet')):
            list_index_files: List[str] = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
            list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
            # At least 3 documents are needed in a working vectorstore
            if len(list_index_files) > 3:
                return True
    return False

def main() -> None:
    """
    The main function that orchestrates the ingestion process.
    """
    embeddings: HuggingFaceEmbeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    if does_vectorstore_exist(persist_directory):
        # Update and store locally vectorstore
        print(f"Appending to existing vectorstore at {persist_directory}")
        db: Chroma = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
        collection: Tuple[Document, ...] = db.get()
        texts: List[Document] = process_documents([metadata['source'] for metadata in collection['metadatas']])
        print(f"Creating embeddings. May take some minutes...")
        db.add_documents(texts)
    else:
        # Create and store locally vectorstore
        print("Creating new vectorstore")
        texts: List[Document] = process_documents()
        print(f"Creating embeddings. May take some minutes...")
        db: Chroma = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None

    print(f"Ingestion complete!")

if __name__ == "__main__":
    main()
import os
from dotenv import load_dotenv
from chromadb.config import Settings

def load_environment_variables() -> None:
    """
    Load environment variables from a .env file.
    """
    load_dotenv()

# Load environment variables from a .env file
load_environment_variables()

# Define the folder for storing the database
PERSIST_DIRECTORY: str = os.environ.get('PERSIST_DIRECTORY')
"""
The path to the directory where the Chroma database will be stored.

:type: str
"""

# Define the Chroma settings
CHROMA_SETTINGS: Settings = Settings(
    chroma_db_impl='duckdb+parquet',  # Database implementation using DuckDB and Parquet
    persist_directory=PERSIST_DIRECTORY,  # Directory path where the Chroma database will be stored
    anonymized_telemetry=False  # Disable anonymized telemetry for Chroma
)
"""
The settings for the Chroma database.

:type: Settings
"""

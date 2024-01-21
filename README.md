# LLM App

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Generic badge](https://img.shields.io/badge/Python-Passed:_3.10.13-Green.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Win_OS-Untested:_Win_11_(22H2)-orange.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Mac_OS-Passed:_Sonoma_14.2_(M1)-Green.svg)](https://shields.io/)

Welcome to the LLM App repository! This GitHub project serves as a template for students undertaking their capstone projects, incorporating a Large Language Model (LLM) with a Retrieval-Augmented Generation (RAG) model. Follow the instructions below to set up the development environment and run the LLM App.

## Setup Instructions

1. **Install Anaconda / Miniconda:**

   Ensure that Anaconda or Miniconda is installed on your system.

2. **Create a Virtual Environment:**

   Open your terminal and create a virtual environment using the following command:

   ```bash
   conda create --prefix=venv python=3.10 -y
   ```

3. **Activate the Environment:**

   Activate the virtual environment:

   ```bash
   conda activate ./venv
   ```

4. **Install Dependencies:**

   Install the required dependencies using pip:

   ```bash
   python -m pip install -r requirements.txt
   ```

5. **Download the Language Model:**
   
   Download the LLM (Large Language Model) from https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin.

6. **Place the Downloaded Model:**
   
   Put the downloaded LLM Model into the `models` folder.

7. **Add Files:**
   
   Place the files you want to process in the `documents` folder.

   The supported extensions are:
   - `.csv`: CSV,
   - `.docx`: Word Document,
   - `.enex`: EverNote,
   - `.eml`: Email,
   - `.epub`: EPub,
   - `.html`: HTML File,
   - `.md`: Markdown,
   - `.msg`: Outlook Message,
   - `.odt`: Open Document Text,
   - `.pdf`: Portable Document Format (PDF),
   - `.pptx` : PowerPoint Document,
   - `.txt`: Text file (UTF-8),

## Running the Application

1. **Run the Ingestion Script:**
   
   Run the ingestion script to process and load the documents:

   ```bash
   python ingest.py
   ```

2. **Launch the App:**
   
   Finally, run the LLM App:

   ```bash
   python app.py
   ```

   Feel free to explore and customise this template for your capstone project. Happy coding!